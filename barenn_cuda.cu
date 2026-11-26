#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace Barenn {

// Cada imagem tem 64 atributos e pertence a uma das dez classes.
constexpr int entradas = 64;
constexpr int classes = 10;

// Referência CPU usada apenas pelo modo --cpu e pela validação.
class DenseLayerCPU {
    int in_features;
    int out_features;
    bool final;
    // weights[i][j] liga a entrada j ao neurônio de saída i.
    vector<vector<float>> weights;
    vector<float> bias;
    // Dados do último forward usados para calcular os gradientes.
    vector<float> last_input;
    vector<float> last_z;

public:
    DenseLayerCPU(int input_size, int output_size, bool is_final = false)
        : in_features(input_size), out_features(output_size), final(is_final),
          weights(output_size, vector<float>(input_size)), bias(output_size, 0) {
        // A semente fixa torna a inicialização de cada camada reproduzível.
        mt19937 gen(1337);
        uniform_real_distribution<float> distribution(-0.05f, 0.05f);
        for (auto& row : weights) {
            for (float& weight : row) {
                weight = distribution(gen);
            }
        }
    }

    vector<float> Forward(vector<float> input) {
        last_input = input;
        last_z.resize(out_features);
        vector<float> out(out_features);
        // Cada neurônio soma o viés aos produtos entre entradas e pesos.
        for (int i = 0; i < out_features; ++i) {
            float sum = bias[i];
            for (int j = 0; j < in_features; ++j) {
                sum += input[j] * weights[i][j];
            }
            last_z[i] = sum;
            // As camadas ocultas usam ReLU. A última devolve escores lineares.
            out[i] = final ? sum : max(0.0f, sum);
        }
        return out;
    }

    // Inferência com pesos fixos: não altera os dados usados pelo treinamento.
    vector<float> Predict(const vector<float>& input) const {
        vector<float> out(out_features);
        for (int i = 0; i < out_features; ++i) {
            float sum = bias[i];
            for (int j = 0; j < in_features; ++j) {
                sum += input[j] * weights[i][j];
            }
            out[i] = final ? sum : max(0.0f, sum);
        }
        return out;
    }

    vector<float> Parametros() const {
        vector<float> valores;
        for (const auto& linha : weights) {
            valores.insert(valores.end(), linha.begin(), linha.end());
        }
        valores.insert(valores.end(), bias.begin(), bias.end());
        return valores;
    }

    vector<float> Backward(vector<float> grad_out, float learning_rate) {
        // Delta é o gradiente da perda em relação à soma antes da ativação.
        // Na ReLU, o gradiente passa apenas quando essa soma é positiva.
        vector<float> delta(out_features);
        for (int i = 0; i < out_features; ++i) {
            delta[i] = final ? grad_out[i]
                             : grad_out[i] * (last_z[i] > 0.0f ? 1.0f : 0.0f);
        }

        vector<float> grad_in(in_features);
        int inicio = 0;

        for (; inicio + 8 <= in_features; inicio += 8) {
            float somas[8] = {};

            for (int j = 0; j < out_features; ++j) {
                const float d = delta[j];
                const auto& linha = weights[j];

                for (int k = 0; k < 8; ++k) {
                    somas[k] += d * linha[inicio + k];
                }
            }

            for (int k = 0; k < 8; ++k) {
                grad_in[inicio + k] = somas[k];
            }
        }

        // Trata posições restantes, caso existam.
        for (; inicio < in_features; ++inicio) {
            float sum = 0.0f;

            for (int j = 0; j < out_features; ++j) {
                sum += delta[j] * weights[j][inicio];
            }

    grad_in[inicio] = sum;
}

        // SGD por amostra. O gradiente do peso é delta vezes a entrada.
        // O gradiente do viés é o próprio delta.
        for (int i = 0; i < out_features; ++i) {
            for (int j = 0; j < in_features; ++j) {
                weights[i][j] -= learning_rate * (delta[i] * last_input[j]);
            }
            bias[i] -= learning_rate * delta[i];
        }
        return grad_in;
    }
};


void verificar_cuda(cudaError_t erro, const char* chamada, int linha) {
    if (erro != cudaSuccess) {
        throw runtime_error(string("CUDA na linha ") + to_string(linha) +
                            " (" + chamada + "): " + cudaGetErrorString(erro));
    }
}

#define CUDA_CHECK(chamada) verificar_cuda((chamada), #chamada, __LINE__)

// Buffers persistentes por camada. O destrutor libera a memória da GPU.
template<class T>
class BufferGPU {
    T* ponteiro = nullptr;
    size_t bytes;
public:
    explicit BufferGPU(size_t elementos) : bytes(elementos * sizeof(T)) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ponteiro), bytes));
    }
    ~BufferGPU() { if (ponteiro) cudaFree(ponteiro); }
    BufferGPU(const BufferGPU&) = delete;
    BufferGPU& operator=(const BufferGPU&) = delete;
    T* data() const { return ponteiro; }
    void enviar(const T* origem) {
        CUDA_CHECK(cudaMemcpy(ponteiro, origem, bytes, cudaMemcpyHostToDevice));
    }
    void receber(T* destino) const {
        CUDA_CHECK(cudaMemcpy(destino, ponteiro, bytes, cudaMemcpyDeviceToHost));
    }
};

constexpr int threads_bloco = 128;
int blocos(int elementos) {
    return (elementos + threads_bloco - 1) / threads_bloco;
}

__global__ void forward_gpu(const float* entrada, const float* pesos,
                            const float* vies, float* z, float* saida,
                            int n_entradas, int n_saidas, bool final) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_saidas) return;
    float soma = vies[i];
    for (int j = 0; j < n_entradas; ++j) {
        soma += entrada[j] * pesos[i * n_entradas + j];
    }
    z[i] = soma;
    saida[i] = final ? soma : (soma > 0.0f ? soma : 0.0f);
}

__global__ void delta_gpu(const float* grad_saida, const float* z,
                          float* delta, int n_saidas, bool final) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_saidas) return;
    delta[i] = final ? grad_saida[i]
                    : grad_saida[i] * (z[i] > 0.0f ? 1.0f : 0.0f);
}

__global__ void gradiente_gpu(const float* pesos, const float* delta,
                              float* grad_entrada, int n_entradas,
                              int n_saidas) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_entradas) return;
    float soma = 0.0f;
    for (int j = 0; j < n_saidas; ++j) {
        soma += delta[j] * pesos[j * n_entradas + i];
    }
    grad_entrada[i] = soma;
}

__global__ void atualizar_gpu(float* pesos, float* vies, const float* entrada,
                              const float* delta, int n_entradas,
                              int n_pesos, float taxa) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_pesos) return;
    int i = k / n_entradas;
    int j = k % n_entradas;
    pesos[k] -= taxa * (delta[i] * entrada[j]);
    if (j == 0) vies[i] -= taxa * delta[i];
}

class DenseLayerGPU {
    int in_features;
    int out_features;
    bool final;
    BufferGPU<float> pesos, vies, entrada, z, saida, grad_saida, delta, grad_entrada;
public:
    DenseLayerGPU(int n_entradas, int n_saidas, bool ultima = false)
        : in_features(n_entradas), out_features(n_saidas), final(ultima),
          pesos(n_entradas * n_saidas), vies(n_saidas), entrada(n_entradas),
          z(n_saidas), saida(n_saidas), grad_saida(n_saidas), delta(n_saidas),
          grad_entrada(n_entradas) {
        mt19937 gen(1337);
        uniform_real_distribution<float> distribution(-0.05f, 0.05f);
        vector<float> pesos_host(n_entradas * n_saidas);
        for (float& peso : pesos_host) peso = distribution(gen);
        vector<float> vies_host(n_saidas, 0.0f);
        pesos.enviar(pesos_host.data());
        vies.enviar(vies_host.data());
    }

    vector<float> Forward(const vector<float>& input) {
        entrada.enviar(input.data());
        forward_gpu<<<blocos(out_features), threads_bloco>>>(
            entrada.data(), pesos.data(), vies.data(), z.data(), saida.data(),
            in_features, out_features, final);
        CUDA_CHECK(cudaGetLastError());
        vector<float> out(out_features);
        // A cópia síncrona espera o kernel antes de devolver o vetor à CPU.
        saida.receber(out.data());
        return out;
    }

    vector<float> Predict(const vector<float>& input) {
        // Chamado sequencialmente no host; os buffers pertencem à camada.
        return Forward(input);
    }

    vector<float> Backward(const vector<float>& grad_out, float taxa) {
        grad_saida.enviar(grad_out.data());
        delta_gpu<<<blocos(out_features), threads_bloco>>>(
            grad_saida.data(), z.data(), delta.data(), out_features, final);
        CUDA_CHECK(cudaGetLastError());
        gradiente_gpu<<<blocos(in_features), threads_bloco>>>(
            pesos.data(), delta.data(), grad_entrada.data(),
            in_features, out_features);
        CUDA_CHECK(cudaGetLastError());
        // Na mesma stream, o gradiente usa os pesos antigos antes da atualização.
        atualizar_gpu<<<blocos(in_features * out_features), threads_bloco>>>(
            pesos.data(), vies.data(), entrada.data(), delta.data(),
            in_features, in_features * out_features, taxa);
        CUDA_CHECK(cudaGetLastError());
        vector<float> grad_in(in_features);
        grad_entrada.receber(grad_in.data());
        return grad_in;
    }

    float* PesosGPU() const { return pesos.data(); }
    float* ViesGPU() const { return vies.data(); }

    vector<float> Parametros() const {
        vector<float> valores(in_features * out_features + out_features);
        pesos.receber(valores.data());
        vies.receber(valores.data() + in_features * out_features);
        return valores;
    }
};
bool cuda_otimizado = false;

// Especializado para a arquitetura 64 -> 32 -> 16 -> 10 da APS.
// Um bloco coopera dentro da amostra. As amostras continuam sequenciais.
__global__ void treinar_epoca_gpu(
    const float* imagens, const int* rotulos, int n_amostras,
    float* pesos1, float* vies1, float* pesos2, float* vies2,
    float* pesos3, float* vies3, double* perdas, int epoca, float taxa) {
    // Uma posição extra por linha distribui as leituras entre os bancos.
    __shared__ float w1[32][65], w2[16][33], w3[10][17];
    __shared__ float b1[32], b2[16], b3[10];
    __shared__ float x[64], a1[32], z1[32], a2[16], z2[16], out[10];
    __shared__ float d3[10], d2[16], d1[32];
    __shared__ double perda_total;
    const int t = threadIdx.x;

    for (int k = t; k < 32 * 64; k += blockDim.x) w1[k / 64][k % 64] = pesos1[k];
    for (int k = t; k < 16 * 32; k += blockDim.x) w2[k / 32][k % 32] = pesos2[k];
    for (int k = t; k < 10 * 16; k += blockDim.x) w3[k / 16][k % 16] = pesos3[k];
    if (t < 32) b1[t] = vies1[t];
    if (t < 16) b2[t] = vies2[t];
    if (t < 10) b3[t] = vies3[t];
    if (t == 0) perda_total = 0.0;
    __syncthreads();

    for (int amostra = 0; amostra < n_amostras; ++amostra) {
        if (t < 64) x[t] = imagens[amostra * 64 + t];
        __syncthreads();
        if (t < 32) {
            float soma = b1[t];
            for (int j = 0; j < 64; ++j) soma += x[j] * w1[t][j];
            z1[t] = soma;
            a1[t] = soma > 0.0f ? soma : 0.0f;
        }
        __syncthreads();
        if (t < 16) {
            float soma = b2[t];
            for (int j = 0; j < 32; ++j) soma += a1[j] * w2[t][j];
            z2[t] = soma;
            a2[t] = soma > 0.0f ? soma : 0.0f;
        }
        __syncthreads();
        if (t < 10) {
            float soma = b3[t];
            for (int j = 0; j < 16; ++j) soma += a2[j] * w3[t][j];
            out[t] = soma;
        }
        __syncthreads();
        // Conserva a ordem e os tipos da soma de perdas da referência CPU.
        if (t == 0) {
            float perda_amostra = 0.0f;
            for (int j = 0; j < 10; ++j) {
                float erro = out[j] - (j == rotulos[amostra] ? 1.0f : 0.0f);
                perda_amostra += 0.5f * erro * erro;
                d3[j] = erro;
            }
            perda_total += perda_amostra;
        }
        __syncthreads();
        // Os gradientes usam os pesos anteriores às atualizações.
        if (t < 16) {
            float soma = 0.0f;
            for (int j = 0; j < 10; ++j) soma += d3[j] * w3[j][t];
            d2[t] = soma * (z2[t] > 0.0f ? 1.0f : 0.0f);
        }
        __syncthreads();
        if (t < 32) {
            float soma = 0.0f;
            for (int j = 0; j < 16; ++j) soma += d2[j] * w2[j][t];
            d1[t] = soma * (z1[t] > 0.0f ? 1.0f : 0.0f);
        }
        __syncthreads();
        for (int k = t; k < 10 * 16; k += blockDim.x) {
            int i = k / 16, j = k % 16;
            w3[i][j] -= taxa * (d3[i] * a2[j]);
        }
        for (int k = t; k < 16 * 32; k += blockDim.x) {
            int i = k / 32, j = k % 32;
            w2[i][j] -= taxa * (d2[i] * a1[j]);
        }
        for (int k = t; k < 32 * 64; k += blockDim.x) {
            int i = k / 64, j = k % 64;
            w1[i][j] -= taxa * (d1[i] * x[j]);
        }
        if (t < 10) b3[t] -= taxa * d3[t];
        if (t < 16) b2[t] -= taxa * d2[t];
        if (t < 32) b1[t] -= taxa * d1[t];
        // A próxima amostra só começa com todos os pesos atualizados.
        __syncthreads();
    }
    for (int k = t; k < 32 * 64; k += blockDim.x) pesos1[k] = w1[k / 64][k % 64];
    for (int k = t; k < 16 * 32; k += blockDim.x) pesos2[k] = w2[k / 32][k % 32];
    for (int k = t; k < 10 * 16; k += blockDim.x) pesos3[k] = w3[k / 16][k % 16];
    if (t < 32) vies1[t] = b1[t];
    if (t < 16) vies2[t] = b2[t];
    if (t < 10) vies3[t] = b3[t];
    if (t == 0) perdas[epoca] = perda_total;
}

// Um bloco de um warp por imagem, todos no mesmo lançamento.
__global__ void inferir_lote_gpu(
    const float* imagens, float* escores, int n_amostras,
    const float* pesos1, const float* vies1,
    const float* pesos2, const float* vies2,
    const float* pesos3, const float* vies3) {
    const int amostra = blockIdx.x;
    if (amostra >= n_amostras) return;
    const int t = threadIdx.x;
    __shared__ float a1[32], a2[16];
    const float* x = imagens + amostra * 64;
    if (t < 32) {
        float soma = vies1[t];
        for (int j = 0; j < 64; ++j) soma += x[j] * pesos1[t * 64 + j];
        a1[t] = soma > 0.0f ? soma : 0.0f;
    }
    __syncthreads();
    if (t < 16) {
        float soma = vies2[t];
        for (int j = 0; j < 32; ++j) soma += a1[j] * pesos2[t * 32 + j];
        a2[t] = soma > 0.0f ? soma : 0.0f;
    }
    __syncthreads();
    if (t < 10) {
        float soma = vies3[t];
        for (int j = 0; j < 16; ++j) soma += a2[j] * pesos3[t * 16 + j];
        escores[amostra * 10 + t] = soma;
    }
}

struct Amostra {
    vector<float> imagem;
    int rotulo;
};

vector<Amostra> ler_dados(const string& caminho) {
    ifstream arquivo(caminho);
    if (!arquivo) {
        throw runtime_error("Não foi possível abrir " + caminho);
    }

    vector<Amostra> dados;
    string linha;
    size_t numero_linha = 0;
    while (getline(arquivo, linha)) {
        ++numero_linha;
        istringstream campos(linha);
        Amostra amostra{};
        // Cada atributo deve ser seguido por vírgula, inclusive o último.
        for (int i = 0; i < entradas; ++i) {
            float valor;
            char separador;
            if (!(campos >> valor >> separador) || separador != ',' ||
                !isfinite(valor) || valor < 0 || valor > 16) {
                break;
            }
            // Converte os valores de [0, 16] para [0, 1].
            amostra.imagem.push_back(valor / 16.0f);
        }
        // Depois dos atributos, deve restar apenas o rótulo de 0 a 9.
        if (amostra.imagem.size() != entradas ||
            !(campos >> amostra.rotulo) ||
            amostra.rotulo < 0 || amostra.rotulo >= classes ||
            !(campos >> ws).eof()) {
            throw runtime_error(caminho + ":" + to_string(numero_linha) +
                " — esperados 64 atributos de 0 a 16 e um rótulo inteiro de 0 a 9");
        }
        dados.push_back(move(amostra));
    }
    if (arquivo.bad() || dados.empty()) {
        throw runtime_error("Falha de leitura ou arquivo vazio: " + caminho);
    }
    return dados;
}

// Calcula metade da soma dos erros quadráticos. O alvo vale 1 para a
// classe correta e 0 para as demais. O erro também é a derivada por escore.
float perda(const vector<float>& escores, int rotulo,
            vector<float>* gradiente = nullptr) {
    float total = 0;
    for (int i = 0; i < classes; ++i) {
        float erro = escores[i] - (i == rotulo ? 1.0f : 0.0f);
        total += 0.5f * erro * erro;
        if (gradiente) {
            (*gradiente)[i] = erro;
        }
    }
    return total;
}

void mostrar_avaliacao(const vector<vector<float>>& escores_lote,
                       const vector<Amostra>& teste) {
    struct Exemplo {
        size_t linha;
        int real;
        int previsto;
        int segundo;
        float maior_escore;
        float segundo_escore;
    };

    // Linhas representam a classe esperada e colunas, a classe prevista.
    array<array<size_t, classes>, classes> confusoes{};
    vector<Exemplo> exemplos;
    size_t acertos = 0;
    size_t exemplos_acerto = 0;
    size_t exemplos_erro = 0;
    double perda_total = 0;

    for (size_t i = 0; i < teste.size(); ++i) {
        const auto& amostra = teste[i];
        const auto& escores = escores_lote[i];
        // O índice do maior escore indica a classe prevista.
        int previsto = static_cast<int>(
            max_element(escores.begin(), escores.end()) - escores.begin());
        bool acertou = previsto == amostra.rotulo;
        acertos += acertou;
        ++confusoes[amostra.rotulo][previsto];
        perda_total += perda(escores, amostra.rotulo);

        // Guarda até três acertos e três erros para ilustrar a saída.
        // A acurácia usa todas as imagens, inclusive as não exibidas.
        size_t& exemplos_tipo = acertou ? exemplos_acerto : exemplos_erro;
        if (exemplos_tipo < 3) {
            int segundo = previsto == 0 ? 1 : 0;
            for (int classe = 0; classe < classes; ++classe) {
                if (classe != previsto && escores[classe] > escores[segundo]) {
                    segundo = classe;
                }
            }
            exemplos.push_back({i + 1, amostra.rotulo, previsto, segundo,
                                escores[previsto], escores[segundo]});
            ++exemplos_tipo;
        }
    }

    cout << "\nAVALIACAO FINAL\n"
         << "Acuracia " << setprecision(1) << 100.0 * acertos / teste.size()
         << "% | " << acertos << " acertos em " << teste.size() << " imagens\n"
         << "Perda media " << setprecision(4) << perda_total / teste.size() << "\n";

    cout << "\nALGUMAS INFERENCIAS\n";
    cout << "Esperado  Previsto\n";

    for (const auto& exemplo : exemplos) {
        cout << right << setw(8) << exemplo.real
             << "  " << setw(8) << exemplo.previsto << '\n';
    }

    struct Confusao {
        int real;
        int previsto;
        size_t ocorrencias;
    };
    vector<Confusao> erros;
    for (int real = 0; real < classes; ++real) {
        for (int previsto = 0; previsto < classes; ++previsto) {
            if (real != previsto && confusoes[real][previsto] > 0) {
                erros.push_back({real, previsto, confusoes[real][previsto]});
            }
        }
    }
    // Ordena os pares incorretos do mais frequente para o menos frequente.
    stable_sort(erros.begin(), erros.end(), [](const auto& a, const auto& b) {
        return a.ocorrencias > b.ocorrencias;
    });
    cout << "\nCONFUSOES MAIS FREQUENTES\n";

    if (erros.empty()) {
        cout << "Nenhuma neste conjunto.\n";
    } else {
        cout << "Esperado  Previsto  Ocorrencias\n";

        for (size_t i = 0; i < min<size_t>(3, erros.size()); ++i) {
            const auto& erro = erros[i];
            cout << right << setw(8) << erro.real
                << "  " << setw(8) << erro.previsto
                << "  " << setw(11) << erro.ocorrencias << '\n';
        }
    }
}


template<class Camada>
vector<double> treinar(Camada& l1, Camada& l2, Camada& l3,
                       const vector<Amostra>& treino, int epocas) {
    constexpr float taxa = 0.02f;
    vector<float> gradiente(classes);
    vector<double> perdas(epocas);
    for (int epoca = 0; epoca < epocas; ++epoca) {
        double perda_total = 0;
        for (const auto& amostra : treino) {
            auto escores = l3.Forward(l2.Forward(l1.Forward(amostra.imagem)));
            perda_total += perda(escores, amostra.rotulo, &gradiente);
            auto grad3 = l3.Backward(gradiente, taxa);
            auto grad2 = l2.Backward(grad3, taxa);
            l1.Backward(grad2, taxa);
        }
        perdas[epoca] = perda_total / treino.size();
        if (!isfinite(perdas[epoca])) {
            throw runtime_error("Perda não finita no treinamento");
        }
    }
    return perdas;
}

template<class Camada>
vector<vector<float>> inferir(Camada& l1, Camada& l2, Camada& l3,
                              const vector<Amostra>& teste) {
    vector<vector<float>> escores(teste.size());
    for (size_t i = 0; i < teste.size(); ++i) {
        escores[i] = l3.Predict(l2.Predict(l1.Predict(teste[i].imagem)));
    }
    return escores;
}

vector<float> empacotar(const vector<Amostra>& dados) {
    vector<float> imagens(dados.size() * entradas);
    for (size_t i = 0; i < dados.size(); ++i) {
        copy(dados[i].imagem.begin(), dados[i].imagem.end(),
             imagens.begin() + i * entradas);
    }
    return imagens;
}

vector<double> treinar(DenseLayerGPU& l1, DenseLayerGPU& l2, DenseLayerGPU& l3,
                       const vector<Amostra>& treino, int epocas) {
    if (!cuda_otimizado) return treinar<DenseLayerGPU>(l1, l2, l3, treino, epocas);
    // Empacotamento, alocações do lote e transferências estão no tempo de treino.
    auto imagens = empacotar(treino);
    vector<int> rotulos(treino.size());
    for (size_t i = 0; i < treino.size(); ++i) rotulos[i] = treino[i].rotulo;
    BufferGPU<float> imagens_gpu(imagens.size());
    BufferGPU<int> rotulos_gpu(rotulos.size());
    BufferGPU<double> perdas_gpu(epocas);
    imagens_gpu.enviar(imagens.data());
    rotulos_gpu.enviar(rotulos.data());
    for (int epoca = 0; epoca < epocas; ++epoca) {
        // A mesma stream termina uma época antes de iniciar a seguinte.
        treinar_epoca_gpu<<<1, threads_bloco>>>(
            imagens_gpu.data(), rotulos_gpu.data(), static_cast<int>(treino.size()),
            l1.PesosGPU(), l1.ViesGPU(), l2.PesosGPU(), l2.ViesGPU(),
            l3.PesosGPU(), l3.ViesGPU(), perdas_gpu.data(), epoca, 0.02f);
        CUDA_CHECK(cudaGetLastError());
    }
    vector<double> perdas(epocas);
    // Uma única cópia devolve as perdas de todas as épocas e espera os kernels.
    perdas_gpu.receber(perdas.data());
    for (double& perda_epoca : perdas) {
        perda_epoca /= treino.size();
        if (!isfinite(perda_epoca)) throw runtime_error("Perda não finita no treinamento");
    }
    return perdas;
}

vector<vector<float>> inferir(DenseLayerGPU& l1, DenseLayerGPU& l2,
                              DenseLayerGPU& l3, const vector<Amostra>& teste) {
    if (!cuda_otimizado) return inferir<DenseLayerGPU>(l1, l2, l3, teste);
    auto imagens = empacotar(teste);
    BufferGPU<float> imagens_gpu(imagens.size());
    BufferGPU<float> escores_gpu(teste.size() * classes);
    imagens_gpu.enviar(imagens.data());
    inferir_lote_gpu<<<static_cast<int>(teste.size()), 32>>>(
        imagens_gpu.data(), escores_gpu.data(), static_cast<int>(teste.size()),
        l1.PesosGPU(), l1.ViesGPU(), l2.PesosGPU(), l2.ViesGPU(),
        l3.PesosGPU(), l3.ViesGPU());
    CUDA_CHECK(cudaGetLastError());
    vector<float> lote(teste.size() * classes);
    escores_gpu.receber(lote.data());
    vector<vector<float>> escores(teste.size(), vector<float>(classes));
    for (size_t i = 0; i < teste.size(); ++i) {
        copy(lote.begin() + i * classes, lote.begin() + (i + 1) * classes,
             escores[i].begin());
    }
    return escores;
}

template<class Camada>
vector<float> parametros_rede(const Camada& l1, const Camada& l2,
                              const Camada& l3) {
    auto parametros = l1.Parametros();
    auto p2 = l2.Parametros();
    auto p3 = l3.Parametros();
    parametros.insert(parametros.end(), p2.begin(), p2.end());
    parametros.insert(parametros.end(), p3.begin(), p3.end());
    return parametros;
}

struct Comparacao {
    size_t total = 0;
    size_t divergentes = 0;
    double maior_erro = 0;
    void adicionar(double obtido, double referencia) {
        ++total;
        const double erro = abs(obtido - referencia);
        if (!isfinite(obtido) || !isfinite(referencia) ||
            erro > 1e-5 + 1e-5 * abs(referencia)) {
            ++divergentes;
        }
        if (isfinite(erro)) maior_erro = max(maior_erro, erro);
    }
};

void validar(const vector<double>& perdas_gpu,
              const vector<vector<float>>& escores_gpu,
              const vector<float>& parametros_gpu,
              const vector<Amostra>& treino, const vector<Amostra>& teste,
              int epocas) {
    DenseLayerCPU c1(entradas, 32), c2(32, 16), c3(16, classes, true);
    auto perdas_cpu = treinar(c1, c2, c3, treino, epocas);
    auto escores_cpu = inferir(c1, c2, c3, teste);
    auto parametros_cpu = parametros_rede(c1, c2, c3);
    if (perdas_gpu.size() != perdas_cpu.size() ||
        escores_gpu.size() != escores_cpu.size() ||
        parametros_gpu.size() != parametros_cpu.size()) {
        throw runtime_error("Dimensões divergentes na validação");
    }
    Comparacao perdas, escores, parametros;
    for (size_t i = 0; i < perdas_cpu.size(); ++i) {
        perdas.adicionar(perdas_gpu[i], perdas_cpu[i]);
    }
    size_t predicoes_diferentes = 0;
    for (size_t i = 0; i < escores_cpu.size(); ++i) {
        if (escores_gpu[i].size() != escores_cpu[i].size()) {
            throw runtime_error("Número de classes divergente na validação");
        }
        for (int j = 0; j < classes; ++j) {
            escores.adicionar(escores_gpu[i][j], escores_cpu[i][j]);
        }
        auto gpu = max_element(escores_gpu[i].begin(), escores_gpu[i].end())
                   - escores_gpu[i].begin();
        auto cpu = max_element(escores_cpu[i].begin(), escores_cpu[i].end())
                   - escores_cpu[i].begin();
        predicoes_diferentes += gpu != cpu;
    }
    for (size_t i = 0; i < parametros_cpu.size(); ++i) {
        parametros.adicionar(parametros_gpu[i], parametros_cpu[i]);
    }
    cout << "\nVALIDACAO CONTRA CPU | " << epocas << " epoca(s)\n"
         << "Tolerancia: absoluta 1e-5 + relativa 1e-5\n" << scientific
         << "Perdas: " << perdas.total << "; maior erro absoluto: "
         << perdas.maior_erro << "; fora da tolerancia: " << perdas.divergentes
         << "\nEscores: " << escores.total << "; maior erro absoluto: "
         << escores.maior_erro << "; fora da tolerancia: " << escores.divergentes
         << "\nParametros: " << parametros.total << "; maior erro absoluto: "
         << parametros.maior_erro << "; fora da tolerancia: " << parametros.divergentes
         << "\nPredicoes diferentes: " << predicoes_diferentes << '\n';
    if (perdas.divergentes || escores.divergentes || parametros.divergentes ||
        predicoes_diferentes) {
        throw runtime_error("Validacao numerica FALHOU");
    }
    cout << "Validacao numerica: OK\n";
}

template<class Camada>
void executar(const vector<Amostra>& treino, const vector<Amostra>& teste,
               int epocas, bool validar_gpu) {
    Camada l1(entradas, 32), l2(32, 16), l3(16, classes, true);
    const auto t0 = chrono::steady_clock::now();
    auto perdas = treinar(l1, l2, l3, treino, epocas);
    const auto t1 = chrono::steady_clock::now();
    // O retorno do treino ocorre depois da cópia síncrona dos resultados.
    const double tempo = chrono::duration<double>(t1 - t0).count();
    cout << fixed << setprecision(4) << "TREINO | " << treino.size() << " imagens\n";
    for (int epoca = 0; epoca < epocas; ++epoca) {
        if (epoca % 5 != 0 && epoca != epocas - 1) continue;
        cout << "Epoca " << setw(2) << epoca + 1 << '/' << epocas
             << " | perda media " << perdas[epoca] << '\n';
    }
    cout << "Tempo de treino " << tempo << " s\n";
    const auto a0 = chrono::steady_clock::now();
    auto escores = inferir(l1, l2, l3, teste);
    mostrar_avaliacao(escores, teste);
    const auto a1 = chrono::steady_clock::now();
    cout << "Tempo de avaliacao " << setprecision(6)
         << chrono::duration<double, milli>(a1 - a0).count() << " ms\n";
    if (validar_gpu) {
        // Validação fora dos cronômetros de treinamento e avaliação.
        validar(perdas, escores, parametros_rede(l1, l2, l3), treino, teste, epocas);
    }
}

} // namespace Barenn

int main(int argc, char** argv) {
    try {
        int epocas = 51;
        bool cpu = false, validar = false;
        vector<string> arquivos;
        for (int i = 1; i < argc; ++i) {
            string argumento = argv[i];
            if (argumento == "--cpu") cpu = true;
            else if (argumento == "--validar") validar = true;
            else if (argumento == "--otimizado") Barenn::cuda_otimizado = true;
            else if (argumento == "--epocas" && i + 1 < argc) {
                string valor = argv[++i];
                size_t fim = 0;
                epocas = stoi(valor, &fim);
                if (fim != valor.size() || epocas <= 0) {
                    throw runtime_error("Numero de epocas deve ser inteiro positivo");
                }
            } else if (argumento.rfind("--", 0) == 0) {
                throw runtime_error("Opcao desconhecida ou incompleta: " + argumento);
            } else arquivos.push_back(argumento);
        }
        if (!arquivos.empty() && arquivos.size() != 2) {
            throw runtime_error("Informe ambos os arquivos de treino e teste");
        }
        const string treino_path = arquivos.empty() ? "optdigits.tra" : arquivos[0];
        const string teste_path = arquivos.empty() ? "optdigits.tes" : arquivos[1];
        if (filesystem::equivalent(treino_path, teste_path)) {
            throw runtime_error("Use arquivos distintos para treino e teste");
        }
        const auto treino = Barenn::ler_dados(treino_path);
        const auto teste = Barenn::ler_dados(teste_path);
        if (cpu) {
            if (validar || Barenn::cuda_otimizado) {
                throw runtime_error("Use --validar e --otimizado no modo GPU");
            }
            Barenn::executar<Barenn::DenseLayerCPU>(treino, teste, epocas, false);
        } else {
            cudaDeviceProp propriedades{};
            Barenn::verificar_cuda(cudaGetDeviceProperties(&propriedades, 0),
                                  "cudaGetDeviceProperties", __LINE__);
            cout << "GPU: " << propriedades.name << '\n';
            cout << "Modo: " << (Barenn::cuda_otimizado ? "CUDA com fusao" : "CUDA inicial")
                 << '\n';
            Barenn::executar<Barenn::DenseLayerGPU>(treino, teste, epocas, validar);
        }
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << '\n';
        return 1;
    }
    return 0;
}
