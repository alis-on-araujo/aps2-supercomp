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
class BufferGPU {
    float* ponteiro = nullptr;
    size_t bytes;
public:
    explicit BufferGPU(size_t elementos) : bytes(elementos * sizeof(float)) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ponteiro), bytes));
    }
    ~BufferGPU() { if (ponteiro) cudaFree(ponteiro); }
    BufferGPU(const BufferGPU&) = delete;
    BufferGPU& operator=(const BufferGPU&) = delete;
    float* data() const { return ponteiro; }
    void enviar(const float* origem) {
        CUDA_CHECK(cudaMemcpy(ponteiro, origem, bytes, cudaMemcpyHostToDevice));
    }
    void receber(float* destino) const {
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
    BufferGPU pesos, vies, entrada, z, saida, grad_saida, delta, grad_entrada;
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

    vector<float> Parametros() const {
        vector<float> valores(in_features * out_features + out_features);
        pesos.receber(valores.data());
        vies.receber(valores.data() + in_features * out_features);
        return valores;
    }
};
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
    // As cópias D2H síncronas do Backward completam cada atualização.
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
            if (validar) throw runtime_error("Use --validar no modo GPU");
            Barenn::executar<Barenn::DenseLayerCPU>(treino, teste, epocas, false);
        } else {
            cudaDeviceProp propriedades{};
            Barenn::verificar_cuda(cudaGetDeviceProperties(&propriedades, 0),
                                  "cudaGetDeviceProperties", __LINE__);
            cout << "GPU: " << propriedades.name << '\n';
            Barenn::executar<Barenn::DenseLayerGPU>(treino, teste, epocas, validar);
        }
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << '\n';
        return 1;
    }
    return 0;
}
