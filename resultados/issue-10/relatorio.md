## Resultado

O treinamento passou a usar um kernel por época, com pesos e intermediários em memória compartilhada. A avaliação passou a usar um único lançamento para o lote inteiro.

As amostras de treinamento continuam sequenciais, com as mesmas atualizações SGD. As threads cooperam dentro de cada amostra. As barreiras garantem a conclusão das etapas e o cálculo dos gradientes com os pesos anteriores à atualização.

## Medições

Cinco rodadas alternadas no compute22, executando CPU de referência, CUDA inicial e CUDA otimizada na mesma alocação. Configuração: merry_gpu, 1 processo, 1 CPU, 1 GPU e 1 GB. GPU NVIDIA RTX A1000, 8.188 MiB reportados, driver 570.124.06.

Compilação CUDA: nvcc -std=c++17 -O2 -lineinfo --fmad=false, toolkit 12.8.1. São 51 épocas sobre optdigits.tra e avaliação com optdigits.tes. A CPU de referência é o modo sequencial --cpu do mesmo programa CUDA.

| Modo | Treinamento médio | Avaliação média |
|:---|---:|---:|
| CPU de referência | 0,36172 s | 1,247235 ms |
| CUDA inicial | 17,80420 s | 77,209470 ms |
| CUDA otimizada | 0,49926 s | 0,329031 ms |

O treinamento CUDA ficou 35,66x mais rápido que a versão CUDA inicial, com redução de 97,20%. Na avaliação, o speedup foi de 234,66x sobre a CUDA inicial e 3,79x sobre a CPU de referência.

A CPU ainda foi mais rápida no treinamento: a CUDA otimizada levou 38,02% mais tempo. Portanto, o ganho comprovado no treinamento é sobre a CUDA inicial. Não houve superioridade da GPU sobre a CPU em todas as etapas.

## Profiling e explicação

O Nsight Systems foi executado separadamente, com uma época e a avaliação completa, em ambas as versões:

| Métrica | CUDA inicial | CUDA otimizada |
|:---|---:|---:|
| Lançamentos de kernels | 51.267 | 2 |
| Chamadas cudaMemcpy | 56.664 | 11 |
| Tempo acumulado dos kernels na GPU | 90,763354 ms | 12,390191 ms |
| Tempo acumulado das cópias na GPU | 45,050417 ms | 0,129405 ms |

A versão inicial fazia cálculos curtos e devolvia intermediários à CPU entre as camadas. Na versão otimizada, os dados de treino são enviados uma vez e as perdas de todas as épocas voltam em uma cópia. Os pesos são reutilizados em memória compartilhada durante cada época. Uma posição extra entre as linhas distribui as leituras entre os bancos.

A redução de chamadas diminui a repetição de custos de lançamento, comunicação e espera. O ganho observado não deve ser atribuído apenas à velocidade das operações aritméticas nem interpretado como prova de saturação da largura de banda PCIe na versão inicial. O tempo da API cudaMemcpy pode incluir a espera pelo trabalho na GPU.

O bloco único de treinamento preserva as dependências entre amostras, mas limita o uso simultâneo dos SMs. Esse é um compromisso da implementação: reduz bastante os custos da migração inicial, sem explorar paralelismo entre imagens durante o treino. A rede pequena e os custos de runtime também limitam a vantagem da GPU. Na avaliação, os pesos ficam fixos e as imagens podem ser distribuídas entre blocos, oferecendo mais paralelismo.

Os tempos com profiler são instrumentados e usam uma época; não substituem as médias sem profiler de 51 épocas. Os percentuais de cada relatório pertencem àquele relatório, não ao tempo total do programa. O percentual alto de cudaMalloc na coleta otimizada inclui um custo isolado de inicialização/alocação; as alocações das camadas acontecem antes do cronômetro de treino.

O Nsight Compute foi tentado com LaunchStats, Occupancy e SpeedOfLight, mas retornou ERR_NVGPUCTRPERM. O usuário do cluster não tem acesso aos contadores de desempenho da GPU. A tentativa está registrada em resumo_ncu.txt. Não foram obtidas métricas de ocupação real ou throughput de hardware; nenhuma conclusão depende de valores não coletados.

## Corretude e escopo dos tempos

As cinco execuções otimizadas passaram na validação contra a CPU, comparando 51 perdas, 17.970 escores, 2.778 parâmetros e classes previstas. A tolerância é erro absoluto de 1e-5 mais erro relativo de 1e-5, e todas as classes previstas devem coincidir. A referência final manteve 95,5% de acurácia, com 1.717 acertos em 1.797 imagens.

Empacotamento, alocações dos buffers do lote e transferências da versão otimizada estão incluídos nos tempos medidos. A criação e a inicialização das camadas ficam fora, como na versão inicial. A avaliação inclui inferência, métricas e impressão. A validação numérica contra a CPU ocorre depois dos cronômetros. Cada medição iniciou um processo novo, sem fase separada de aquecimento dos kernels.

As saídas das cinco rodadas, o resumo do Nsight Systems e a tentativa do Nsight Compute estão em resultados/issue-10. O README reúne os comandos de reprodução e os resultados das etapas anteriores.

Marco: [v0.6-cuda-fusao](https://github.com/alis-on-araujo/aps2-supercomp/tree/v0.6-cuda-fusao).
