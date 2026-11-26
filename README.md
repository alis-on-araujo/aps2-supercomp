# Resultados da APS2

Rede densa 64 → 32 → 16 → 10, treinada com optdigits.tra (3.823 imagens) e avaliada com optdigits.tes (1.797 imagens). Mantivemos 51 épocas, taxa de aprendizado 0,02, semente 1337 em cada camada e a ordem das amostras do arquivo.

## Comparação consolidada

Médias de cinco execuções. Os resultados históricos de CPU e os resultados de GPU foram obtidos em nós diferentes; a comparação controlada no mesmo nó está na tabela seguinte.

| Versão | Ambiente | Treino (s) | Avaliação (ms) | Marco |
|:---|:---|---:|---:|:---|
| CPU original | compute10, merry_cpu, 1 CPU | 0,47122 | Não medida neste teste | [v0.1-baseline](https://github.com/alis-on-araujo/aps2-supercomp/tree/v0.1-baseline) |
| CPU com blocos de oito | compute10, merry_cpu, 1 CPU | 0,36396 | Não medida neste teste | [v0.2-cpu-blocos](https://github.com/alis-on-araujo/aps2-supercomp/tree/v0.2-cpu-blocos) |
| CUDA inicial | compute22, merry_gpu, 1 CPU e 1 GPU | 17,80420 | 77,209470 | [v0.5-cuda-inicial](https://github.com/alis-on-araujo/aps2-supercomp/tree/v0.5-cuda-inicial) |
| CUDA com fusão e memória compartilhada | compute22, merry_gpu, 1 CPU e 1 GPU | 0,49926 | 0,329031 | [v0.6-cuda-fusao](https://github.com/alis-on-araujo/aps2-supercomp/tree/v0.6-cuda-fusao) |

A otimização em blocos reduziu o tempo de treinamento CPU em 22,76%. A fusão CUDA reduziu o treino em 97,20% em relação à CUDA inicial, com speedup de 35,66x.

### Comparação controlada no compute22

Cinco rodadas alternadas, cada uma executando CPU de referência, CUDA inicial e CUDA otimizada na mesma alocação: 1 processo, 1 CPU, 1 GPU e 1 GB de memória. GPU NVIDIA RTX A1000, 8.188 MiB reportados, driver 570.124.06. A referência CPU desta tabela é o modo sequencial --cpu do programa CUDA.

| Modo | Treino (s) | Avaliação (ms) |
|:---|---:|---:|
| CPU de referência | 0,36172 | 1,247235 |
| CUDA inicial | 17,80420 | 77,209470 |
| CUDA otimizada | 0,49926 | 0,329031 |

A CPU permaneceu mais rápida no treino: a CUDA otimizada levou aproximadamente 38,02% mais tempo. Na avaliação, a CUDA otimizada foi 3,79x mais rápida que a CPU de referência e 234,66x mais rápida que a CUDA inicial. O ganho em relação à CUDA inicial não significa ganho sobre a CPU em todas as etapas.

### Paralelismo multicore na avaliação

No compute10, cinco rodadas alternadas do mesmo executável OpenMP, com alocação de 2 CPUs e 1 GB:

| Configuração | Tempo médio da avaliação (ms) |
|:---|---:|
| 1 thread | 1,321218 |
| 2 threads | 0,865017 |

Redução de 34,53%, com speedup de 1,53x. As inferências usam pesos fixos e dados temporários privados; as métricas são reunidas na ordem original. Esse ganho pertence à avaliação, não ao treinamento.

## Issues, tags e evidências

| Bloco de trabalho | Discussão | Tag anotada | Evidências |
|:---|:---|:---|:---|
| Referência original | [Issue #1](https://github.com/alis-on-araujo/aps2-supercomp/issues/1) | [v0.1-baseline](https://github.com/alis-on-araujo/aps2-supercomp/tree/v0.1-baseline) | [Medições e gprof](resultados/issue-1/) |
| Otimização CPU em blocos | [Issue #1](https://github.com/alis-on-araujo/aps2-supercomp/issues/1) | [v0.2-cpu-blocos](https://github.com/alis-on-araujo/aps2-supercomp/tree/v0.2-cpu-blocos) | [Comparação](resultados/issue-1/comparacao_blocos.txt) |
| Experimento OpenMP no Backward e restauração | [Issue #4](https://github.com/alis-on-araujo/aps2-supercomp/issues/4) | [v0.3-cpu-openmp-avaliado](https://github.com/alis-on-araujo/aps2-supercomp/tree/v0.3-cpu-openmp-avaliado) | [Pré-teste](resultados/issue-4/) |
| Avaliação multicore | [Issue #6](https://github.com/alis-on-araujo/aps2-supercomp/issues/6) | [v0.4-cpu-avaliacao-openmp](https://github.com/alis-on-araujo/aps2-supercomp/tree/v0.4-cpu-avaliacao-openmp) | [Resumo e validação](resultados/issue-6/resumo.txt) |
| Migração funcional CUDA | [Issue #8](https://github.com/alis-on-araujo/aps2-supercomp/issues/8) | [v0.5-cuda-inicial](https://github.com/alis-on-araujo/aps2-supercomp/tree/v0.5-cuda-inicial) | [Medições](resultados/issue-8/medicoes.txt) e [Nsight](resultados/issue-8/resumo_nsys.txt) |
| Fusão CUDA e memória compartilhada | [Issue #10](https://github.com/alis-on-araujo/aps2-supercomp/issues/10) | [v0.6-cuda-fusao](https://github.com/alis-on-araujo/aps2-supercomp/tree/v0.6-cuda-fusao) | [Relatório técnico](resultados/issue-10/relatorio.md), [comparação](resultados/issue-10/comparacao.txt), [Nsight Systems](resultados/issue-10/resumo_nsys.txt) e [tentativa Nsight Compute](resultados/issue-10/resumo_ncu.txt) |

## Análise dos resultados

O gprof atribuiu 65,85% do tempo amostrado do programa original ao Backward, com 584.919 chamadas. A primeira melhoria reorganizou o cálculo de grad_in em grupos de oito entradas. O ganho foi medido; cache misses e vetorização não foram medidos diretamente.

O pré-teste com OpenMP no Backward teve 0,4587 s com uma thread e 2,6753 s com duas, em uma execução por configuração. As regiões paralelas muito curtas e frequentes podem explicar o custo adicional. O experimento foi registrado e o código de treinamento sequencial foi restaurado antes do marco v0.3.

Na avaliação, cada imagem é independente depois que os pesos estão fixos. Uma única região OpenMP distribui esse trabalho, enquanto a coleta sequencial mantém a ordem das métricas. Como a avaliação já era curta em relação ao treinamento, seu ganho tem efeito pequeno no tempo conjunto dessas etapas, em linha com a limitação descrita pela Lei de Amdahl.

Na migração CUDA inicial, cada camada devolvia resultados à CPU. As cópias síncronas e os lançamentos repetidos custavam muito em relação aos pequenos cálculos. A versão otimizada mantém o lote na GPU, coopera dentro de cada amostra e usa memória compartilhada para pesos e intermediários durante uma época. As linhas dos pesos têm uma posição extra para distribuir as leituras entre bancos. As barreiras garantem o uso dos pesos antigos no cálculo dos gradientes e a conclusão das atualizações antes da próxima amostra.

O treinamento continua SGD por amostra, na ordem original. Um bloco por época evita a necessidade de sincronizar blocos para essas dependências, mas limita o paralelismo disponível no treino. Não estamos treinando imagens simultaneamente. Na avaliação, os pesos ficam fixos e um único lançamento distribui as imagens entre blocos.

### Profiling GPU: mesma carga de uma época mais avaliação

| Métrica do Nsight Systems | CUDA inicial | CUDA otimizada |
|:---|---:|---:|
| Lançamentos de kernels | 51.267 | 2 |
| Chamadas cudaMemcpy | 56.664 | 11 |
| Tempo acumulado de kernels na GPU | 90,763354 ms | 12,390191 ms |
| Tempo acumulado das cópias na GPU | 45,050417 ms | 0,129405 ms |

As duas execuções com profiler usaram uma época e a avaliação completa. Seus tempos sofrem instrumentação e não substituem as médias sem profiler de 51 épocas. Os percentuais de cada relatório têm seu próprio denominador; não são percentuais do tempo total da aplicação. O tempo da API cudaMemcpy pode incluir espera por kernels e não equivale ao tempo de transmissão na GPU. A redução observada não comprova saturação da largura de banda PCIe na versão inicial.

Na versão otimizada, a primeira inicialização/alocação também pesa no perfil completo. As alocações das camadas ocorrem antes do cronômetro de treinamento. O alto percentual de cudaMalloc no relatório de APIs não significa que esse percentual do treino foi gasto alocando memória por amostra.

O Nsight Compute foi tentado para coletar LaunchStats, Occupancy e SpeedOfLight, mas retornou ERR_NVGPUCTRPERM por falta de acesso aos contadores. Portanto, não apresentamos ocupação real, throughput de hardware ou um roofline medido. Não atribuímos o resultado a métricas que não foram coletadas.

## Corretude e metodologia

- CPU em blocos: cinco saídas impressas coincidiram com a referência, excluindo o tempo.
- OpenMP na avaliação: dez saídas impressas coincidiram com a referência, excluindo os tempos.
- CUDA inicial e otimizada: cinco treinos completos por versão passaram na comparação com a CPU de perdas por época, 17.970 escores, 2.778 parâmetros e classes previstas. A tolerância numérica é erro absoluto de 1e-5 mais erro relativo de 1e-5; todas as classes previstas devem coincidir.
- Referência final: 95,5% de acurácia, 1.717 acertos em 1.797 imagens e perda média de 0,0432. Nos testes de uma época, a acurácia é diferente; a comparação usa CPU com a mesma quantidade de épocas.
- Treino: mede o processamento das épocas; leitura dos arquivos e criação/inicialização das camadas ficam fora. No modo otimizado, empacotamento, alocações dos buffers do lote e transferências de treinamento ficam dentro.
- Avaliação: inclui inferência, métricas e impressão; no modo GPU otimizado inclui preparação e transferências do lote. A validação contra a CPU é feita depois dos cronômetros.
- As execuções foram processos novos, sem uma fase separada de aquecimento de kernels. Os tempos refletem também o primeiro lançamento e os custos de runtime que ocorrem dentro dos intervalos medidos.
- CPU: GCC 14.2.0, -std=c++17 -O2 -g; na avaliação paralela acrescentamos -fopenmp. CUDA: toolkit 12.8.1, -std=c++17 -O2 -lineinfo --fmad=false. Nsight Systems 2024.6.2.
- perf e Valgrind não estavam disponíveis nos nós CPU testados, e foi usado gprof como alternativa. Não há medição própria de cache misses. O arquivo Callgrind antigo herdado no repositório não é utilizado como evidência deste trabalho.

## Como compilar e executar no Franky

Execute a partir da pasta do repositório, onde estão os dois arquivos optdigits. Para a versão final:

```bash
g++ -std=c++17 -O2 -g -fopenmp barenn.cpp -o barenn_cpu_final
module load cuda/12.8.1
nvcc -std=c++17 -O2 -lineinfo --fmad=false barenn_cuda.cu -o barenn_cuda
```

CPU com avaliação em duas threads:

```bash
srun --partition=merry_cpu --ntasks=1 --cpus-per-task=2 --mem=1G env OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE OMP_PLACES=cores OMP_PROC_BIND=TRUE ./barenn_cpu_final
```

CUDA otimizada com 51 épocas e validação:

```bash
srun --partition=merry_gpu --gres=gpu:1 --ntasks=1 --cpus-per-task=1 --mem=1G ./barenn_cuda --otimizado --validar
```

Teste curto: acrescente --epocas 1. O modo --cpu executa a referência sequencial; sem --cpu e sem --otimizado, executa o caminho CUDA inicial. Para a comparação exata dos marcos, use as respectivas tags: o binário CUDA inicial das medições foi compilado a partir da v0.5.

Para verificar um marco, faça git checkout seguido do nome da tag em uma cópia limpa do repositório. Nas tags v0.1 a v0.3, compile barenn.cpp com g++ -std=c++17 -O2 -g. A partir da v0.4, use também -fopenmp para habilitar a avaliação paralela. O arquivo barenn_cuda.cu existe a partir da v0.5; --otimizado existe a partir da v0.6. Ao terminar, volte ao ramo principal master.

### Reproduzir a comparação controlada

Compile o executável inicial na tag v0.5 e preserve-o como barenn_cuda_inicial. Volte à versão final e compile barenn_cuda_otimizado com as mesmas flags CUDA. Dentro de uma alocação merry_gpu com 1 GPU, 1 processo, 1 CPU e 1 GB, execute cinco rodadas alternadas:

```bash
./barenn_cuda_otimizado --cpu
./barenn_cuda_inicial
./barenn_cuda_otimizado --otimizado --validar
```

As saídas completas das cinco rodadas estão em resultados/issue-10. Para reproduzir a avaliação multicore, alterne OMP_NUM_THREADS=1 e OMP_NUM_THREADS=2 do mesmo executável CPU numa alocação de 2 CPUs, mantendo OMP_DYNAMIC=FALSE, OMP_PLACES=cores e OMP_PROC_BIND=TRUE.

### Reproduzir profiling

No nó GPU alocado, para a versão otimizada:

```bash
nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true --output=perfil_cuda ./barenn_cuda --otimizado --epocas 1
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum perfil_cuda.nsys-rep
ncu --section LaunchStats --section Occupancy --section SpeedOfLight --launch-count 1 ./barenn_cuda --otimizado --epocas 1
```

Para o perfil inicial, use o executável da v0.5 sem --otimizado. Os resumos de texto e o erro de permissão do Nsight Compute estão versionados. Os relatórios binários do profiler podem ser regenerados pelos comandos acima.

Para gprof, compile a versão CPU desejada com -std=c++17 -O2 -g -pg, execute no nó CPU e gere o relatório com gprof -b -p EXECUTAVEL gmon.out. O tempo de uma execução instrumentada não substitui as medições sem profiler.

## Referências de estudo

- [Estudo de caso e profiling em GPU](https://insper.github.io/supercomp/aulas/profile-nvidia/): referência para a organização da migração e da cooperação entre threads numa rede pequena.
- [Documentação do nvcc 12.8](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-compiler-driver-nvcc/index.html): opção --fmad=false.
- [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/).
- [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).

---

# Projeto Barenn: O Legado, a Sobrevivência e o Salto para a GPU

Era uma segunda-feira aparentemente tranquila quando a demanda chegou à sua mesa. A liderança técnica se aproximou com um sorriso contido e soltou aquela frase clássica que faz qualquer engenheiro gelar por dentro:

> "Temos um componente crítico rodando em produção há anos. O desenvolvedor original já saiu da empresa, mas agora o projeto é oficialmente seu."
    
Você procurou documentação, manuais ou notas de arquitetura e não encontrou rigorosamente nada. Restou apenas um código hermético a correr em produção.

Depois de analisar o código entendeu que é a implementação de uma Feedforward Neural Network, mas por que razão reinventar a roda? No passado, exigências estritas de conformidade e restrições de licenciamento impediram o recurso a bibliotecas de mercado. Sem alternativa, a empresa optou por criar a sua própria implementação de raiz. O resultado funcionou durante anos, mas tornou-se um labirinto técnico esquecido.

Com o aumento exponencial do volume de dados, a máquina atual começou a falhar nos prazos operacionais. Antes de qualquer migração complexa, **o seu primeiro desafio é estabilizar o código e espremer todo o desempenho possível da CPU — atacando gargalos, repensando o acesso à memória e explorando o paralelismo nativo.**

Para resolver o problema a longo prazo, a empresa adquiriu servidores blade equipados com GPUs corporativas NVIDIA. O hardware de alto desempenho já está montado no datacenter: **a sua missão final é reescrever os pontos críticos do algoritmo em CUDA e levar o processamento ao nível do paralelismo massivo.**

---

## Diretrizes de Conduta e Governança no Repositório

Neste projeto, o repositório não é apenas um local de entrega de arquivos, mas a **comprovação do seu processo investigativo de engenharia**. Espera-se uma postura estritamente empírica: nenhuma otimização deve ser feita no escuro e nenhuma melhoria de velocidade tem valor se corromper o resultado final do programa.

---

### O Princípio do Bloco Lógico de Trabalho

Você tem autonomia para decidir **o que** otimizar e **em qual ordem**, mas cada entrega deve ser estruturada como um **bloco lógico fechado**.

> **O que é um bloco lógico?**
> É uma unidade completa de intervenção técnica que engloba: **um problema identificado por profiling $\rightarrow$ uma hipótese de melhoria $\rightarrow$ a alteração no código $\rightarrow$ a validação da corretude $\rightarrow$ a medição do ganho real.**

* **Não crie tags por tempo decorrido:** Evite tags como `versao-dia-2` ou commits de "fim de expediente".
* **Não crie tags para alterações inconclusivas:** Se a alteração piorou o desempenho ou quebrou os testes, documente na Issue, reverta ou ajuste antes de formalizar o marco.
* **Toda tag deve ser reprodutível:** O professor ou qualquer colega deve conseguir executar `git checkout <tag>`, compilar o código e obter exatamente o mesmo comportamento e ordem de grandeza de desempenho documentados.

---

### O Fluxo Esperado no GitHub

Para cada intervenção técnica planejada, seu ciclo de trabalho no repositório deve seguir três pilares:

#### 1. Abertura de Issue (A Hipótese e o Ponto de Partida)

Antes de alterar uma única linha de código, abra uma Issue explicando a intenção do bloco lógico:

* **Diagnóstico:** Qual gargalo você encontrou usando ferramentas de análise (profilers)?
* **Hipótese:** Qual conceito técnico você pretende aplicar para resolver esse gargalo?
* **Métrica Base:** Qual é o tempo ou custo de processamento atual dessa rotina?

#### 2. Desenvolvimento Isolado e Commits Semânticos

* Trabalhe em uma *branch* dedicada àquela hipótese.
* Faça commits descritivos focados na causa da mudança (ex: `perf: ...`, `refactor: ...`, `fix: ...`).
* Certifique-se de que a saída do modelo/algoritmo continua correta. Código rápido que calcula resultados errados é considerado código quebrado.

#### 3. Formalização com Tag Anotada e Fechamento da Issue

Quando o bloco estiver concluído e validado, crie uma **Git Tag anotada**:

* A tag marca o estado exato da base de código naquele degrau de maturidade.
* A mensagem da tag deve resumir sucintamente o resultado alcançado.
* Feche a Issue anexando as evidências (dados de tempo, prints do profiler ou tabelas comparando o "antes e depois").

---

### Exemplo Ilustrativo de um Bloco Lógico

*(Este é apenas um exemplo de como organizar a narrativa técnica e as ferramentas; não representa um passo obrigatório ou a solução do seu problema).*

---

#### 1. Cenário: O aluno percebeu lentidão na etapa de processamento sequencial

* **Abertura da Issue #04:**
> **Título:** *Redução de tempo de computação na rotina de cálculo vetorial*
> **Contexto:** Ao inspecionar o código com o profiler, identificou-se que 65% do tempo de execução é consumido em um laço interno que realiza operações matemáticas repetitivas.
> **Métricas atuais:** Tempo total de execução: `14.2s`.
> **Hipótese:** Reestruturar a ordem dos dados para favorecer o uso de registradores e paralelizar o laço com múltiplos núcleos deve reduzir o tempo dessa rotina.


* **Implementação na branch:** `feature/otimizacao-laco-vetorial`
* Commits atômicos ajustando o laço e aplicando a estratégia escolhida.
* Verificação de que o resultado final do programa bate com o original.


* **Criação da Tag:**
```bash
git tag -a v0.3-laco-paralelo -m "Otimizacao do calculo vetorial: tempo reduzido de 14.2s para 4.8s (Speedup 2.95x). Corretude validada."
git push origin v0.3-laco-paralelo

```


* **Fechamento da Issue:**
* O aluno responde na Issue #04 anexando o comparativo de tempo (antes: `14.2s` / depois: `4.8s`), faz o merge da branch e referencia a tag criada:


> *"Otimização implementada e validada na tag `v0.3-laco-paralelo`. Issue fechada pelo PR #05."*


---

# Rubrica de Avaliação (Total: 10,0 Pontos)

<div style="background:rgba(255, 0, 0, 0.65)"> 

> ### ⚠️ CLÁUSULA ELIMINATÓRIA GERAL (CONDIÇÃO DE NOTA ZERO)
> 
> 
> A avaliação adota uma política estrita de controle de qualidade de engenharia:
> **Se em QUALQUER um dos critérios ou subcritérios de avaliação for observado o indicativo de `0,0 ponto`, a ATIVIDADE SERÁ INTEGRALMENTE ZERADA (Nota Final = 0,0).**
> Não é tolerada a ausência completa de qualquer pilar fundamental da atividade (seja por código inoperante/incorreto, falta de governança no Git ou ausência de embasamento empírico nas Issues).

* **Implementação e Desempenho Técnico:** 4,0 pontos
* **Governança, Reprodutibilidade e Tags:** 2,0 pontos
* **Relatório Técnico nas Issues (Análise e Discussão):** 4,0 pontos

</div>

---

### Critério 1: Implementação e Desempenho Técnico (4,0 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — O código não compila, aborta durante a execução, não é funcional ou gera saídas matematicamente corrompidas.
* **+0,5 ponto** — Executou com sucesso o código legado, mediu o baseline de referência de forma reproduzível e identificou os gargalos iniciais.
* **+1,0 ponto** — Aplicou otimizações eficazes na CPU com ganhos consistentes de velocidade e paralelismo multicore preservando a integridade numérica.
* **+1,0 ponto** — **Migração Funcional para CUDA:** Porting inicial para GPU completo e correto (kernel funcional, alocação e liberação de memória no Device e transferências Host/Device íntegras).
* **+0,75 ponto** — Otimizou a execução em GPU.
* **+0,75 ponto** — Implementou técnicas avançadas em GPU, comprovando ganhos.

---

### Critério 2: Governança, Reprodutibilidade e Tags (2,0 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — Ausência de tags, ou repositório com commit único ("tudo pronto") / commits lineares sem separação lógica, inviabilizando a reprodutibilidade dos marcos intermediários.
* **+0,5 ponto** — Utilizou tags, mas elas marcam datas/horários arbitrários em vez de blocos lógicos fechados de trabalho, ou as tags não compilam de forma independente.
* **+1,2 ponto** — Trabalho estruturado em **blocos lógicos**: cada intervenção possui uma Git Tag anotada associada a uma Issue, permitindo compilar e checar a métrica de cada versão via `git checkout <tag>`.
* **+2,0 pontos** — Reprodutibilidade e histórico impecáveis: commits atômicos com mensagens semânticas (`perf:`, `refactor:`, `test:`), branches limpas integradas via Pull Requests documentados e rastreabilidade total de cada degrau de performance.

---

### Critério 3: Relatório Técnico nas Issues (Análise e Discussão) (4,0 pontos)

#### 3.1 Profiling e Evidências Empíricas (1,5 ponto)

* **0,0 ponto [ZERA A ATIVIDADE]** — Não apresenta dados de medição nas Issues ou baseia-se unicamente em impressões subjetivas sem qualquer dado empírico.
* **+0,5 ponto** — Apresenta apenas tempos brutos de relógio (*wall-clock*) colados em texto corrido, sem dados de profilers ou repetições estatísticas.
* **+1,0 ponto** — Apresenta dados e capturas das ferramentas oficiais (**perf/callgrind** e **Nsight Systems/Compute**), mas faltam informações cruciais de ambiente (flags, parâmetros de teste ou hardware).
* **+1,5 ponto** — Issues contêm relatórios completos de profiling (CPU e GPU), métricas de hardware detalhadas (tempo de kernel vs transferência PCIe, cache misses, ocupação teórica vs real de SMs) e metodologia experimental rigorosa.

#### 3.2 Qualidade da Discussão Técnica e Argumentação de Engenharia (2,5 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — Discussão inexistente, Issues fechadas apenas com "ok/concluído" ou explicações inconsistentes/conflitantes com os dados coletados.
* **+1,0 ponto** — Discussão puramente descritiva (lista o que foi alterado nas linhas de código, mas não analisa *por que* houve ou não ganho com base em hardware).
* **+1,8 ponto** — Discute tecnicamente a relação entre o código e o comportamento de hardware (ex: impacto do layout de memória nos caches da CPU, balanceamento de carga entre threads).
* **+2,5 pontos** — Análise técnica aprofundada: correlaciona os resultados com conceitos avançados de arquitetura (roofline model, limites da Lei de Amdahl, latência do PCIe vs throughput do kernel CUDA) e debate criticamente as decisões de engenharia tomadas.

---

## Tabela de Cenários e Balizadores

| Cenário | Situação Técnica no GitHub | Resultado / Nota |
| --- | --- | --- |
| **Desclassificado** | Tirou `0,0` em qualquer critério (ex: código não compila, sem tags, sem dados de profiling nas issues ou issues vazias). | **0,0 (Zerada)** |
| **Abaixo do Esperado** | Código compila e funciona, mas há otimizações superficiais, profiling fraco e tags incompletas. | **0,1 a 5,9** |
| **Linha de Base (Alvo: 6,0)** | CPU razoavelmente otimizada + **Migração simples para CUDA funcional** + Issues estruturadas com diagnóstico e tags atreladas. | **6,0 a 6,5** |
| **Avançado / Destaque** | CPU altamente eficiente + **CUDA com otimizações avançadas (Shared Memory/Streams)** + discussões ricas nas Issues com Nsight/Perf + governança impecável. | **8,5 a 10,0** |

---

## Formato de Entrega

A entrega consiste exclusivamente no **link do repositório no GitHub**.

1. O `README.md` principal deve atuar como o **sumário executivo do relatório**, sendo adiciocionado **NO TOPO DO README**:
* A tabela consolidada final comparando todas as versões (Baseline Legado, CPU Otimizada, CUDA Inicial, CUDA Otimizado);
* Um **índice navegável com links diretos** para cada uma das Issues/Tags que contêm a discussão detalhada;

2. Todas as branches de trabalho devem estar integradas à `main`;
3. Todas as Issues de experimentos devem estar fechadas e associadas à sua respectiva Git Tag anotada.
