

**Recuperação Densa de Passagens com Modelagem de Tópicos BERTopic Aplicada ao Corpus do Blog ClickHouse**

**GUILHERME ESTEVES MARRET¹, JOÃO PAULO MARTINS DE BRITO¹, MATHEUS CASTRO ALEXANDRE¹, LEONARDO DE LELIS ROSSI¹**

guimarret@gmail.com, joao.paulomartins107@gmail.com, castro3filho@gmail.com, leoboralelis@gmail.com

¹Faculdade de Tecnologia de Jundiaí  
Jundiaí - SP

**Resumo:** A busca por informações em corpora técnicos extensos é frequentemente limitada por sistemas baseados em palavras-chave, que falham ao recuperar conteúdo semanticamente relevante quando os termos exatos não estão presentes no texto-fonte. Este trabalho apresenta um sistema de Recuperação Densa de Passagens (Dense Passage Retrieval — DPR) aplicado ao corpus do blog técnico ClickHouse, composto por 791 artigos, com o objetivo de oferecer busca semântica sobre o conhecimento ali registrado. O pipeline implementado segmenta os textos em 6.171 trechos de até 512 tokens, gera embeddings densos de 1.024 dimensões com o modelo BAAI/bge-large-en-v1.5, organiza os trechos em 73 tópicos automaticamente descobertos via BERTopic (UMAP + HDBSCAN + KeyBERTInspired), e atribui rótulos descritivos a cada tópico via Claude Haiku 4.5. A recuperação inicial por similaridade cosseno é refinada por um re-ranqueador cruzado (BAAI/bge-reranker-large), formando um pipeline em dois estágios. O sistema foi avaliado em um conjunto de 30 consultas curadas com verdade-fundamental ao nível de URL do artigo. Os resultados mostraram recall@10 de 0,933 e MRR de 0,662 na configuração com re-ranqueamento — um ganho de +10 pontos percentuais em relação à recuperação puramente densa (recall@10 de 0,833). O diferencial do trabalho está na combinação de DPR com modelagem de tópicos explicável e preservação de metadados temporais e de versão por trecho, viabilizando citação contextual e abrindo caminho para futuras consultas filtradas.

**Palavras Chaves:** Recuperação Densa de Passagens, Modelagem de Tópicos, BERTopic, BGE, Cross-Encoder, ClickHouse.

***Abstract:*** *Information retrieval over large technical corpora is often limited by keyword-based systems that fail to surface semantically relevant content when exact terms are absent from the source text. This work presents a Dense Passage Retrieval (DPR) system applied to the ClickHouse technical blog corpus of 791 articles, aimed at semantic search over the embedded knowledge. The implemented pipeline splits texts into 6,171 chunks of up to 512 tokens, generates 1,024-dimensional dense embeddings using BAAI/bge-large-en-v1.5, organizes chunks into 73 automatically discovered topics via BERTopic (UMAP + HDBSCAN + KeyBERTInspired), and assigns descriptive labels using Claude Haiku 4.5. Initial cosine-similarity retrieval is refined by a cross-encoder reranker (BAAI/bge-reranker-large), forming a two-stage pipeline. The system was evaluated on a curated 30-query set with article-level URL ground truth. Results show recall@10 of 0.933 and MRR of 0.662 under reranking — a +10 percentage-point gain over dense-only retrieval (recall@10 of 0.833). The contribution lies in combining DPR with explainable topic modeling and per-chunk preservation of temporal and version metadata, enabling contextual citation and future filtered queries.*

***Keywords:*** *Dense Passage Retrieval, Topic Modeling, BERTopic, BGE, Cross-Encoder, ClickHouse.*

1. # **Introdução**

A crescente quantidade de documentação técnica publicada em blogs corporativos representa um desafio para a recuperação de informação: usuários precisam localizar conhecimento sobre características específicas, comparações de desempenho e padrões de uso em meio a centenas ou milhares de artigos. Sistemas de busca por palavras-chave, como o BM25 \[Robertson e Zaragoza, 2009\], oferecem boa precisão em consultas literais, mas falham quando o usuário formula a pergunta com termos diferentes dos utilizados no texto-fonte — o chamado problema do *vocabulary mismatch* \[Furnas et al., 1987\].

A Recuperação Densa de Passagens (Dense Passage Retrieval — DPR) surge como alternativa neuronal a esse problema \[Karpukhin et al., 2020\]. No paradigma DPR, consulta e passagem são codificadas independentemente em vetores de alta dimensionalidade por um modelo de linguagem pré-treinado, e a relevância é medida pela similaridade entre os vetores resultantes. Esse mapeamento captura paráfrases, sinônimos e relações semânticas inacessíveis a abordagens lexicais.

Os modelos da família BGE \[Xiao et al., 2024\] representam o estado da arte atual em codificadores *bi-encoder* para o inglês, oferecendo embeddings de 1.024 dimensões com forte desempenho na benchmark MTEB. Para a etapa de re-ranqueamento, modelos de atenção cruzada (cross-encoders) que processam consulta e passagem conjuntamente apresentam ganhos consistentes em recall e precisão a custo computacional maior \[Nogueira e Cho, 2019\]. Em paralelo, a modelagem de tópicos via BERTopic \[Grootendorst, 2022\] oferece organização não-supervisionada dos documentos em grupos semanticamente coerentes, com rótulos interpretáveis derivados das palavras mais distintivas (c-TF-IDF) e refinados por re-ranqueamento neural de palavras-chave (KeyBERTInspired).

A motivação deste trabalho é integrar essas três abordagens — DPR com bi-encoder BGE, re-ranqueamento por cross-encoder e modelagem de tópicos BERTopic — em um único pipeline de recuperação aplicado a um corpus técnico real (791 artigos do blog ClickHouse), com avaliação quantitativa por meio de um conjunto de consultas curadas. O diferencial em relação a sistemas DPR clássicos está em (i) anexar a cada trecho recuperado metadados temporais (data de publicação) e de versão do software citado, viabilizando citações contextuais e consultas filtradas; (ii) atribuir a cada trecho um rótulo de tópico gerado por modelo de linguagem, oferecendo explicabilidade sem comprometer o paradigma de recuperação por similaridade vetorial.

*Este artigo encontra-se organizado da seguinte forma: a seção 2 apresenta a fundamentação teórica dos modelos utilizados. A seção 3 descreve a arquitetura do trabalho proposto. A seção 4 detalha a metodologia de avaliação. Os resultados são apresentados e discutidos na seção 5, e as conclusões na seção 6. A seção 7 declara as ferramentas de IA empregadas.*

2. # **Fundamentação Teórica**

   1. ## **Recuperação Densa com Bi-Encoders**

Um bi-encoder produz duas representações vetoriais separadas — uma para a consulta e outra para a passagem — de modo que a similaridade pode ser pré-computada por simples produto interno (ou cosseno, se os vetores forem L2-normalizados). Esse desacoplamento permite indexar previamente todas as passagens do corpus e responder a uma consulta em tempo proporcional ao número de passagens, escalando para milhões de documentos.

O modelo bge-large-en-v1.5 \[Xiao et al., 2024\] aplicado neste trabalho tem aproximadamente 335 milhões de parâmetros, produz vetores de 1.024 dimensões e aceita janelas de até 512 tokens. Adota a convenção assimétrica do BGE: passagens são codificadas sem prefixo, enquanto consultas recebem o prefixo *"Represent this sentence for searching relevant passages: "*. Os vetores resultantes são L2-normalizados, tornando o cosseno equivalente ao produto escalar.

   2. ## **Re-ranqueamento com Cross-Encoders**

O custo da economia computacional do bi-encoder é a perda de interações cruzadas: consulta e passagem nunca compartilham camadas de atenção. Um cross-encoder, em contraste, recebe o par concatenado e produz um único escore de relevância, permitindo que a atenção se especialize sobre correspondências locais, negações e dependências sintáticas \[Nogueira e Cho, 2019\]. Cross-encoders são proibitivamente caros para a etapa inicial de busca (uma inferência por par consulta-passagem sobre o corpus completo), mas eficientes para refinar os k primeiros resultados retornados pelo bi-encoder, em pipelines de dois estágios.

   3. ## **Modelagem de Tópicos com BERTopic**

BERTopic \[Grootendorst, 2022\] organiza um corpus em tópicos coerentes em quatro estágios: (i) projeção dos embeddings densos para baixa dimensionalidade via UMAP \[McInnes e Healy, 2018\]; (ii) clusterização por densidade via HDBSCAN \[McInnes et al., 2017\], que diferentemente do k-means não exige número predefinido de clusters e identifica pontos como *outliers*; (iii) extração de palavras-chave por tópico via c-TF-IDF — uma adaptação de TF-IDF que trata cada cluster como um documento; (iv) opcionalmente, refinamento das palavras-chave por re-ranqueamento neural (KeyBERTInspired) e rotulação por modelo de linguagem.

3. # **O Trabalho Proposto**

A hipótese deste trabalho é a de que um pipeline combinando bi-encoder BGE, BERTopic e re-ranqueamento por cross-encoder pode oferecer recuperação semântica de alta qualidade sobre um corpus técnico real, com explicabilidade adicional vinda da modelagem de tópicos e preservação de metadados que viabilizem citações contextuais.

   1. ## **Arquitetura**

O pipeline implementado consiste em seis etapas, ilustradas na Figura 1.

```
Sitemap clickhouse.com/blog
        │ trafilatura
        ▼
   791 artigos canônicos → blog.parquet
        │ chunker recursivo (512 tokens, 64 de overlap)
        ▼
   6.171 trechos → blog_chunks.parquet
        │ bge-large-en-v1.5 (1024-d, prefixo assimétrico)
        ▼
   blog_chunks_embedded.parquet
        │ BERTopic (UMAP + HDBSCAN + KeyBERTInspired)
        ▼
   73 tópicos descobertos
        │ Claude Haiku 4.5 (uma sentença por tópico)
        ▼
   topics.parquet + chunks_topics.parquet
        │ consulta do usuário
        ▼
   consulta → BGE-Q → similaridade cosseno → top-50
        │ bge-reranker-large (cross-encoder)
        ▼
   top-k com data, versão e rótulo de tópico
```

**Figura 1 — Arquitetura do pipeline de recuperação densa proposto.**

   2. ## **Tecnologias Utilizadas**

A Tabela 1 sintetiza as principais tecnologias empregadas no pipeline e a função de cada uma.

**Tabela 1 — Tecnologias empregadas em cada etapa do pipeline.**

| Componente | Tecnologia | Função |
| :---- | :---- | :---- |
| Extração de texto | trafilatura | Limpeza de HTML para texto |
| Segmentação | Tokenizer BGE \+ *splitter* recursivo | Trechos de até 512 tokens com 64 de sobreposição |
| Codificador (bi-encoder) | BAAI/bge-large-en-v1.5 | Embeddings densos de 1.024 dimensões |
| Modelagem de tópicos | BERTopic 0.17 \+ UMAP \+ HDBSCAN | Descoberta automática de 73 tópicos |
| Representação de tópicos | KeyBERTInspired sobre c-TF-IDF | Palavras-chave reordenadas por relevância |
| Rotulação de tópicos | Anthropic Claude Haiku 4.5 (via API) | Sentenças descritivas (uma por tópico) |
| Re-ranqueador | BAAI/bge-reranker-large | Refinamento dos 50 melhores candidatos |
| Armazenamento | Apache Parquet | Tabelas de trechos, embeddings e tópicos |
| Interface | Typer (CLI em Python) | Comandos `chunk`, `embed`, `fit-topics`, `query`, `eval` |

   3. ## **Aspectos de Reprodutibilidade**

Para garantir reprodutibilidade entre execuções, a semente do UMAP foi fixada em 42, e cada execução do BERTopic é serializada em um diretório datado e imutável (`topic_model_AAAA-MM-DD/`), contendo o modelo em formato safetensors, as tabelas de tópicos e trechos, a hierarquia de tópicos e um manifesto com os parâmetros usados. O ambiente é gerenciado pela ferramenta `uv`, com bloqueio de versões em `uv.lock`.

4. # **Materiais e Métodos**

   1. ## **Geração do Conjunto de Avaliação**

Para validação científica da recuperação, construiu-se um conjunto de 30 pares pergunta-resposta. A seleção dos trechos-semente seguiu critério estratificado: para cada um dos 30 tópicos com maior número de trechos, foi escolhido o trecho-medóide — aquele com maior similaridade cosseno ao centróide do tópico. Em seguida, o modelo Claude Haiku 4.5 foi instruído a gerar, para cada trecho, uma pergunta realista de usuário (5 a 15 palavras, com paráfrase obrigatória das frases distintivas do texto-fonte) cuja resposta estivesse contida na passagem.

As 30 perguntas geradas passaram então por revisão manual. Três perguntas originalmente em japonês — provenientes de tópicos derivados de versões traduzidas dos artigos — foram traduzidas para o inglês para manter o conjunto monolíngue. Em quatro casos, a URL de verdade-fundamental foi re-apontada do artigo japonês para sua versão inglesa equivalente, presente no corpus, de modo a evitar penalizar o codificador (treinado em inglês) por incompetência cross-lingual não relacionada à qualidade da recuperação.

   2. ## **Critério de Verdade-Fundamental**

A relevância foi definida ao nível de URL do artigo (*post-level relevance*): uma consulta é considerada "acerto" no rank `r` se qualquer trecho dentre os primeiros `r` recuperados compartilhar o mesmo `source_url` que o trecho-semente. Essa granularidade é mais branda do que correspondência exata de `chunk_id` e adequada a uma tarefa de recuperação ao nível de documento, evitando penalizar o sistema por retornar trechos vizinhos do mesmo artigo.

   3. ## **Métricas**

Foram computadas as métricas padrão de recuperação:

* **recall@k** (k ∈ {1, 3, 5, 10}): fração de consultas em que ao menos um trecho relevante aparece nos primeiros `k` resultados;
* **MRR** (Mean Reciprocal Rank): média do recíproco do rank da primeira ocorrência relevante, com valor 0 quando ausente do top-10.

   4. ## **Configurações Comparadas**

Duas configurações foram avaliadas sobre o mesmo conjunto de 30 consultas:

1. **Recuperação densa simples**: similaridade cosseno do BGE bi-encoder sobre os 6.171 trechos do corpus, retornando os top-k diretamente.
2. **Recuperação densa com re-ranqueamento**: BGE bi-encoder retorna os 50 melhores candidatos, e o cross-encoder bge-reranker-large rescora esses 50 pares (consulta, trecho), produzindo a ordenação final.

   5. ## **Ambiente de Execução**

Todos os experimentos foram executados em estação local com processador AMD Ryzen, 64 GB de memória RAM e GPU AMD Radeon RX 6700 XT (12 GB de VRAM, com ROCm 7.0). A etapa de geração de embeddings utilizou a GPU; o ajuste do BERTopic foi executado em CPU para evitar contenção de memória de vídeo com outros processos. As chamadas ao Claude Haiku 4.5 foram feitas via API oficial da Anthropic.

5. # **Resultados e Discussão**

   1. ## **Métricas de Recuperação**

Os resultados quantitativos das duas configurações são apresentados na Tabela 2.

**Tabela 2 — Métricas de recuperação no conjunto de 30 consultas.**

| Métrica | Densa | Densa \+ Re-rank | Δ |
| :---- | ----: | ----: | ----: |
| recall@1 | 0,533 | 0,500 | −0,033 |
| recall@3 | 0,700 | 0,800 | \+0,100 |
| recall@5 | 0,733 | 0,867 | \+0,133 |
| recall@10 | 0,833 | 0,933 | \+0,100 |
| MRR | 0,628 | 0,662 | \+0,034 |
| Consultas sem acerto (de 30\) | 5 | 2 | −3 |

   2. ## **Análise dos Resultados**

A configuração com re-ranqueamento apresentou ganhos substanciais em recall a partir de `k=3`, com pico de +13,3 pontos percentuais em recall@5 e +10 pontos em recall@10, atingindo 93,3% de cobertura no top-10. O número de consultas sem qualquer acerto entre os dez primeiros resultados caiu de cinco para duas.

Observa-se também uma leve queda de 3,3 pontos percentuais em recall@1 quando o re-ranqueador é aplicado. Esse efeito é conhecido na literatura \[Nogueira e Cho, 2019\]: cross-encoders, ao re-ordenar uma janela ampla, podem deslocar para a primeira posição um trecho marginalmente mais relevante de um artigo distinto daquele do trecho-semente, transformando um acerto em recall@1 em um acerto somente a partir de recall@3. Para tarefas em que o documento de origem único é estritamente crítico, essa troca pode ser indesejável; para tarefas de recuperação ao nível de documento, o ganho em ranks mais profundos compensa amplamente a perda.

   3. ## **Análise das Duas Consultas Sem Acerto**

Os dois erros residuais sob re-ranqueamento foram inspecionados manualmente:

1. *"How can you optimize ClickHouse trace queries with materialized views and indexing strategies?"* — a verdade-fundamental aponta para um artigo específico sobre armazenamento de *traces* com OpenTelemetry, porém a consulta é genérica o suficiente para que outros artigos sobre otimização e *views* materializadas sejam recuperados em posições superiores. Trata-se de ambiguidade legítima da verdade-fundamental.
2. *"What new feature became production-ready in ClickHouse version 24.10?"* — a semente está em um boletim mensal que resume *features* da versão 24.10, enquanto o sistema corretamente prioriza o artigo de *release notes* oficial da própria 24.10, considerado pelo cross-encoder como resposta mais direta à pergunta.

Em ambos os casos, a falha é menos uma limitação do codificador e mais um indicativo de que a verdade-fundamental, ao nível de URL único, é simplificação útil mas imperfeita.

   4. ## **Custos e Tempos**

O processo completo das etapas 1 a 4 do pipeline (segmentação, embedding, ajuste de tópicos e rotulação) levou aproximadamente:

* Segmentação: 32 segundos (CPU);
* Geração de embeddings: 5 minutos e 16 segundos (GPU, cerca de 1,65 s por *batch* de 32 trechos);
* Ajuste do BERTopic: cerca de 1 minuto (CPU; UMAP 18 s, HDBSCAN abaixo de 1 s, KeyBERTInspired 35 s);
* Rotulação dos 73 tópicos via Haiku: cerca de 1 minuto e 30 segundos.

O custo monetário das 73 chamadas de rotulação de tópicos somou aproximadamente US$ 0,10 a US$ 0,15. A geração das 30 consultas-semente via Haiku consumiu cerca de US$ 0,10 adicionais. O re-ranqueamento das 30 consultas (1.500 inferências cruzadas executadas em GPU) totalizou cerca de 15 segundos.

   5. ## **Discussão Adicional**

O ganho do re-ranqueador sobre o bi-encoder está alinhado com o intervalo esperado pela literatura (+10 a +20 pontos em recall@k) e ratifica a hipótese central do trabalho: arquiteturas de dois estágios são particularmente eficazes em corpora técnicos densos, nos quais a similaridade lexical alta entre artigos sobre temas próximos (por exemplo, várias *release notes* sucessivas, comparações entre bancos de dados, otimização de consultas) confunde o bi-encoder, mas é resolvida pelo cross-encoder, que atende às nuances de cada par específico.

A modelagem de tópicos, embora não diretamente avaliada como filtro de recuperação neste trabalho (foi adotado o chamado modo A — metadado anexo ao resultado), provou-se útil para explicabilidade: cada um dos 73 tópicos recebeu rótulo descritivo gerado por LLM, e cada trecho retornado pelo `query` carrega o rótulo do tópico ao qual pertence, facilitando inspeção qualitativa pelo usuário.

6. # **Conclusões**

Este trabalho implementou e avaliou um sistema de Recuperação Densa de Passagens sobre o blog técnico ClickHouse, combinando o codificador bi-encoder BAAI/bge-large-en-v1.5, a biblioteca de modelagem de tópicos BERTopic e o re-ranqueador cruzado BAAI/bge-reranker-large. O sistema atingiu recall@10 de 93,3% e MRR de 0,662 em um conjunto de avaliação de 30 consultas curadas, com infraestrutura totalmente reprodutível.

Como pontos fortes destacam-se: (i) a arquitetura em dois estágios permite combinar a escala do bi-encoder com a precisão do cross-encoder sem custos proibitivos no tempo de consulta; (ii) a preservação de metadados temporais e de versão por trecho viabiliza citações ricas e abre caminho para futuras consultas com filtro temporal; (iii) a modelagem de tópicos adiciona explicabilidade sem prejudicar o paradigma de recuperação vetorial; (iv) o pipeline inteiro foi construído com bibliotecas de código aberto, exceto pela rotulação de tópicos, que utilizou API comercial — passível de substituição por LLM local, conforme demonstrado pelo *script* de instalação do Qwen 2.5 via Ollama incluído no repositório como alternativa.

Como limitações reconhecem-se: (i) o codificador bi-encoder utilizado é primariamente treinado em inglês, e consultas em outras línguas sobre passagens em inglês (ou vice-versa) apresentam degradação severa, conforme observado em experimentos preliminares com perguntas em japonês; (ii) o conjunto de avaliação, embora curado manualmente, foi *seeded* por geração via LLM (Haiku), o que pode introduzir viés a favor das construções típicas desse modelo; (iii) o sistema implementa apenas a etapa de recuperação, sem componente generativo de resposta (estágio "G" do paradigma RAG completo); (iv) a verdade-fundamental ao nível de URL é uma simplificação útil, mas inadequada para tarefas que exijam precisão ao nível de parágrafo.

Como recomendações para trabalhos similares: (a) utilizar codificadores multilíngues, como o bge-m3, caso o corpus contenha múltiplos idiomas; (b) construir conjunto de avaliação com perguntas reais de usuários quando possível, complementando — não substituindo — perguntas geradas por LLM; (c) considerar arquiteturas híbridas combinando BM25 com recuperação densa via *Reciprocal Rank Fusion*, para consultas dominadas por nomes próprios ou acrônimos.

Como direções futuras, propõem-se: (a) adicionar o estágio generativo de resposta, transformando o sistema em RAG completo; (b) implementar recuperação em dois estágios com filtro por tópico (modo B), avaliando se a restrição prévia ao escopo do tópico melhora a precisão; (c) persistir os embeddings em um índice vetorial dedicado (pgvector ou índice nativo do ClickHouse), eliminando a necessidade de recomputar a multiplicação matricial a cada consulta; (d) avaliar a transferência do pipeline para outros corpora técnicos de natureza semelhante.

7. # **Uso de IA**

Foram utilizadas as seguintes ferramentas de Inteligência Artificial neste trabalho:

* **Claude Code (Anthropic)** — assistente de programação utilizado durante o desenvolvimento do código do *pipeline* (geração inicial de *boilerplate*, refatoração, depuração e auxílio na elaboração da redação deste artigo, com revisão e edição finais pelo autor).
* **Claude Haiku 4.5 (Anthropic, via API)** — geração das 73 sentenças descritivas dos tópicos e das 30 perguntas-semente do conjunto de avaliação.
* **BAAI/bge-large-en-v1.5** — modelo *bi-encoder* utilizado para a geração de *embeddings* densos de consultas e passagens.
* **BAAI/bge-reranker-large** — modelo *cross-encoder* utilizado na etapa de re-ranqueamento.
* **BERTopic 0.17** — biblioteca empregada para descoberta automática de tópicos, incluindo o componente KeyBERTInspired, que utiliza internamente um *SentenceTransformer* para refinamento de palavras-chave.
* **UMAP e HDBSCAN** — algoritmos de redução de dimensionalidade e clusterização integrados ao BERTopic.

**Referências Bibliográficas**

Furnas, G. W., Landauer, T. K., Gomez, L. M. and Dumais, S. T. (1987). The vocabulary problem in human-system communication. *Communications of the ACM*, Vol. 30, No. 11, pp. 964−971\.

Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv preprint*, arXiv:2203.05794\.

Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D. and Yih, W. (2020). Dense Passage Retrieval for Open-Domain Question Answering. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 6769−6781\.

McInnes, L. and Healy, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv preprint*, arXiv:1802.03426\.

McInnes, L., Healy, J. and Astels, S. (2017). hdbscan: Hierarchical density based clustering. *The Journal of Open Source Software*, Vol. 2, No. 11, p. 205\.

Nogueira, R. and Cho, K. (2019). Passage Re-ranking with BERT. *arXiv preprint*, arXiv:1901.04085\.

Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 3982−3992\.

Robertson, S. and Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*, Vol. 3, No. 4, pp. 333−389\.

Xiao, S., Liu, Z., Zhang, P. and Muennighoff, N. (2024). C-Pack: Packed Resources For General Chinese Embeddings. In *Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval*, pp. 641−649\.
