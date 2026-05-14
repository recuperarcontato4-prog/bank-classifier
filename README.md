# Classificador de Despesas Bancarias

Categorizacao automatica de transacoes bancarias usando **Zero-Shot Classification** com Hugging Face Transformers.

Sem treinamento, sem dados rotulados — o modelo classifica textos em categorias que nunca viu.

## Arquitetura

```
Descricao da Transacao
        |
        v
+---------------------------+
| pipeline(                 |
|   "zero-shot-classification", |
|   model="mDeBERTa-v3"    |
| )                         |
+---------------------------+
        |
        v
+---------------------------+
| Categorias Candidatas:    |
| Alimentacao, Transporte,  |
| Contas, Lazer, Compras,   |
| Saude                     |
+---------------------------+
        |
        v
  Categoria + Score (%)
```

O modelo compara a descricao da transacao contra cada categoria candidata usando Natural Language Inference (NLI), retornando a probabilidade de cada uma.

## Como executar

### Google Colab (recomendado)
1. Abra `classificador_despesas.ipynb` no Google Colab
2. Execute as celulas em ordem
3. Nenhuma configuracao necessaria — Colab ja tem `transformers` instalado

### Local
```bash
git clone https://github.com/recuperarcontato4-prog/bank-classifier.git
cd bank-classifier
pip install -r requirements.txt
jupyter notebook classificador_despesas.ipynb
```

## Resultados

O classificador analisa 10 transacoes e retorna:
- **Categoria prevista** com score de confianca
- **2a opcao** para comparacao
- **Indicador de ambiguidade** (quando a diferenca entre 1a e 2a categoria e < 20%)

Exemplo:
```
Compra no supermercado Pao de Acucar  -> Alimentacao (84.2%)
Pagamento Uber viagem centro          -> Transporte  (91.3%)
Conta de luz Enel Sao Paulo           -> Contas e Servicos (88.7%)
```

## Tecnologias

- **Python 3.12**
- **Hugging Face Transformers** — pipeline de zero-shot-classification
- **mDeBERTa-v3-base-mnli-xnli** — modelo multilingual treinado em NLI
- **Pandas** — organizacao dos resultados em DataFrame

## Conceitos aplicados

| Conceito | Implementacao |
|---|---|
| Zero-Shot Classification | Classificacao sem treinamento previo nas categorias |
| NLI (Natural Language Inference) | Modelo infere relacao entre texto e cada label |
| Pipeline Hugging Face | Abstrai tokenizacao, inferencia e pos-processamento |
| Analise de ambiguidade | Deteccao de transacoes com scores proximos entre categorias |

## Por que mDeBERTa e nao mBERT?

O modelo `vidhur2k/mBERT-Portuguese-Mono` foi treinado para tarefas gerais de NLP em portugues, mas nao especificamente para NLI. A pipeline `zero-shot-classification` precisa de um modelo NLI para funcionar corretamente. O `mDeBERTa-v3-base-mnli-xnli` foi treinado em MNLI + XNLI, cobrindo portugues com alta precisao.

## Licenca

MIT
