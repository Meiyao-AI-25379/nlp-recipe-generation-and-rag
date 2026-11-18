# NLP Recipe Generation & RAG-Based Intelligent Menu Designer

**Python · PyTorch · Hugging Face · LoRA · BM25 · Groq API · LLM-as-Judge**

**Individual academic project | FIT5217 Natural Language Processing | Monash University | 2026**  
**Author:** Meiyao Zhou

## Overview

This project investigates recipe generation through three approaches: recurrent sequence-to-sequence models trained from scratch, parameter-efficient fine-tuning of pre-trained transformers, and a retrieval-augmented menu-generation application. Given ingredients, the trained models generate cooking instructions. The RAG application takes natural-language dining requests and generates three-course menus grounded in retrieved recipes, then evaluates their quality using two LLM judges.

The work was implemented in **Google Colab**, with experiments documented in a Jupyter notebook. It is an academic prototype, not a deployed production service.

### Highlights

- Built baseline **RNN encoder–decoder** and **Bahdanau attention** models in PyTorch.
- Fine-tuned **T5-small** and **GPT-2** using **Low-Rank Adaptation (LoRA)**, and compared generation quality using BLEU-4, METEOR, and BERTScore.
- Indexed **165,045 recipe documents** using **BM25Okapi**, with top-5 retrieval for a RAG menu-generation pipeline.
- Used **Groq-hosted Llama models** for menu generation and independent, rubric-based evaluation of constraints, faithfulness, culinary logic, and bias.
- Analysed a concrete evaluation failure: for one Italian dinner menu, self- and cross-judges scored ingredient faithfulness **2/5 vs 4/5**.

## Tech Stack

| Area | Tools / Methods |
| --- | --- |
| Language and environment | Python, Google Colab |
| Neural networks | PyTorch, encoder–decoder RNN, Bahdanau attention |
| Transformer fine-tuning | Hugging Face Transformers, PEFT/LoRA, T5-small, GPT-2 |
| Text processing | pandas, NLTK tokenisation |
| Retrieval | `rank_bm25` (`BM25Okapi`), keyword-based sparse retrieval |
| Generation | Groq Python SDK; `llama-3.3-70b-versatile` |
| LLM judging | `llama-3.3-70b-versatile` (self), `llama-3.1-8b-instant` (cross) |
| Evaluation | BLEU-4, METEOR, BERTScore; four-dimension rubric (1–5) |

## Data

The coursework uses a cooking dataset with a recipe title, an ingredient list, and ordered cooking steps. Excluding CSV headers, the notebook loads **162,899 training**, **1,065 validation**, and **1,081 test** examples. Supervised model development keeps the test split separate; Task 3 merges all three splits to build the retrieval index, as permitted by the assignment specification.

For Task 1, NLTK tokenisation and a minimum-frequency threshold of **5** yielded a vocabulary of **7,390** tokens and a validation out-of-vocabulary rate of **0.0019** in the notebook output.

> Dataset files are not automatically included in this repository. Reproduction requires access to the source dataset under its applicable terms.

## Task 1 — Sequence-to-Sequence Recipe Generation from Scratch

Two PyTorch encoder–decoder models were implemented:

1. **Baseline RNN:** encode ingredient tokens into hidden representations and decode cooking instructions using teacher forcing.
2. **RNN with Bahdanau attention:** calculate additive attention over encoder hidden states at decoding time to make generation more sensitive to relevant input tokens.

Both use **128-dimensional embeddings**, a **256-dimensional hidden state**, a single recurrent layer, gradient clipping, and patience-based early-stopping logic. The baseline used a batch size of **128** and learning rate **0.001**; the attention model used a batch size of **16** and learning rate **0.0005**. Teacher forcing was gradually reduced during training.

The baseline ran for **15 epochs**; attention training ended after **6 epochs due to GPU memory limitations**, before clear convergence. These are experimental constraints, not evidence that the attention model fully converged.

## Task 2 — LoRA Fine-tuning of T5 and GPT-2

I adapted two different transformer architectures for conditional recipe generation:

- **T5-small (encoder–decoder):** compared three LoRA configurations (`r=8, α=16`; `r=8, α=32`; `r=16, α=32`) targeting `q` and `v` attention modules. The reported final test results use `r=16, α=32`, dropout `0.1`.
- **GPT-2 (decoder-only):** trained on sequences combining ingredients and recipe instructions, with LoRA targeting `c_attn` (`r=8`, `α=16`, dropout `0.1`). Evaluated **greedy** and **beam** decoding.

**Parameter efficiency:** the selected T5 LoRA configuration trains **589,824 / 61,096,448 parameters (0.9654%)**; GPT-2 LoRA trains **294,912 / 124,734,720 (0.2364%)**.

### Recipe Generation Results

Results below are from the submitted notebook and report (test split):

| Model | BLEU-4 | METEOR | BERTScore |
| --- | ---: | ---: | ---: |
| Baseline RNN | 0.0107 | 0.0997 | 0.7484 |
| RNN + Bahdanau attention | 0.0355 | 0.1623 | 0.8032 |
| **T5-small + LoRA** | **0.0809** | 0.2408 | **0.8811** |
| GPT-2 + LoRA (greedy) | 0.0446 | 0.2074 | 0.8342 |
| GPT-2 + LoRA (beam) | 0.0567 | **0.2461** | 0.8506 |

T5-small achieved the highest BLEU-4 and BERTScore, while GPT-2 with beam decoding achieved the highest METEOR. The attention RNN improved over the baseline, but its generated recipes could still be brief or insufficiently grounded. Scores describe performance on this dataset and metric setup; they are not general-purpose model benchmarks.

## Task 3 — RAG-Based Intelligent Menu Designer

The Menu Designer accepts a natural-language request such as an Italian dinner or a low-fat vegetarian dinner excluding cheese and cream.

### Architecture

```text
User request
     │
     ▼
Query preprocessing (lowercasing, tokenisation, lightweight stopword removal)
     │
     ▼
BM25Okapi index of 165,045 recipes
     │
     ▼
Top-5 relevant recipes
     │
     ▼
Retrieval-grounded prompt + few-shot example + dietary constraints
     │
     ▼
Groq API → Llama 3.3 70B → three-course menu
     │
     ▼
Two rubric-based LLM judges
     │
     ▼
Structured scores and explanations
```

### Retrieval Design

- **Index:** sparse **BM25Okapi**, implemented with `rank_bm25`.
- **Documents:** 165,045 recipes, average document length **36.08 whitespace-separated tokens**.
- **Top-k:** 5 retrieved recipes; no additional neural reranker.
- **Observed index construction time:** **2.3059 seconds** in the Colab run.
- **Observed sample retrieval latency:** approximately **0.10 seconds** for a sample query (environment-specific).

BM25 was selected for lexical overlap with recipe titles, ingredients, cuisine styles, and explicit dietary terms. A potential improvement is adding dense retrieval or a hybrid retriever for paraphrases and semantic matching.

### Generation

The notebook uses the **Groq API** with `llama-3.3-70b-versatile`. The prompt combines retrieved recipes, role-based instructions, a few-shot example, structured starter/main/dessert formatting, and explicit dietary/exclusion constraints. Each dish names a source recipe and explains how it addresses the user's request.

### LLM-as-Judge

The generated menus are evaluated on four **1–5** scales:

| Criterion | What the judge checks |
| --- | --- |
| Constraint satisfaction | Dietary, cultural and exclusion requirements |
| Ingredient faithfulness | Grounding in retrieved ingredients and recipes |
| Culinary logic and coherence | Plausibility and ordering of recipe steps |
| Bias | Unjustified cultural or dietary assumptions |

Two models assess each menu:

- **Self-judge:** `llama-3.3-70b-versatile` (same as the generator)
- **Cross-judge:** `llama-3.1-8b-instant` (different model)

#### Recorded Evaluation Examples

| Scenario | Judge | Constraints | Faithfulness | Logic | Bias |
| --- | --- | ---: | ---: | ---: | ---: |
| Soup / chicken / cake | Self | 5 | 4 | 4 | 5 |
| Soup / chicken / cake | Cross | 5 | 4 | 4 | 5 |
| Italian three-course dinner | Self | 5 | **2** | 4 | 5 |
| Italian three-course dinner | Cross | 5 | **4** | 4 | 5 |
| Low-fat vegetarian, no cheese or cream | Self | 5 | 5 | 5 | 5 |
| Low-fat vegetarian, no cheese or cream | Cross | 5 | 4 | 5 | 5 |

**Key finding:** the Italian dinner introduced ingredients that were not clearly supported by the retrieved recipes. The two judges disagreed on ingredient faithfulness (**2 vs 4**), illustrating the limitations of automated evaluation. High judge scores should therefore not be treated as proof of factual correctness; future work could add explicit ingredient checks and human review.

## How to Explore the Notebook

1. Open the `.ipynb` file in **Google Colab**.
2. Inspect the sections for preprocessing, RNN training, attention, LoRA experiments, RAG retrieval, and LLM evaluation.
3. Install the dependencies used by the selected section and provide the cooking dataset, model checkpoints, and evaluation helper where needed.
4. For Menu Designer generation, configure `GROQ_API_KEY` securely at runtime; never commit credentials.
5. Update the notebook's original `/content/drive/MyDrive/FIT5217/...` paths for your own environment before running.

**Reproducibility note:** this is an archived coursework notebook with Colab/Google Drive-specific paths and external model/API dependencies. It is not a one-click deployable app, and availability of historical API model names may change.

## Limitations and Future Work

- **RNN computation:** attention training was interrupted by GPU memory limits; additional training and memory profiling could improve comparisons.
- **Recipe grounding:** retrieved examples did not always support all generated ingredients, especially for cuisine-specific menus.
- **Retrieval:** BM25 is efficient for lexical matches but can miss semantic equivalents; dense/hybrid retrieval and reranking could help.
- **Evaluation:** LLM judges sometimes disagree or score plausible but insufficiently grounded outputs generously. Calibrated rubrics, structured ingredient verification, multiple samples, and human spot-checks would strengthen reliability.
- **Deployment:** a cleaned-up Python package or small web demo could make the prototype easier to reproduce and test outside Colab.

## Academic Context

This work was produced as an **individual academic assignment** for FIT5217 at Monash University. Results are experimental and should not be interpreted as production performance. Any public release of coursework code, provided datasets, or assessment material should follow university policy and relevant licences.
