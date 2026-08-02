# Habari — Swahili news classification with modern and classical NLP

Habari (Swahili for "news") is an end-to-end NLP project on low-resource language text classification. It takes the Swahili portion of the [MasakhaNEWS](https://huggingface.co/datasets/masakhane/masakhanews) dataset and asks a simple question with a rigorous answer: how much does a fine-tuned language model actually buy you over strong classical baselines on a low-resource African language, and how confident can we be in the difference?

Most NLP portfolio work reports a single accuracy number on a high-resource English dataset. This project deliberately does neither. Every result below is reported with bootstrap confidence intervals, and every model is compared against a properly tuned classical baseline before any deep learning is used.

## Results

| Model | Accuracy | Macro F1 | 95% CI (F1) |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.834 | 0.806 | 0.761–0.846 |
| Fine-tuned AfriBERTa | _pending_ | _pending_ | _pending_ |
| QLoRA fine-tuned small LLM | _pending_ | _pending_ | _pending_ |

## Why Swahili

Swahili is spoken by over 100 million people and remains underserved by NLP tooling. Models that look solved on English degrade sharply on low-resource languages, which makes this a more honest test of methods than another English benchmark. The [Masakhane](https://www.masakhane.io/) community maintains the datasets this project builds on.

## Project structure

```
habari/
  src/habari/
    data.py        # dataset download and preparation
    evaluate.py    # metrics with bootstrap confidence intervals
    baseline.py    # TF-IDF + logistic regression baseline
  data/            # cached dataset (not committed)
  results/         # metrics, confusion matrices, error analysis
```

## Reproducing

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m habari.data
python -m habari.baseline
```

## Roadmap

1. Classical baseline with full evaluation harness (done)
2. Fine-tuned masked language model (AfriBERTa)
3. QLoRA fine-tune of a small open LLM, compared under the same harness
4. Deployment on a self-built Kubernetes MLOps platform with monitoring
