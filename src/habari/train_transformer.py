"""Fine-tune AfriBERTa for Swahili news classification.

AfriBERTa was pretrained on 11 African languages including Swahili, which
makes it a better starting point than multilingual models dominated by
high-resource languages. The training loop is written in plain PyTorch on
purpose: epoch-level checkpoint selection happens on the validation split,
and the test split is only touched once, through the shared evaluation
harness in evaluate.py.
"""

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from habari import data, evaluate

MODEL_NAME = "castorini/afriberta_base"
MAX_LENGTH = 256
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class NewsDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.encodings = tokenizer(
            df["full_text"].tolist(),
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )
        self.labels = torch.tensor(df["label"].to_numpy())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": self.encodings["input_ids"][i],
            "attention_mask": self.encodings["attention_mask"][i],
            "labels": self.labels[i],
        }


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_logits = []
    for batch in loader:
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        ).logits
        all_logits.append(logits.float().cpu())
    logits = torch.cat(all_logits)
    probabilities = torch.softmax(logits, dim=-1).numpy()
    return probabilities.argmax(axis=1), probabilities


def main(epochs=5, batch_size=16, lr=2e-5):
    set_seed()
    device = pick_device()
    print(f"device: {device}")

    labels = data.label_names()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(labels)
    ).to(device)

    train_df, val_df, test_df = data.load("train"), data.load("validation"), data.load("test")
    train_loader = DataLoader(NewsDataset(train_df, tokenizer), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(NewsDataset(val_df, tokenizer), batch_size=batch_size * 2)
    test_loader = DataLoader(NewsDataset(test_df, tokenizer), batch_size=batch_size * 2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    from sklearn.metrics import f1_score

    best_f1, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += out.loss.item()

        val_pred, _ = predict(model, val_loader, device)
        val_f1 = f1_score(val_df["label"], val_pred, average="macro")
        print(f"epoch {epoch}: train loss {running_loss / len(train_loader):.4f}, val macro F1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"best validation macro F1: {best_f1:.4f}, evaluating that checkpoint on test")
    model.load_state_dict(best_state)
    model.to(device)
    test_pred, test_prob = predict(model, test_loader, device)
    metrics = evaluate.evaluate(
        "afriberta",
        test_df["label"].to_numpy(),
        test_pred,
        labels,
        texts=test_df["full_text"].tolist(),
        probabilities=test_prob,
    )
    evaluate.print_report(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    main(args.epochs, args.batch_size, args.lr)
