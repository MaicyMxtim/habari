"""LoRA fine-tune of Qwen2.5-1.5B-Instruct for Swahili news classification.

A generative counterpart to the AfriBERTa encoder in train_transformer.py:
the model is asked to emit the category name, and adapters are trained on
that objective. Runs on Apple Silicon (MPS) in bfloat16, so no bitsandbytes
and no 4-bit quantisation, neither of which support Metal.

Written to be directly comparable with the other models: same splits, same
label set, and the shared evaluation harness, so the test split is scored
once at the end.
"""

import argparse
import random
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from habari import data, evaluate

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LENGTH = 512
SEED = 42

INSTRUCTION = (
    "Classify this Swahili news article into exactly one category from this list: "
    "{labels}. Reply with the category only.\n\n"
)


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


def prompt_for(row, labels):
    text = (row["headline"] + "\n" + row["text"]).strip()[:1500] if "headline" in row else row["full_text"][:1500]
    return INSTRUCTION.format(labels=", ".join(labels)) + text


class ChatDataset(Dataset):
    """Prompt tokens are masked out; loss applies to the answer only."""

    def __init__(self, df, tokenizer, labels):
        self.examples = []
        for _, row in df.iterrows():
            messages = [{"role": "user", "content": prompt_for(row, labels)}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            answer = row["category"] + tokenizer.eos_token
            p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            a_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
            # keep the answer: truncate the prompt from the left if needed
            budget = MAX_LENGTH - len(a_ids)
            p_ids = p_ids[-budget:]
            ids = p_ids + a_ids
            labels_ids = [-100] * len(p_ids) + a_ids
            self.examples.append((ids, labels_ids))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def collate(batch, pad_id):
    width = max(len(ids) for ids, _ in batch)
    input_ids, labels, attention = [], [], []
    for ids, lab in batch:
        pad = width - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        attention.append([1] * len(ids) + [0] * pad)
    return (
        torch.tensor(input_ids),
        torch.tensor(labels),
        torch.tensor(attention),
    )


@torch.no_grad()
def classify(model, tokenizer, df, labels, device):
    model.eval()
    preds = []
    for _, row in df.iterrows():
        messages = [{"role": "user", "content": prompt_for(row, labels)}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        ids = {k: v.to(device) for k, v in ids.items()}
        out = model.generate(**ids, max_new_tokens=6, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        answer = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
        preds.append(next((i for i, l in enumerate(labels) if l in answer), 0))
    return np.array(preds)


def main(epochs=1, batch_size=2, accum=8, lr=2e-4):
    set_seed()
    device = pick_device()
    labels = data.label_names()
    train_df, test_df = data.load("train"), data.load("test")
    print(f"device {device} | {len(train_df)} train / {len(test_df)} test | labels {labels}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to(device)
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    ds = ChatDataset(train_df, tokenizer, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    steps = (len(loader) // accum) * epochs
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    print(f"{steps} optimizer steps ({len(loader)} batches x {epochs} epochs / accum {accum})")

    model.train()
    start = time.time()
    done = 0
    for epoch in range(epochs):
        running = 0.0
        for i, (input_ids, lab, attn) in enumerate(loader):
            input_ids, lab, attn = input_ids.to(device), lab.to(device), attn.to(device)
            loss = model(input_ids=input_ids, attention_mask=attn, labels=lab).loss
            (loss / accum).backward()
            running += loss.item()
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                done += 1
                if done % 10 == 0:
                    rate = (time.time() - start) / done
                    print(f"step {done}/{steps} | loss {running / (i + 1):.4f} "
                          f"| {rate:.1f}s/step | eta {(steps - done) * rate / 60:.0f}m",
                          flush=True)
        print(f"epoch {epoch + 1}: train loss {running / len(loader):.4f}", flush=True)

    print(f"training took {(time.time() - start) / 60:.1f} minutes", flush=True)

    y_true = np.array([labels.index(c) for c in test_df["category"]])
    preds = classify(model, tokenizer, test_df, labels, device)
    metrics = evaluate.evaluate("qwen25_1.5b_lora", y_true, preds, labels)
    evaluate.print_report(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()
    main(args.epochs, args.batch_size, args.accum, args.lr)
