"""
Train constraint network with BERT token-level encoder.
Same pipeline as train_nl_fast.py but using BERTWindowEncoder.

Changes from MiniLM version:
- encoder_dim: 384 → 768 (BERT base hidden dim)
- d_model: 256 → 384 (scale up to match richer input)
- Window pooling happens in the encoder, not the constraint network

Run: python train_bert.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import random
import math
import os
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler

from model import ConstraintNetwork, ConstraintLoss
from bert_encoder import BERTWindowEncoder, precompute_bert_embeddings
from nl_adaptation import SentenceEncoderWrapper  # for split_sentences


CONFIG = dict(
    # Data
    num_paragraphs=50000,
    min_sentences=4,
    max_sentences=32,     # doubled — more windows with smaller window_size
    corruptions_per_paragraph=5,

    # Encoder
    bert_model="bert-base-uncased",
    encoder_dim=768,
    window_size=8,        # was 16 — finer granularity preserves more token detail

    # Model — larger to match richer input
    d_model=384,
    d_state=64,
    dropout=0.15,

    # Training
    batch_size=256,
    lr=1e-4,             # was 3e-4 — model peaked at epoch 1 when LR was 1e-4
    weight_decay=0.01,
    epochs=40,
    margin=5.0,
    grad_clip=1.0,
    use_fp16=True,
    patience=10,          # more patience at lower LR

    # Hard negatives
    hard_neg_start_epoch=10,  # earlier now that base LR is stable
    hard_neg_fraction=0.3,

    # LR schedule
    warmup_epochs=2,      # quick warmup since max LR is already low
)


def load_paragraphs(num, min_sents, max_sents):
    from datasets import load_dataset
    print(f"Loading {num} Wikipedia paragraphs...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    paragraphs = []
    for item in ds:
        text = item["text"].strip()
        if len(text) < 100:
            continue
        sents = SentenceEncoderWrapper.split_sentences(text)
        if len(sents) >= min_sents:
            paragraphs.append(sents[:max_sents])
            if len(paragraphs) >= num:
                break
    print(f"  Loaded {len(paragraphs)} paragraphs")
    return paragraphs


def load_val_paragraphs(num=2000, min_sents=4, max_sents=16):
    from datasets import load_dataset
    print(f"Loading {num} validation paragraphs...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    paragraphs = []
    for item in ds:
        text = item["text"].strip()
        if len(text) < 100:
            continue
        sents = SentenceEncoderWrapper.split_sentences(text)
        if len(sents) >= min_sents:
            paragraphs.append(sents[:max_sents])
            if len(paragraphs) >= num:
                break
    print(f"  Loaded {len(paragraphs)} validation paragraphs")
    return paragraphs


def train():
    config = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"BERT encoder: {config['bert_model']}")
    print(f"Config: {config['num_paragraphs']} paragraphs, d_model={config['d_model']}, "
          f"window={config['window_size']}, batch={config['batch_size']}, "
          f"lr={config['lr']}, epochs={config['epochs']}\n")

    # Encoder
    ws = config["window_size"]
    encoder = BERTWindowEncoder(
        config["bert_model"], str(device),
        layer=-2, window_size=ws, max_tokens=512)

    # Data
    paragraphs = load_paragraphs(
        config["num_paragraphs"], config["min_sentences"], config["max_sentences"])
    val_paragraphs = load_val_paragraphs(2000, config["min_sentences"], config["max_sentences"])

    # Pre-compute (cache name includes window size)
    print("\nPre-computing training embeddings (BERT)...")
    pos_embs, neg_embs, neg_types = precompute_bert_embeddings(
        paragraphs, encoder, config["max_sentences"], device,
        config["corruptions_per_paragraph"],
        cache_path=f"train_bert_w{ws}_cache.pt")

    print("\nPre-computing validation embeddings (BERT)...")
    val_pos, val_neg, val_types = precompute_bert_embeddings(
        val_paragraphs, encoder, config["max_sentences"], device,
        corruptions_per_para=6,
        cache_path=None)

    N = pos_embs.shape[0]
    K = config["corruptions_per_paragraph"]

    # Model
    model = ConstraintNetwork(
        d_model=config["d_model"],
        d_state=config["d_state"],
        vocab_size=None,
        max_seq_len=config["max_sentences"],
        dropout=config["dropout"],
        alpha=0.3,
    ).to(device)
    model.input_proj = nn.Linear(config["encoder_dim"], config["d_model"]).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"\nModel params: {params:,}")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    total_steps = config["epochs"] * (N // config["batch_size"])
    warmup_steps = config["warmup_epochs"] * (N // config["batch_size"])

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = ConstraintLoss(config["margin"])
    scaler = GradScaler() if config["use_fp16"] else None

    # Training
    history = []
    best_acc = 0
    no_improve = 0
    step = 0

    print(f"\nTraining: {N} paragraphs x {K} corruptions = {N*K} pairs")
    print(f"  {N // config['batch_size']} batches/epoch")
    print("=" * 70)

    for epoch in range(config["epochs"]):
        t0 = time.time()
        model.train()

        perm = torch.randperm(N)
        neg_choices = torch.randint(0, K, (N,))
        tl = 0; n = 0
        bs = config["batch_size"]

        for b in range(0, N - bs, bs):
            idx = perm[b:b+bs]
            neg_idx = idx * K + neg_choices[idx]
            pos_batch = pos_embs[idx].to(device)
            neg_batch = neg_embs[neg_idx].to(device)

            # Hard negative mining
            if (epoch >= config["hard_neg_start_epoch"] and
                random.random() < config["hard_neg_fraction"]):
                with torch.no_grad():
                    all_neg_e = []
                    for k in range(K):
                        nk = neg_embs[idx * K + k].to(device)
                        if config["use_fp16"]:
                            with autocast(device_type="cuda"):
                                all_neg_e.append(model(nk))
                        else:
                            all_neg_e.append(model(nk))
                    all_neg_e = torch.stack(all_neg_e, dim=1)
                    hardest = all_neg_e.argmin(dim=1)
                    hard_idx = idx * K + hardest.cpu()
                    neg_batch = neg_embs[hard_idx].to(device)

            if config["use_fp16"]:
                with autocast(device_type="cuda"):
                    ep = model(pos_batch)
                    en = model(neg_batch)
                    loss = criterion(ep, en)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                ep = model(pos_batch)
                en = model(neg_batch)
                loss = criterion(ep, en)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()
            tl += loss.item(); n += 1
            step += 1

        # Validate
        model.eval()
        correct = 0; total = 0; vl = 0; vn = 0
        type_correct = {}; type_total = {}
        val_K = 6

        with torch.no_grad():
            val_pos_exp = val_pos.repeat_interleave(val_K, dim=0)
            vbs = config["batch_size"]
            for b in range(0, len(val_pos_exp) - vbs, vbs):
                vp = val_pos_exp[b:b+vbs].to(device)
                vneg = val_neg[b:b+vbs].to(device)

                if config["use_fp16"]:
                    with autocast(device_type="cuda"):
                        ep_v = model(vp)
                        en_v = model(vneg)
                        loss_v = criterion(ep_v, en_v)
                else:
                    ep_v = model(vp)
                    en_v = model(vneg)
                    loss_v = criterion(ep_v, en_v)

                vl += loss_v.item(); vn += 1
                preds = (ep_v < en_v)
                correct += preds.sum().item()
                total += preds.shape[0]

                for j in range(preds.shape[0]):
                    ct = val_types[b + j]
                    type_correct[ct] = type_correct.get(ct, 0) + (1 if preds[j] else 0)
                    type_total[ct] = type_total.get(ct, 0) + 1

        acc = correct / max(total, 1)
        dt = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        tag = ""
        if acc > best_acc:
            best_acc = acc
            no_improve = 0
            torch.save(model.state_dict(), "nl_bert_constraint_best.pt")
            tag = " *"
        else:
            no_improve += 1

        rec = {"epoch": epoch+1, "train_loss": tl/n, "val_loss": vl/max(vn,1),
               "val_acc": acc, "lr": lr, "time": dt,
               "per_type": {ct: type_correct.get(ct,0)/max(type_total.get(ct,1),1)
                            for ct in type_total}}
        history.append(rec)

        type_str = " | ".join(
            f"{ct[:4]}={type_correct.get(ct,0)/max(type_total.get(ct,1),1):.0%}"
            for ct in sorted(type_total.keys()))

        if (epoch+1) % 1 == 0 or tag:
            print(f"E{epoch+1:3d} | loss={tl/n:.4f} val={vl/max(vn,1):.4f} "
                  f"acc={acc:.3f} lr={lr:.1e} | {type_str} | {dt:.0f}s{tag}")

        if no_improve >= config.get("patience", 8):
            print(f"  Early stopping (patience={config['patience']})")
            break

    print(f"\nBest accuracy: {best_acc:.3f}")

    # Final breakdown
    print(f"\n{'='*70}")
    print("Final per-corruption accuracy:")
    print(f"{'='*70}")
    for ct in sorted(type_total.keys()):
        ct_acc = type_correct.get(ct, 0) / max(type_total.get(ct, 1), 1)
        print(f"  {ct:<20s}: {ct_acc:.1%} ({type_total[ct]} samples)")

    results = {
        "config": {k: v for k, v in config.items()},
        "history": history,
        "best_acc": best_acc,
        "per_type": {ct: type_correct.get(ct,0)/max(type_total.get(ct,1),1)
                     for ct in type_total},
        "params": params,
    }
    with open("bert_constraint_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to bert_constraint_results.json")


if __name__ == "__main__":
    train()
