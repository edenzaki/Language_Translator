# mt.py: data, vocabularies, the training loop and BLEU for Assignment 4. You may edit it.
import csv
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from model import PAD, Vocab, tokenize

csv.field_size_limit(1 << 24)


def read_tsv(path):
    """[(english, spanish), ...] from a tab-separated file (extra columns are ignored)."""
    with open(path, encoding="utf-8", newline="") as f:
        return [(r[0], r[1]) for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE) if len(r) >= 2]


def load_pairs(path, n_pairs=None, max_len=None, seed=0):
    """Tokenised pairs: shuffled with `seed`, both sides at most max_len tokens, the first n_pairs."""
    pairs = [(tokenize(e), tokenize(s)) for e, s in read_tsv(path)]
    if max_len:
        pairs = [(e, s) for e, s in pairs if len(e) <= max_len and len(s) <= max_len]
    random.Random(seed).shuffle(pairs)
    return pairs[:n_pairs] if n_pairs else pairs


def build_vocabs(pairs, min_freq=1):
    return Vocab.build((e for e, _ in pairs), min_freq), Vocab.build((s for _, s in pairs), min_freq)


def write_vocab(path, en, es, cfg):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"en": en.itos, "es": es.itos, "cfg": cfg}, f, ensure_ascii=False)


def batches(pairs, en, es, batch_size, seed):
    """Shuffled minibatches of (src, tgt_in, tgt_out), padded with 0."""
    order = list(range(len(pairs)))
    random.Random(seed).shuffle(order)
    for b in range(0, len(order), batch_size):
        chunk = [pairs[i] for i in order[b:b + batch_size]]
        src = pad_sequence([torch.tensor(en.encode(e)) for e, _ in chunk], batch_first=True)
        tgt = pad_sequence([torch.tensor(es.encode(s)) for _, s in chunk], batch_first=True)
        yield src, tgt[:, :-1], tgt[:, 1:]


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def train(model, pairs, epochs, ckpt=None, batch_size=64, lr=1e-3, label_smoothing=0.0,
          warmup=0, log=print):
    """Teacher forcing, cross-entropy that skips <pad>. Resumes from `ckpt` if it exists and
    saves there after every epoch, so a Colab disconnect costs at most one epoch. Returns
    the list of per-epoch mean losses."""
    device = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=label_smoothing)
    start, losses, step = 1, [], 0
    if ckpt and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"])
        start, losses, step = state["epoch"] + 1, state["losses"], state.get("step", 0)
        log(f"resumed from {ckpt}: {state['epoch']} epochs done")
    steps_per_epoch = math.ceil(len(pairs) / batch_size)
    for epoch in range(start, epochs + 1):
        model.train(); t0 = time.time(); total = 0.0; count = 0
        for src, tgt_in, tgt_out in batches(pairs, model.en, model.es, batch_size, seed=epoch):
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            if warmup:                                        # linear warm-up, then 1/sqrt(step) decay
                step += 1
                for g in opt.param_groups:
                    g["lr"] = lr * min(step / warmup, math.sqrt(warmup / step))
            logits = model(src, tgt_in)
            loss = loss_fn(logits.flatten(0, 1), tgt_out.flatten())
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(src); count += len(src)
        losses.append(total / count)
        log(f"epoch {epoch:3d}/{epochs}  loss {losses[-1]:.4f}  ({time.time() - t0:.0f} s, "
            f"{steps_per_epoch} steps)")
        if ckpt:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                        "losses": losses, "step": step}, ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)
    return losses


def bleu(hypotheses, references):
    """Corpus BLEU as the grader computes it: sacreBLEU, international tokenisation (so that
    '¿' and '¡' count as tokens of their own), case-insensitive, one reference per sentence.
    0 to 100, higher is better."""
    import sacrebleu
    return sacrebleu.corpus_bleu(hypotheses, [references], lowercase=True, tokenize="intl", force=True).score


def evaluate(model, dev_pairs, batch_size=64):
    """BLEU of model.translate on raw (english, spanish) pairs."""
    hyps = model.translate([e for e, _ in dev_pairs], batch_size=batch_size)
    return bleu(hyps, [s for _, s in dev_pairs]), hyps
