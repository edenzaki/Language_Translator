# model.py: your English -> Spanish translator for CSCI 6379 Assignment 4.
#
# WHAT I BASED THIS DESIGN ON (required, keep this up to date):
#   The Topic 32 Transformer from the course (one attention class, encoder and decoder
#   layers, greedy decoding), unchanged except for dropout.
#   (Replace this with your own words: the papers, tutorials or course notes you used,
#    and what you changed. One to five lines is enough.)
#
# The contract with the grader (the only things that must stay true):
#   * This file defines class Translator(torch.nn.Module). The grader runs it in a folder
#     holding exactly model.py, vocab.json and weights.pt, builds it as Translator() with no
#     arguments (so everything it needs comes from vocab.json), loads weights.pt into it with
#     strict=True, moves it to the GPU, and calls
#         translator.translate(sentences)     # a list of English strings
#     which must return a list of Spanish strings, one per input, in the same order.
#   * At most 30,000,000 parameters, trained from scratch on the provided data only.
#   * Only torch, numpy and the Python standard library are available in the grader
#     (plus the `tokenizers` and `sentencepiece` packages, if you want subword units).
#     Do not import other files of your own.
#
# Everything else is yours: the tokenizer, the vocabularies, the network, the decoding.
import json
import math
import re
import unicodedata

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

PAD, SOS, EOS, UNK = 0, 1, 2, 3      # the four special token ids, as in Topic 32


# ---------------------------------------------------------------- tokens and vocabularies
_TOKEN = re.compile(r"\w+|[^\w\s]")  # words, and each punctuation mark on its own


def tokenize(text):
    """'Where's Tom?' -> ['where', "'", 's', 'tom', '?']   (lower-cased; accents are kept)"""
    return _TOKEN.findall(unicodedata.normalize("NFC", text).lower())


def detokenize(tokens):
    """['¿', 'dónde', 'está', 'tom', '?'] -> '¿dónde está tom?'"""
    out = ""
    for t in tokens:
        if out and t not in ".,;:!?)" and not out.endswith(("¿", "¡", "(")):
            out += " "
        out += t
    return out


class Vocab:
    """itos: the list of tokens, index = id. The first four are <pad>, <sos>, <eos>, <unk>."""

    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def build(cls, token_lists, min_freq=1):
        counts = {}
        for toks in token_lists:
            for w in toks:
                counts[w] = counts.get(w, 0) + 1
        words = sorted(w for w, c in counts.items() if c >= min_freq)
        return cls(["<pad>", "<sos>", "<eos>", "<unk>"] + words)

    def encode(self, tokens):                     # tokens -> [<sos>, ids..., <eos>]
        return [SOS] + [self.stoi.get(w, UNK) for w in tokens] + [EOS]

    def decode(self, ids):                        # ids -> tokens, stopping at <eos>
        out = []
        for i in ids:
            if i == EOS:
                break
            if i not in (PAD, SOS):
                out.append(self.itos[i])
        return out

    def __len__(self):
        return len(self.itos)


# ---------------------------------------------------------------- the Transformer (Topic 32)
class MultiHeadAttention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.dk = h, d // h
        self.q, self.k = nn.Linear(d, d), nn.Linear(d, d)
        self.v, self.o = nn.Linear(d, d), nn.Linear(d, d)

    def forward(self, x_q, x_kv, mask):           # mask: True = may look
        B, Tq, d = x_q.shape

        def split(t):                             # (B,T,d) -> (B,h,T,dk)
            return t.view(B, -1, self.h, self.dk).transpose(1, 2)
        Q = split(self.q(x_q))
        K, V = split(self.k(x_kv)), split(self.v(x_kv))
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.dk)
        scores = scores.masked_fill(~mask, float("-inf"))
        out = torch.softmax(scores, dim=-1) @ V
        out = out.transpose(1, 2).reshape(B, Tq, d)
        return self.o(out)


def pad_mask(src):                                # (B, 1, 1, Ts): nobody looks at <pad>
    return (src != PAD)[:, None, None, :]


def causal_mask(tgt):                             # (B, 1, Tt, Tt): position t sees <= t, not <pad>
    T = tgt.size(1)
    past = torch.ones(T, T, dtype=torch.bool, device=tgt.device).tril()
    return past & (tgt != PAD)[:, None, None, :]


def feed_forward(d, d_ff, dropout):
    return nn.Sequential(nn.Linear(d, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d))


class EncoderLayer(nn.Module):
    def __init__(self, d, h, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d, h)
        self.ff = feed_forward(d, d_ff, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x = self.norm1(x + self.drop(self.attn(x, x, src_mask)))
        return self.norm2(x + self.drop(self.ff(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d, h, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d, h)
        self.cross = MultiHeadAttention(d, h)
        self.ff = feed_forward(d, d_ff, dropout)
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(d) for _ in range(3))
        self.drop = nn.Dropout(dropout)

    def forward(self, y, memory, tgt_mask, src_mask):
        y = self.norm1(y + self.drop(self.self_attn(y, y, tgt_mask)))
        y = self.norm2(y + self.drop(self.cross(y, memory, src_mask)))
        return self.norm3(y + self.drop(self.ff(y)))


class Transformer(nn.Module):
    """max_len: the most tokens a sentence may have (without <sos>/<eos>)."""

    def __init__(self, src_vocab, tgt_vocab, d=128, h=4, n_layers=1, d_ff=256, max_len=10, dropout=0.1):
        super().__init__()
        self.d, self.max_len = d, max_len
        self.src_emb = nn.Embedding(src_vocab, d)
        self.tgt_emb = nn.Embedding(tgt_vocab, d)
        self.pos = nn.Embedding(max_len + 2, d)   # + <sos> and <eos>
        self.drop = nn.Dropout(dropout)
        self.enc = nn.ModuleList(EncoderLayer(d, h, d_ff, dropout) for _ in range(n_layers))
        self.dec = nn.ModuleList(DecoderLayer(d, h, d_ff, dropout) for _ in range(n_layers))
        self.out = nn.Linear(d, tgt_vocab)

    def embed(self, emb, ids):
        pos = torch.arange(ids.size(1), device=ids.device)
        return self.drop(emb(ids) + self.pos(pos))

    def encode(self, src):
        x = self.embed(self.src_emb, src)
        for layer in self.enc:
            x = layer(x, pad_mask(src))
        return x                                  # memory

    def decode(self, tgt, memory, src):
        y = self.embed(self.tgt_emb, tgt)
        for layer in self.dec:
            y = layer(y, memory, causal_mask(tgt), pad_mask(src))
        return self.out(y)                        # (B, Tt, tgt_vocab) logits

    def forward(self, src, tgt_in):
        return self.decode(tgt_in, self.encode(src), src)


# ---------------------------------------------------------------- what the grader uses
class Translator(nn.Module):
    """Built from vocab.json alone: {"en": [...], "es": [...], "cfg": {...}}."""

    def __init__(self, vocab_file="vocab.json"):
        super().__init__()
        with open(vocab_file, encoding="utf-8") as f:
            v = json.load(f)
        self.en, self.es, self.cfg = Vocab(v["en"]), Vocab(v["es"]), dict(v["cfg"])
        self.net = Transformer(len(self.en), len(self.es), **self.cfg)

    def forward(self, src, tgt_in):               # training: logits for the whole target at once
        return self.net(src, tgt_in)

    def encode_src(self, sentences, device):
        L = self.cfg["max_len"]
        ids = [torch.tensor(self.en.encode(tokenize(s)[:L])) for s in sentences]
        return pad_sequence(ids, batch_first=True).to(device)

    @torch.no_grad()
    def translate(self, sentences, batch_size=64):
        """English strings -> Spanish strings, greedy, one word at a time (Topic 32, batched)."""
        self.eval()
        device = next(self.parameters()).device
        L = self.cfg["max_len"]
        out = []
        for b in range(0, len(sentences), batch_size):
            src = self.encode_src(sentences[b:b + batch_size], device)
            memory = self.net.encode(src)
            ys = torch.full((src.size(0), 1), SOS, dtype=torch.long, device=device)
            done = torch.zeros(src.size(0), dtype=torch.bool, device=device)
            for _ in range(L + 1):
                nxt = self.net.decode(ys, memory, src)[:, -1].argmax(-1)
                nxt[done] = PAD
                ys = torch.cat([ys, nxt[:, None]], 1)
                done |= nxt == EOS
                if bool(done.all()):
                    break
            out += [detokenize(self.es.decode(row.tolist())) for row in ys]
        return out
