#!/usr/bin/env python3
"""
minimal_gpt.py -- a single-file, teaching-oriented re-implementation of minGPT's
"gpt-nano", loading the *exact* weights that drive the bbycroft.net/llm 3D
visualization.  Pure PyTorch, CPU, no training code -- just a clean forward pass
you can read top-to-bottom, organised in the same "stations" as the 3D tour.

Model : n_layer=3, n_head=3, n_embd=48, vocab_size=3 (A,B,C), block_size=11  (~85k params)
Task  : sort 6 letters.  Feed "CBABBC", autoregress 6 steps -> "ABBBCC".

Weight-JSON schema (verified against llm-viz src/utils/tensor.ts + src/utils/data.ts):
    "<key>": {"shape":[...], "dtype":"torch.float32", "data":"<base64>"}
    base64 --atob--> raw bytes --new Float32Array()--> LITTLE-ENDIAN float32 ('<f4'),
    reshaped ROW-MAJOR (C-order) into `shape`.  nn.Linear weights are stored [out, in].
    So each tensor == np.frombuffer(b64decode(data), '<f4').reshape(shape).
    lm_head.weight is a SEPARATE key (minGPT does NOT tie it to wte).

Run:
    python minimal_gpt.py                     # decode "CBABBC" + 20-sample accuracy
    python minimal_gpt.py --input CBABBC --trace   # print every station (shapes+tensors)
    python minimal_gpt.py --eval 100          # accuracy over 100 random sequences
"""
import argparse, base64, json, math, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

VOCAB = "ABC"                                   # token id = index into this string
encode = lambda s: [VOCAB.index(c) for c in s]
decode = lambda ids: "".join(VOCAB[i] for i in ids)


# ------------------------------------------------------------------ trace helpers
def _row(t, k=6):
    """Compact one-line view: first k values of the flattened tensor."""
    flat = t.detach().reshape(-1)
    body = ", ".join(f"{v:+.4f}" for v in flat[:k].tolist())
    return f"[{body}{' ...' if flat.numel() > k else ''}]"

def p(name, t, k=6):
    print(f"    {name:<26} shape={tuple(t.shape)!s:<14} {_row(t, k)}")

def pmat(name, m):
    """Full matrix print (used for the 11x11-or-smaller attention maps)."""
    print(f"    {name} shape={tuple(m.shape)}:")
    for r in m.detach().tolist():
        print("      " + " ".join(f"{v:5.2f}" for v in r))


# ------------------------------------------------------------------ NewGELU
def new_gelu(x):
    # minGPT's tanh APPROXIMATION of GELU (deliberately not the exact erf version).
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


# ================================================================== Self Attention
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.n_head = n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)     # one matmul makes Q, K and V
        self.c_proj = nn.Linear(n_embd, n_embd)         # the "Projection" station
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask, persistent=False)   # causal (lower-triangular)

    def forward(self, x, trace=False):
        B, T, C = x.shape
        nh, hd = self.n_head, C // self.n_head          # heads, and dims-per-head

        # --- (1) QKV projection FIRST: one Linear -> a fat (B,T,3C) tensor ----------
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)                   # three (B,T,C) tensors

        # --- (2) THEN split each into heads: (B,T,C) -> (B, nh, T, hd) --------------
        q = q.view(B, T, nh, hd).transpose(1, 2)
        k = k.view(B, T, nh, hd).transpose(1, 2)
        v = v.view(B, T, nh, hd).transpose(1, 2)

        # --- (3) scaled dot-product attention, masked so a token sees only the past -
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)         # (B, nh, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v                                            # (B, nh, T, hd)

        # --- (4) re-assemble the heads back into (B,T,C) ---------------------------
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        if trace:
            print("  === Self Attention ===")
            p("qkv = c_attn(x)", qkv)
            print(f"    split -> Q,K,V each {tuple(q.shape)}   (n_head={nh}, head_dim={hd})")
            for h in range(nh):
                pmat(f"head {h} attention (softmax, rows=query, cols=key)", att[0, h])

        # === Projection === (mix the heads back together)
        y = self.c_proj(y)
        if trace:
            print("  === Projection ===")
            p("y = c_proj(y)", y)
        return y


# ================================================================== Transformer block
class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)                       # pre-LN, before attention
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)                       # pre-LN, before MLP
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(n_embd, 4 * n_embd),                # widen  48 -> 192
            c_proj=nn.Linear(4 * n_embd, n_embd),              # narrow 192 -> 48
        ))

    def forward(self, x, trace=False):
        # === Layer Norm ===  then attention, added back as a residual
        x = x + self.attn(self.ln_1(x), trace=trace)
        # === Layer Norm ===  then MLP, added back as a residual
        h = self.ln_2(x)
        # === MLP ===  Linear -> NewGELU -> Linear
        h = self.mlp.c_proj(new_gelu(self.mlp.c_fc(h)))
        x = x + h
        if trace:
            print("  === MLP === (c_fc -> NewGELU -> c_proj, then + residual)")
            p("block output x", x)
        return x


# ================================================================== the whole model
class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.block_size = cfg["block_size"]
        C, nh = cfg["n_embd"], cfg["n_head"]
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg["vocab_size"], C),            # token   embedding
            wpe=nn.Embedding(cfg["block_size"], C),            # position embedding (learned)
            h=nn.ModuleList([Block(C, nh, cfg["block_size"]) for _ in range(cfg["n_layer"])]),
            ln_f=nn.LayerNorm(C),                              # final layer norm
        ))
        self.lm_head = nn.Linear(C, cfg["vocab_size"], bias=False)   # NOT tied to wte

    def forward(self, idx, trace=False):
        B, T = idx.shape
        # === Embedding ===  token vector + position vector, summed
        tok = self.transformer.wte(idx)                        # (B,T,C)
        pos = self.transformer.wpe(torch.arange(T, device=idx.device))   # (T,C)
        x = tok + pos
        if trace:
            print("=== Embedding ===  (token ids -> wte, plus learned wpe)")
            p("token embedding wte", tok)
            p("position embedding wpe", pos)
            p("x = wte + wpe", x)

        # === Transformer (stack) ===  n_layer identical blocks, in sequence
        for li, block in enumerate(self.transformer.h):
            if trace:
                print(f"\n================ Transformer block {li} ================")
            x = block(x, trace=trace)

        # === Softmax / Output ===  final LN, then project to vocab logits
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        if trace:
            print("\n=== Softmax / Output ===")
            p("logits (last position)", logits[0, -1])
            p("probs  (last position)", F.softmax(logits[0, -1], -1), k=3)
        return logits


# ------------------------------------------------------------------ weight loading
def load_weights(model, path):
    """Fill `model` from the llm-viz JSON.  Byte order = little-endian float32,
    flatten = row-major.  `config` and the `attn.bias` mask buffer are skipped."""
    raw = json.load(open(path))
    sd = {}
    for k, v in raw.items():
        if k == "config" or k.endswith(".attn.bias"):   # skip config + causal-mask buffer
            continue                                     # (".attn.bias" != "...c_attn.bias")
        arr = np.frombuffer(base64.b64decode(v["data"]), dtype="<f4").reshape(v["shape"]).copy()
        sd[k] = torch.from_numpy(arr)
    model.load_state_dict(sd, strict=True)     # strict => any schema mismatch is caught
    return raw["config"]


# ------------------------------------------------------------------ greedy decode
@torch.no_grad()
def generate(model, prompt_ids, steps, trace=False):
    idx = torch.tensor([prompt_ids])
    if trace:                                  # trace the stations once, on the prompt
        model(idx, trace=True)
        print("\n=== greedy decode ===")
    for s in range(steps):
        cond = idx[:, -model.block_size:]      # crop context to block_size
        logits = model(cond)
        probs = F.softmax(logits[0, -1], dim=-1)
        nxt = int(probs.argmax())
        if trace:
            print(f"  step {s}: -> '{VOCAB[nxt]}'  p={probs[nxt]:.4f}  "
                  f"(A={probs[0]:.3f} B={probs[1]:.3f} C={probs[2]:.3f})")
        idx = torch.cat([idx, torch.tensor([[nxt]])], dim=1)
    return idx[0, len(prompt_ids):].tolist()   # only the newly generated tokens


def accuracy(model, n, length=6, seed=0):
    rng = random.Random(seed)
    ok = 0
    for _ in range(n):
        seq = "".join(rng.choice(VOCAB) for _ in range(length))
        out = decode(generate(model, encode(seq), length))
        ok += (out == "".join(sorted(seq)))
    return ok / n


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="gpt-nano-sort-model.json")
    ap.add_argument("--input", default="CBABBC")
    ap.add_argument("--steps", type=int, default=6)      # 6 letters -> 6 sorted letters
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--eval", type=int, default=20)      # random-accuracy sample count
    args = ap.parse_args()

    torch.manual_seed(0)
    with open(args.weights) as f:
        cfg = json.load(f)["config"]
    model = GPT(cfg).eval()
    load_weights(model, args.weights)
    print(f"loaded {args.weights}: n_layer={cfg['n_layer']} n_head={cfg['n_head']} "
          f"n_embd={cfg['n_embd']} block_size={cfg['block_size']} "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    out = decode(generate(model, encode(args.input), args.steps, trace=args.trace))
    want = "".join(sorted(args.input))
    print(f"\ninput  '{args.input}'  ->  greedy '{out}'   (sorted = '{want}')  "
          f"{'OK' if out == want else 'MISMATCH'}")

    if args.eval > 0:
        acc = accuracy(model, args.eval)
        print(f"random sort accuracy over {args.eval} length-6 sequences: {acc*100:.1f}%")


if __name__ == "__main__":
    main()
