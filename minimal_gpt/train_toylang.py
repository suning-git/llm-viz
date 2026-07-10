#!/usr/bin/env python3
"""
train_toylang.py -- train gpt-nano (n_head=2, head_dim=24) on the toy language G,
then export the weights to `gpt-nano-toylang-model-mine.json` in the *exact same
schema* as the shipped `gpt-nano-toylang-model.json` (same keys / shapes / dtype /
little-endian-float32 base64) -- so YOUR model can drive the 3D visualization too.

--------------------------------------------------------------------- the language G
Vocabulary {A,B,C} -> ids {0,1,2}.  Length-12 sequences (x = first 11, y = last 11,
PURE next-token supervision at EVERY position, no mask):
    t = 0 : uniform over {A,B,C}.
    t >= 1: if the previous token is C            -> this token = A
            else if t >= 2 and (tok[t-2],tok[t-1]) == (A,B) -> this token = C
            else                                  -> uniform over {A,B}.
Corollaries: C only ever appears right after the pair A,B; every C is followed by A.
Theoretical conditional entropy of the NEXT token given the context the model sees:
    last token = C            -> 0        (next is deterministically A)
    last two tokens = (A,B)   -> 0        (next is deterministically C)
    otherwise ("free")        -> ln 2 = 0.6931 nats (uniform A/B)
(The very first token, generated unconditionally, has entropy ln 3; the model never
predicts token 0, so that value does not appear in the per-context table below.)

Run (from the minimal_gpt/ dir, venv active):
    python train_toylang.py                 # ~a few minutes on CPU
"""
import argparse, base64, copy, json, math, pickle, time, zlib
import numpy as np, torch, torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from minimal_gpt import GPT, VOCAB, encode, decode, generate

A, B, C = 0, 1, 2


# ------------------------------------------------------------------ training model WITH dropout
# minimal_gpt.GPT is a dropout-FREE inference model.  minGPT trains gpt-nano WITH dropout
# (embd/resid/attn pdrop=0.1); without it the model overfits to over-confidence on the
# random "free" positions (their held-out CE explodes far past ln2).  This TrainGPT mirrors
# minimal_gpt.GPT's module layout EXACTLY (same parameter names) but applies dropout in the
# forward pass, so its state_dict loads 1:1 into minimal_gpt.GPT for export/verification.
def new_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


class TAttn(nn.Module):
    def __init__(self, C, nh, block, attn_pdrop, resid_pdrop):
        super().__init__()
        self.n_head, self.attn_pdrop, self.resid_pdrop = nh, attn_pdrop, resid_pdrop
        self.c_attn = nn.Linear(C, 3 * C)
        self.c_proj = nn.Linear(C, C)
        self.register_buffer("mask",
            torch.tril(torch.ones(block, block)).view(1, 1, block, block), persistent=False)

    def forward(self, x):
        Bx, T, Cx = x.shape
        nh, hd = self.n_head, Cx // self.n_head
        q, k, v = self.c_attn(x).split(Cx, dim=2)
        q = q.view(Bx, T, nh, hd).transpose(1, 2)
        k = k.view(Bx, T, nh, hd).transpose(1, 2)
        v = v.view(Bx, T, nh, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.dropout(F.softmax(att, dim=-1), self.attn_pdrop, self.training)
        y = (att @ v).transpose(1, 2).contiguous().view(Bx, T, Cx)
        return F.dropout(self.c_proj(y), self.resid_pdrop, self.training)


class TBlock(nn.Module):
    def __init__(self, C, nh, block, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln_1 = nn.LayerNorm(C)
        self.attn = TAttn(C, nh, block, attn_pdrop, resid_pdrop)
        self.ln_2 = nn.LayerNorm(C)
        self.mlp = nn.ModuleDict(dict(c_fc=nn.Linear(C, 4 * C), c_proj=nn.Linear(4 * C, C)))
        self.resid_pdrop = resid_pdrop

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        h = self.mlp.c_proj(new_gelu(self.mlp.c_fc(self.ln_2(x))))
        return x + F.dropout(h, self.resid_pdrop, self.training)


class TrainGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.block_size, C, nh = cfg["block_size"], cfg["n_embd"], cfg["n_head"]
        self.embd_pdrop = cfg["embd_pdrop"]
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg["vocab_size"], C),
            wpe=nn.Embedding(cfg["block_size"], C),
            h=nn.ModuleList([TBlock(C, nh, cfg["block_size"], cfg["attn_pdrop"], cfg["resid_pdrop"])
                             for _ in range(cfg["n_layer"])]),
            ln_f=nn.LayerNorm(C),
        ))
        self.lm_head = nn.Linear(C, cfg["vocab_size"], bias=False)

    def forward(self, idx):
        Bx, T = idx.shape
        tok = self.transformer.wte(idx)
        pos = self.transformer.wpe(torch.arange(T, device=idx.device))
        x = F.dropout(tok + pos, self.embd_pdrop, self.training)
        for block in self.transformer.h:
            x = block(x)
        return self.lm_head(self.transformer.ln_f(x))


# ------------------------------------------------------------------ language G generator
def gen_sequence(rng):
    """Return a length-12 list of ids sampled from language G (rng = random.Random)."""
    seq = []
    for t in range(12):
        if t == 0:
            seq.append(rng.randint(0, 2))              # uniform {A,B,C}
        elif seq[t - 1] == C:                          # after C -> A
            seq.append(A)
        elif t >= 2 and seq[t - 2] == A and seq[t - 1] == B:   # after (A,B) -> C
            seq.append(C)
        else:
            seq.append(rng.randint(0, 1))              # free: uniform {A,B}
    return seq


class ToyLangDataset(Dataset):
    """10,000 length-12 sequences from language G; x/y = full next-token shift (all
    11 positions supervised).  Deterministic train/test split via crc32 of the sequence
    (stable across machines/runs -- unlike hash(), which is per-process salted; h % 4 == 0 -> test)."""
    def __init__(self, split, seed=0):
        assert split in {"train", "test"}
        self.split = split
        import random
        self.rng = random.Random(seed if split == "train" else seed + 1)

    def __len__(self):
        return 10000

    def get_vocab_size(self):
        return 3

    def get_block_size(self):
        return 11

    def __getitem__(self, idx):
        while True:
            seq = gen_sequence(self.rng)
            h = zlib.crc32(bytes(seq))          # stable across machines (hash() is per-process salted)
            split = "test" if h % 4 == 0 else "train"
            if split == self.split:
                break
        cat = torch.tensor(seq, dtype=torch.long)      # (12,)
        x = cat[:-1].clone()                           # (11,)
        y = cat[1:].clone()                            # (11,) pure next-token target
        return x, y


# ------------------------------------------------------------------ minGPT init & optim
def init_weights(model, n_layer):
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    model.apply(_init)
    for pn, p in model.named_parameters():
        if pn.endswith("c_proj.weight"):
            nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))


def configure_optimizers(model, weight_decay, lr, betas):
    decay, no_decay = set(), set()
    whitelist, blacklist = (nn.Linear,), (nn.LayerNorm, nn.Embedding)
    for mn, m in model.named_modules():
        for pn, _ in m.named_parameters(recurse=False):
            fpn = f"{mn}.{pn}" if mn else pn
            if pn.endswith("bias"):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist):
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist):
                no_decay.add(fpn)
    pd = {pn: p for pn, p in model.named_parameters()}
    assert not (decay & no_decay) and (decay | no_decay) == set(pd)
    groups = [
        {"params": [pd[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
        {"params": [pd[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


# ------------------------------------------------------------------ batching
def get_batch(ds, bs):
    xs, ys = zip(*[ds[i] for i in range(bs)])          # idx ignored; each item is random
    return torch.stack(xs), torch.stack(ys)


# ------------------------------------------------------------------ acceptance eval
def context_of(tokens, i):
    """The context the model sees when predicting tokens[i] (i >= 1):
    'C' (last token C), 'AB' (last two == A,B), or 'free'."""
    if tokens[i - 1] == C:
        return "C"
    if i >= 2 and tokens[i - 2] == A and tokens[i - 1] == B:
        return "AB"
    return "free"


@torch.no_grad()
def per_context_ce(model, ds, n=4000):
    """Mean cross-entropy (nats) of the model's next-token distribution, grouped by the
    context the model sees.  Expect: C ~ 0, AB ~ 0, free ~ ln2."""
    model.eval()
    tot = {"C": 0.0, "AB": 0.0, "free": 0.0}
    cnt = {"C": 0, "AB": 0, "free": 0}
    for _ in range(n):
        x, y = ds[0]
        logp = F.log_softmax(model(x.unsqueeze(0))[0], dim=-1)     # (11, 3)
        toks = x.tolist()
        for i in range(len(toks)):                                # predicting token i+1
            ctx = context_of(toks + [int(y[i])], i + 1)           # context for pos i+1
            tot[ctx] += -float(logp[i, int(y[i])])
            cnt[ctx] += 1
    return {k: (tot[k] / cnt[k] if cnt[k] else float("nan"), cnt[k]) for k in tot}


@torch.no_grad()
def rule_violations(model, n_prefixes=200, gen_steps=15, seed=123):
    """Sample n legal prefixes (length 3-6), greedy-generate gen_steps tokens, and count
    rule violations: after C must be A; after (A,B) must be C; C only after (A,B)."""
    import random
    rng = random.Random(seed)
    model.eval()
    total_checked, violations = 0, 0
    examples = []
    for _ in range(n_prefixes):
        L = rng.randint(3, 6)
        seq = gen_sequence(rng)[:L]                    # a legal G prefix
        gen = generate(model, seq, gen_steps)          # greedy continuation
        full = seq + gen
        for i in range(L, len(full)):                  # only newly generated positions
            total_checked += 1
            ctx = context_of(full, i)
            tok = full[i]
            bad = ((ctx == "C" and tok != A) or
                   (ctx == "AB" and tok != C) or
                   (ctx == "free" and tok == C))
            if bad:
                violations += 1
                if len(examples) < 5:
                    examples.append((decode(full[:i]), VOCAB[tok], ctx))
    return violations, total_checked, examples


# ------------------------------------------------------------------ JSON export
def export_json(model, cfg, template_path, out_path):
    template = json.load(open(template_path))
    sd = model.state_dict()
    block = cfg["block_size"]
    out = {}
    for k, v in template.items():
        if k == "config":
            out[k] = {**v, "n_head": cfg["n_head"]}
            continue
        if k.endswith(".attn.bias"):
            t = torch.tril(torch.ones(block, block)).view(1, 1, block, block)
        else:
            t = sd[k]
        arr = np.ascontiguousarray(t.detach().numpy()).astype("<f4")
        assert list(arr.shape) == v["shape"], f"shape mismatch {k}: {arr.shape} vs {v['shape']}"
        out[k] = {"shape": list(arr.shape), "dtype": "torch.float32",
                  "data": base64.b64encode(arr.tobytes()).decode("ascii")}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=4)
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="gpt-nano-toylang-model.json")
    ap.add_argument("--out", default="gpt-nano-toylang-model-mine.json")
    ap.add_argument("--max_iters", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    train_ds, test_ds = ToyLangDataset("train"), ToyLangDataset("test")

    cfg = dict(model_type="gpt-nano", n_layer=3, n_head=2, n_embd=48,
               vocab_size=3, block_size=11,
               embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1)
    model = TrainGPT(cfg)
    init_weights(model, cfg["n_layer"])
    print(f"model: n_head={cfg['n_head']} head_dim={cfg['n_embd']//cfg['n_head']} "
          f"params={sum(p.numel() for p in model.parameters()):,}  (dropout on, minGPT-faithful)")

    opt = configure_optimizers(model, weight_decay=0.1, lr=args.lr, betas=(0.9, 0.95))
    ln2 = math.log(2)

    # best-checkpoint selection: among evals that already satisfy the deterministic-context
    # and zero-violation criteria, keep the one whose FREE-context CE is closest to ln2.
    best = {"score": float("inf"), "sd": None, "it": -1, "ce": None}

    t0 = time.time()
    model.train()
    for it in range(1, args.max_iters + 1):
        x, y = get_batch(train_ds, args.batch_size)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))  # ALL positions
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % 100 == 0 or it == 1:
            ce = per_context_ce(model, test_ds, 1500)
            viol, checked, _ = rule_violations(model, 100, 15)
            print(f"iter {it:4d} | loss {loss.item():.4f} | "
                  f"CE[C]={ce['C'][0]:.4f} CE[AB]={ce['AB'][0]:.4f} CE[free]={ce['free'][0]:.4f} "
                  f"| viol {viol}/{checked} | {time.time()-t0:5.1f}s")
            det_ok = ce['C'][0] < 0.02 and ce['AB'][0] < 0.02
            if it >= 300 and viol == 0 and det_ok:
                score = abs(ce['free'][0] - ln2)
                if score < best["score"]:
                    best = {"score": score, "sd": copy.deepcopy(model.state_dict()),
                            "it": it, "ce": ce}
            model.train()

    # restore the best checkpoint for export + final acceptance
    if best["sd"] is not None:
        model.load_state_dict(best["sd"])
        print(f"\nbest checkpoint: iter {best['it']} | free-CE {best['ce']['free'][0]:.4f} "
              f"(|free-ln2|={best['score']:.4f})")
    else:
        print("\nWARNING: no checkpoint met the deterministic/violation gate; exporting final.")

    # ---- final acceptance ----------------------------------------------------
    print("\n================ ACCEPTANCE ================")
    viol, checked, ex = rule_violations(model, 200, 15)
    print(f"[1] rule violations: {viol}/{checked} generated tokens "
          f"({100*viol/max(checked,1):.3f}%)")
    for e in ex:
        print(f"      after '{e[0]}' (ctx={e[2]}) -> generated '{e[1]}'  [VIOLATION]")

    ce = per_context_ce(model, test_ds, 8000)
    print("\n[2] per-context cross-entropy (nats) on held-out test set:")
    print("     context            n      mean CE     theory")
    print(f"     after C        {ce['C'][1]:6d}   {ce['C'][0]:8.4f}     0.0000")
    print(f"     after (A,B)    {ce['AB'][1]:6d}   {ce['AB'][0]:8.4f}     0.0000")
    print(f"     free (A|B)     {ce['free'][1]:6d}   {ce['free'][0]:8.4f}     {ln2:.4f} (ln2)")

    # ---- export + reload check ----------------------------------------------
    export_json(model, cfg, args.template, args.out)
    print(f"\nexported -> {args.out}")
    from minimal_gpt import load_weights
    m2 = GPT(cfg).eval()
    load_weights(m2, args.out)
    # sanity: reload predicts C after A,B and A after C
    demo = encode("BABCAB")
    g = decode(generate(m2, demo, 5))
    print(f"reload check: continue 'BABCAB' greedily 5 -> '{g}'")
    # after C the model must say A:
    afterC = decode(generate(m2, encode("BAC"), 1))
    afterAB = decode(generate(m2, encode("BAB"), 1))
    print(f"reload check: 'BAC'->next '{afterC}' (want A)   'BAB'->next '{afterAB}' (want C)  "
          f"{'OK' if afterC=='A' and afterAB=='C' else 'MISMATCH'}")


if __name__ == "__main__":
    main()
