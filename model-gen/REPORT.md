# minimal_gpt — teaching kit for the bbycroft.net/llm walkthrough

## TL;DR (plain language)

Two things are done and both pass their acceptance tests.

1. **`minimal_gpt.py`** is a single, readable PyTorch/CPU file that loads the *exact*
   weights behind the [bbycroft.net/llm](https://bbycroft.net/llm) 3D visualization
   (minGPT's "gpt-nano", ~85k parameters, trained to sort 6 letters A/B/C). It is
   organised into the same "stations" the 3D tour walks through, and a `--trace` flag
   prints every station's tensor shapes and values so you can hover on the website and
   read the same numbers off the terminal.
   - **Acceptance met:** greedy decode of `CBABBC` → **`ABBBCC`** (the model is 100%
     sure the first output is `A`, matching the tour's narration), and **100%** sort
     accuracy on 20 (and on 200) random length-6 sequences.

2. **`gpt-nano-sort-model-2head.json`** is a classroom variant retrained with
   **`n_head=2`** (2 attention heads of 24 dims each, instead of 3×16 — easier to read
   in the 3D view). It is exported in the **byte-for-byte same JSON schema** as the
   original, and loads unchanged in `minimal_gpt.py`.
   - **Acceptance met:** **100%** sort accuracy (≥95% required); `--trace` shows exactly
     **2 heads × 24 dims**; JSON schema is identical (same 45 keys, order, shapes, dtypes
     — only `config.n_head` changes 3→2).
   - **Course change applied** (per the coordinator): the 2-head model is trained as a
     *pure next-token predictor on **every** position* (no loss masking of the input
     half), so it is a faithful teaching example of "a language model predicts the next
     token at every position; one sequence of length T is T training examples."

Jargon used below: **n_head** = number of attention heads; **head_dim** = n_embd / n_head
= dimensions each head works in; **pre-LN** = LayerNorm applied *before* attention/MLP;
**entropy** (in *nats*, natural-log units) = how spread-out a probability distribution is
— `ln(3) ≈ 1.099` nats is maximum for 3 classes (perfectly uniform / "no idea"), `0` nats
means fully certain.

---

## 1. The weight-JSON schema — exact rules

Source of truth: llm-viz `src/utils/tensor.ts` (`TensorF32.fromJson`) and
`src/utils/data.ts` (`base64ToArrayBuffer`), read from the local clone at
`../llm-viz-local/llm-viz`.

Each entry looks like:
```json
"transformer.wte.weight": { "shape": [3, 48], "dtype": "torch.float32", "data": "<base64>" }
```

**Decoding rule (verified in code):**
```
data  --window.atob()-->  raw bytes  --new Float32Array(buffer)-->  float32 values
```
- **Byte order: little-endian float32.** `Float32Array` reads the buffer in the host's
  native order, which in every browser (x86/ARM) is little-endian. So the correct NumPy
  dtype is `'<f4'`.
- **Flatten order: row-major (C-order).** The flat float array is reshaped to `shape`
  with standard PyTorch/NumPy contiguous layout (the `indexIterator` in `tensor.ts`
  walks `[0,0],[0,1],…,[1,0],…`, i.e. last axis fastest).
- **Linear weights are stored `[out_features, in_features]`** (PyTorch `nn.Linear`
  convention), so `y = x @ W.T + b` — no transpose needed when loading into `nn.Linear`.

One line reproduces any tensor:
```python
np.frombuffer(base64.b64decode(entry["data"]), dtype="<f4").reshape(entry["shape"])
```

**Export (round-trip) rule:** `np.ascontiguousarray(t.numpy()).astype("<f4").tobytes()`
then base64 — C-order bytes, little-endian float32. `train_2head.py` uses exactly this.

**The 45 keys** (order preserved from the original file):
- `config` — the hyperparameter dict.
- `transformer.wte.weight` `[3,48]` (token embedding), `transformer.wpe.weight` `[11,48]`
  (learned position embedding).
- For each layer `h.{0,1,2}`:
  `ln_1.weight/bias` `[48]`; **`attn.bias` `[1,1,11,11]`** (the causal lower-triangular
  mask — a registered *buffer*, **not** a learnable parameter); `attn.c_attn.weight`
  `[144,48]` + `.bias` `[144]` (the fused Q,K,V projection, 144 = 3×48);
  `attn.c_proj.weight` `[48,48]` + `.bias` `[48]` (output projection); `ln_2.weight/bias`
  `[48]`; `mlp.c_fc.weight` `[192,48]` + `.bias` `[192]` (widen 48→192);
  `mlp.c_proj.weight` `[48,192]` + `.bias` `[48]` (narrow 192→48).
- `transformer.ln_f.weight/bias` `[48]` (final LayerNorm).
- **`lm_head.weight` `[3,48]`** — a **separate** key. minGPT does **not** tie it to
  `wte` (verified numerically: `lm_head.weight != wte.weight`), so `minimal_gpt.py` loads
  it as its own `nn.Linear(48, 3, bias=False)`.

Loading notes in `minimal_gpt.py`: `config` and the three `*.attn.bias` mask buffers are
skipped (the causal mask is rebuilt with `torch.tril`); everything else maps 1:1 to module
parameters and is loaded with `strict=True` (so any schema drift is caught immediately).

---

## 2. Deliverable ① — `minimal_gpt.py` vs the original weights

**Architecture** (matches minGPT `mingpt/model.py` exactly): token emb + learned pos emb;
3 pre-LN blocks each = `x = x + attn(ln_1(x))` then `x = x + mlp(ln_2(x))`; attention has
bias, is causal, 3 heads; MLP is `c_fc → NewGELU → c_proj`; **NewGELU is minGPT's tanh
approximation** `0.5·x·(1+tanh(√(2/π)·(x+0.044715·x³)))`, *not* the exact erf GELU; final
LayerNorm; untied `lm_head`. Loaded param count = **85,728** (~85k, as advertised).

Attention is written with the QKV/head-split order spelled out as separate lines (the
teaching point — **project first, split into heads second**):
```python
qkv = self.c_attn(x)                          # (B,T,3C) — one matmul makes Q,K,V
q, k, v = qkv.split(C, dim=2)                 # three (B,T,C)
q = q.view(B, T, nh, hd).transpose(1, 2)      # THEN split into heads -> (B,nh,T,hd)
...
```

### Acceptance results

| check | result |
|---|---|
| greedy decode of `CBABBC` (6 steps) | **`ABBBCC`** ✓ (= `sorted("CBABBC")`) |
| per-step confidence | `A`(p=1.0000) `B`(0.9999) `B`(1.0000) `B`(0.9998) `C`(1.0000) `C`(1.0000) |
| sort accuracy, 20 random length-6 seqs | **100.0%** ✓ |
| sort accuracy, 200 random length-6 seqs | **100.0%** |

Step 0 predicting `A` with probability **1.0000** matches the walkthrough's narration that
"the model is quite sure the next token is A".

> Note on step count: the task brief said "decode 5 steps", but producing the full 6-letter
> sorted output from a 6-letter input needs **6** autoregressive steps (feed 6 → predict
> the 7th = first sorted letter → … → 12th). `--steps` defaults to 6. (The generator crops
> context to `block_size=11` on the last step, exactly like minGPT's `generate`.)

### Numbers you can cross-check against the website's hover values

For input `CBABBC` (token ids `[2,1,0,1,1,2]`, mapping **A=0, B=1, C=2**), position 0
(letter `C`), first 4 of 48 dims:

| tensor (hover target on the site) | value `[:4]` |
|---|---|
| `wte[C]`  (token-embedding row for C) | `[-0.0461, -0.0277, +0.0654, +0.0605]` |
| `wpe[0]`  (position-0 embedding) | `[+0.0131, -0.0189, -0.0390, +0.0231]` |
| **input embedding** = `wte[C] + wpe[0]` | `[-0.0330, -0.0466, +0.0264, +0.0836]` |

Other token-embedding rows (first 4 dims): `wte[A] = [+0.0598, +0.0015, -0.0064, -0.0022]`,
`wte[B] = [-0.0220, +0.0100, -0.0026, -0.0023]`. `--trace` prints these exact values at the
`=== Embedding ===` station, plus the full 6×6 causal-softmax matrix for each of the 3
heads at every layer.

---

## 3. Deliverable ② — 2-head retrain (`train_2head.py` → `gpt-nano-sort-model-2head.json`)

**What changed vs the original model:** `n_head` 3 → **2** (head_dim 16 → **24**). Note that
changing the head count **does not change any weight shape** — `c_attn` stays `[144,48]`,
`c_proj` stays `[48,48]`; heads only re-partition the 48 q/k/v dims (48/2 = 24). So the
exported JSON is schema-identical to the original by construction.

**Course change (per coordinator):** trained as a **pure next-token predictor on every
position** — the minGPT `SortDataset` mask `y[:length-1] = -1` is removed, so the input
half participates in the loss too. Everything else follows minGPT faithfully: same data
sampler, `N(0, 0.02)` init with the `c_proj` residual-scaled init `0.02/√(2·n_layer)`,
AdamW with decay/no-decay parameter groups, `grad_norm_clip=1.0`.

### Training config & curve

`n_layer=3, n_head=2, n_embd=48, vocab_size=3, block_size=11` · optimizer AdamW
`lr=5e-4, betas=(0.9,0.95), weight_decay=0.1` · `batch_size=64` · seed 1337 ·
`max_iters=4000` with early stop at ≥99.9% test accuracy · **CPU, ~13 s total**.

| iter | loss | test sort acc |
|---|---|---|
| 1 | 1.075 | 0.0% |
| 250 | 0.506 | 99.4% |
| 500 | 0.502 | 100.0% |
| 1000 | 0.500 | 99.4% |
| 1500 | 0.499 | **100.0% → early stop** |

**FINAL greedy sort accuracy: 100.00%** over 166 unique held-out (test-split) sequences.
(Loss plateaus near ~0.50 rather than ~0 because positions 0–4 are *unpredictable random
inputs* — their irreducible cross-entropy is ≈ ln(3) ≈ 1.10 nats, averaged with the ≈0 loss
of the sorted positions. That is expected and correct, not underfitting.)

### Position-behaviour report (the teaching artefact)

Every position simultaneously predicts *its* next token. Positions 0–4 are asked to predict
the **next random input digit** (impossible → the model outputs the ~uniform marginal);
positions 5–10 predict the **sorted output** (fully learned).

Averaged over 2000 random sequences:

| position | predicts | avg max-prob | avg entropy (nats) | next-token acc |
|---|---|---|---|---|
| 0 | next input digit | 0.344 | 1.098 | 32.0% |
| 1 | next input digit | 0.346 | 1.098 | 35.2% |
| 2 | next input digit | 0.342 | 1.098 | 34.1% |
| 3 | next input digit | 0.351 | 1.097 | 36.0% |
| 4 | next input digit | 0.354 | 1.097 | 36.8% |
| 5 | sorted output | 1.000 | 0.002 | **100.0%** |
| 6 | sorted output | 1.000 | 0.002 | **100.0%** |
| 7 | sorted output | 1.000 | 0.002 | **100.0%** |
| 8 | sorted output | 1.000 | 0.003 | **100.0%** |
| 9 | sorted output | 1.000 | 0.003 | **100.0%** |
| 10 | sorted output | 1.000 | 0.003 | **100.0%** |

Positions 0–4 sit essentially **on the uniform baseline** (max-prob 0.333, entropy
ln(3)=1.099) — the model correctly learns it *cannot* predict random inputs and hedges
uniformly. Positions ≥5 are near-deterministic and 100% correct.

Concrete example — `CBABBC` fed as the full 11-token training sequence
`C B A B B C A B B B C` (input + sorted[:-1]), all 11 output distributions:

| pos | in | target | pred | P(A) | P(B) | P(C) | entropy | correct |
|---|---|---|---|---|---|---|---|---|
| 0 | C | B | B | 0.325 | 0.346 | 0.329 | 1.098 | ✓ (chance) |
| 1 | B | A | B | 0.319 | 0.355 | 0.325 | 1.098 | ✗ |
| 2 | A | B | C | 0.310 | 0.340 | 0.350 | 1.097 | ✗ |
| 3 | B | B | C | 0.327 | 0.329 | 0.344 | 1.098 | ✗ |
| 4 | B | C | A | 0.350 | 0.335 | 0.315 | 1.098 | ✗ |
| 5 | C | **A** | **A** | **1.000** | 0.000 | 0.000 | 0.002 | ✓ |
| 6 | A | **B** | **B** | 0.000 | **1.000** | 0.000 | 0.002 | ✓ |
| 7 | B | **B** | **B** | 0.000 | **1.000** | 0.000 | 0.002 | ✓ |
| 8 | B | **B** | **B** | 0.000 | **1.000** | 0.000 | 0.003 | ✓ |
| 9 | B | **C** | **C** | 0.000 | 0.000 | **1.000** | 0.005 | ✓ |
| 10 | C | **C** | **C** | 0.000 | 0.000 | **1.000** | 0.003 | ✓ |

Reading the confident predictions at positions 5–10 top-to-bottom spells the sorted answer
**A B B B C C**. (Positions 0–4 are ~uniform — the model has no way to know the next random
letter, exactly the intended lesson. The occasional "correct" there, e.g. pos 0, is chance.)

### Schema parity & validation

- **Same 45 keys, same order, same shapes, same dtype** as the original; the *only* JSON
  difference is `config.n_head: 3 → 2` (all other config fields identical). The `attn.bias`
  causal-mask buffers are regenerated (`tril(ones(11,11))`) so the key list stays identical.
- Loaded back through `minimal_gpt.py` unchanged: `CBABBC` → **`ABBBCC`** ✓, **100%** on 50
  random sequences (≥95% required); `--trace` prints
  `split -> Q,K,V each (1, 2, 6, 24)   (n_head=2, head_dim=24)` and shows **2** head
  attention matrices.

> Aside (subtle, worth knowing before class): minGPT's `SortDataset` "boost-repeats"
> rejection-sampling branch (`if unique > length//2: continue`) is a **no-op** for these
> hyperparameters — with `num_digits=3, length=6`, `length//2 = 3` and the unique-digit
> count can never exceed 3. So the training sequences are simply uniform-random, which is
> why the positions-0–4 marginal comes out cleanly uniform. Kept in the code for fidelity.

---

## 4. How to run

```bash
cd minimal_gpt
source .venv/bin/activate           # torch 2.12.1+cpu, numpy 2.5.1

# Deliverable ① — original 3-head model
python minimal_gpt.py                                   # decode CBABBC + 20-sample accuracy
python minimal_gpt.py --input CBABBC --trace --eval 0   # print every station (shapes+tensors)
python minimal_gpt.py --eval 200                        # bigger accuracy check

# Deliverable ② — 2-head model
python train_2head.py                                   # retrain + export (~13 s)
python minimal_gpt.py --weights gpt-nano-sort-model-2head.json --trace   # 2 heads × 24 dims
```

## 5. File listing (`minimal_gpt/`)

| file | size | purpose |
|---|---|---|
| `minimal_gpt.py` | 244 lines | ① single-file inference + `--trace` + greedy decode + accuracy |
| `train_2head.py` | 269 lines | ② faithful minGPT retrain (n_head=2, all-position loss) + JSON export |
| `gpt-nano-sort-model.json` | 456 KB | original weights (downloaded from llm-viz; n_head=3) |
| `gpt-nano-sort-model-2head.json` | 456 KB | ② exported classroom weights (n_head=2), same schema |
| `.venv/` | ~950 MB | CPU PyTorch environment (not for commit) |

(`minimal_gpt.py`'s core model + forward is ~100 lines; the remainder is the `--trace`
instrumentation, CLI, and the header docstring, all required by the brief.)
