# LLM Visualization · 课程定制版

A 3D interactive visualization of a GPT-style language model, running a real (tiny)
network right in the browser. This is a fork of Brendan Bycroft's
**[llm-viz](https://github.com/bbycroft/llm-viz)**, customized for the course
**《从零构建智能模型》**（清华大学 · 丘成桐数学科学中心）.

Original, by the author, live at **https://bbycroft.net/llm**

## 致谢 · Credits

- **[Brendan Bycroft](https://github.com/bbycroft/llm-viz)** — the original 3D LLM
  visualization (`llm-viz`). Everything rendered here is his work; this fork only swaps
  in the course's own models and lightly edits the walkthrough text. See also
  [`CREDITS.md`](./CREDITS.md).
- **[Andrej Karpathy · minGPT](https://github.com/karpathy/minGPT)** — the toy A/B/C
  sorting model that the default network is based on.

## 这个 fork 改了什么 · What's different

Kept **only** the LLM visualization from upstream. Removed the author's personal
homepage, the RISC-V CPU simulator, and the fluid-sim demo.

- `src/` · `public/` — the visualization app (walkthrough narration lightly edited).
- `public/gpt-nano-*.json` — the course's own trained models, shown in the viz.
- `model-gen/` — the training scripts that produce those models
  (`train_toylang.py` / `train_2head.py`), alongside their exported weights.
- `minimal_gpt/` — the **same toy model, runnable on your own laptop (CPU only)**:
  load the weights, print every tensor, and hover-match it against the 3D site.
  See [`minimal_gpt/README.md`](./minimal_gpt/README.md).

## 本地运行 · Running locally

```bash
yarn                # install dependencies
yarn dev            # dev server at http://localhost:3002
yarn build          # static export → ./out/
```

## 部署 · Deploy (GitHub Pages)

Configured for a project page at `https://<user>.github.io/llm-viz/`
(`basePath: '/llm-viz'` in `next.config.js`). Serve the `out/` static export via Pages.
