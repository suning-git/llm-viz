# minimal_gpt —— 课上 3D 可视化背后的同一个模型

这个文件夹里是第一课 3D 可视化网站背后的**同一个模型**：gpt-nano
（3 层 × 2 头 × 48 维，约 85,000 个参数），训练在玩具语言 G 上。
权重文件 `gpt-nano-toylang-model.json` 和网站加载的是同一份——
你在这里打印出的每一个数，都能在网站上悬停对上。

**全程只需要 CPU**，普通笔记本即可；前两节课不需要任何 GPU。

## 语言 G 的三条规则

1. C 之后，永远是 A；
2. 连续的 A、B 之后，永远是 C；
3. 其余位置，A 或 B 自由出现（各 50%）。

例如 `B A B C A B C A A B C` 是一句合法的话。

## 环境（一次性）

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch numpy
```

## 三种玩法

```bash
# ① 推理：载入训练好的权重，接着 "BABCAB" 往下生成，并检查是否守规则
python minimal_gpt.py

# ② 逐站追踪：把 Embedding → LayerNorm → Attention → MLP → … → Output
#    每一站的张量形状和数值打印出来——对照 3D 网站逐站核对（背诵作业的自检工具）
python minimal_gpt.py --input BABCAB --trace

# ③ 从零训练你自己的模型（CPU 几分钟）：现场生成 10,000 句语言 G 的话，
#    训练同结构的 gpt-nano，导出 gpt-nano-toylang-model-mine.json
python train_toylang.py
```

`minimal_gpt.py` 是一个从上到下顺序可读的单文件（不到 300 行），
分节标题和网站的章节一一对应。**建议的读法**：开着 3D 网站，
`--trace` 的输出放旁边，每一站问自己"这个箭头上流过去的是什么形状"。

## 两个提示

- 训练脚本每 100 步会打印一张"分语境交叉熵"表（`CE[C]` / `CE[AB]` / `CE[free]`）。
  前两个会跌到 0，第三个会停在一个怎么也下不去的数附近——**它为什么下不去、
  停在的那个数是什么**，是第二次课的内容。带着这个问题来。
- 改动模型再训练（比如砍掉一个头、去掉 LayerNorm）完全欢迎——这正是
  周日 Hackathon 的工作方式。但请和你自己跑出的未改动版本对照。
