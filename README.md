<!-- markdownlint-disable MD041 -->

> **1Cat-vLLM 1.5.1-heavily-modified-v1**：这是基于 1Cat 官方代码的魔改分支（默认分支 `heavily-modified`，基于官方 `main@02c87ab89`，即 v1.5.0 之后第 670 个提交）。它面向 4 张 V100 上的 Swift 1.5 Qwen3.8-Flash-Next，重点优化并发吞吐和 prefill。改动清单、开关和与官方 `main@357d07bcb` 的实测对比见 [HEAVILY_MODIFIED.md](HEAVILY_MODIFIED.md)。官方原版请看 [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)。
>
> This is a heavily modified fork of 1Cat-vLLM (branch `heavily-modified`, based on upstream `main@02c87ab89`) tuned for concurrency and prefill of Swift 1.5 Qwen3.8-Flash-Next on 4x V100. See [HEAVILY_MODIFIED.md](HEAVILY_MODIFIED.md) for the changes, switches and measurements.
