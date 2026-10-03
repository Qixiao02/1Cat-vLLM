# sx_bench：压测脚本和原始结果

这个目录放的是 README 里那几张实测表用到的脚本和原始数据，任何人都可以拿同样的脚本在自己的机器上重测。

## 脚本

| 文件 | 用途 |
|---|---|
| `pfx_bench.py` | 压测本体。对一个已启动的 OpenAI 兼容接口，按固定并发发送冷 prompt，记录 prefill 和 decode |
| `summarize.py` | 把两条线的详细记录汇总成并排的 `summary_fork_vs_official.json` |
| `as_run/pfx_ab.sh` | 2026-09-30 “前缀缓存开/关”那一轮的外层脚本，原样保留 |
| `as_run/ab_detail.sh` | 2026-09-30 “本分支对官方”那一轮的外层脚本，原样保留 |
| `as_run/mtp_kv.sh`、`mtp_compose.py`、`mtp_chain.sh`、`mtp_summary.py` | 2026-09-30 开 MTP、FP16 KV 那一轮：生成 compose、启动、压测、汇总，原样保留 |
| `as_run/e4m3_chain.sh`、`e4m3_compose.py` | 2026-09-30 开 MTP、E4M3 KV 那一轮的全流程：标定实例、算 scale、生成带 scale 的模型目录、启动、对照输出、压测，原样保留 |
| `as_run/calib_traffic.py` | E4M3 标定时发的 18 条请求。文本取自服务器上的源码和文档，换机器要改路径 |
| `as_run/kvq_check.py` | 输出对照：长文本里找 8 个验证码，加 4 道短题，贪心解码 |
| `as_run/build_fork.sh`、`dockerfile-local-gitmirror.patch` | 2026-10-01 从源码构建 1001 的 wheel 和镜像。补丁只在那次构建时用：GitHub 链路太慢，CMake 要拉的第三方仓库改从本地镜像取（提交哈希都核对过），不在仓库里 |
| `as_run/arm_compose.py`、`run_all.sh`、`mtp_bench.py`、`cmp_summary.py` | 2026-10-01 本分支对官方最新（双方最佳参数）：`arm_compose.py` 写出 6 种配置的 compose（本分支和官方，各自开、关前缀缓存，以及两边开 MTP），`run_all.sh` 依次启动并测，`mtp_bench.py` 用自然 prompt 测 MTP，`cmp_summary.py` 汇总 |
| `as_run/wheel_check.sh`、`mk_forkwheel_compose.py` | 用 1001 的镜像按生产配置启动，和 0930 的镜像对照答题，再跑第 1 组的压测 |
| `as_run/arm_compose_1003.py` | 2026-10-03 本分支 1003 对官方 v1.5.1：写出各配置的 compose（F1、F2、FM、FM2 是本分支，V1、V2、V2b、V2c、VM、VMa 是官方 v1.5.1，O1、O2、OM 是官方 main 的几组），每组的完整启动参数和环境变量都在里面 |
| `as_run/run_all_1003.sh`、`run_main_final2.sh`、`run_v2b.sh`、`run_final2.sh`、`fm_answers_rerun.sh`、`arm_functions.sh` | 2026-10-03 那轮的外层脚本，按实际运行的顺序保留：一次只起一个实例，带内存和生产健康检查看门狗。`run_main_final2.sh` 等脚本里写死了我们服务器的路径和镜像名 |
| `as_run/build_v151.sh`、`gen_dockerfile_v151.py` | 官方 v1.5.1 的 wheel 要求 GLIBC ≥ 2.38，官方镜像是 Ubuntu 22.04，所以做了一个 Ubuntu 24.04 镜像：依赖和 CUDA 工具链取自官方镜像，只把 vllm 换成官方发布的 wheel（`--no-deps`） |
| `as_run/probe_official.sh` | 官方 `main@e53d02171` 起不来的探测脚本：起实例，图捕获结束后 4 分钟还不就绪就判定为挂起，自动保存 py-spy 栈 |
| `as_run/official-main-dockerfile-local.patch` | 构建官方 `main@e53d02171` 时对它的 `docker/Dockerfile` 做的 4 处本地补丁（补 COPY 两个文件、cpython 走镜像源、CMake 第三方库走本地镜像），没有改任何源码 |
| `as_run/build_1003.sh`、`validate_wheel_1003.sh` | 1003 wheel 的构建（仓库自己的 Dockerfile，参数和 1002 相同）和验证：取出 wheel，做验证镜像（1002 镜像换上新 wheel），按生产参数启动，答题加 8K 贪心 C1/C4 加 4 并发 8K–64K，对照第 5 组的覆盖镜像 |
| `as_run/make_report.py` | 从 `results/2026-10-03-fork-vs-official-v1.5.1/` 的原始日志算出第 5 组的所有对比表，不手算 |

`as_run/` 下的脚本带着我们服务器上的路径、容器名和 compose 文件名，换机器要改。`pfx_bench.py` 和 `summarize.py` 不依赖这些，只用 Python 标准库。

## 怎么跑

服务先用正常方式启动好，然后在同一台机器上执行：

```bash
python3 pfx_bench.py --port 8001 --model <served-model-name> --out result.json \
    --conc 4 --lengths 8000,16000,32000,64000 --gen 400 --passes 2 --gpus 0,1,2,3
```

- `--conc`：同时发出的请求数。
- `--lengths`：每条 prompt 的 token 数，用引擎自己的 `/tokenize` 校准，误差约 0.2%。
- `--gen`：每条请求固定生成的 token 数（`ignore_eos`）。
- `--passes`：每个长度重复几遍，每遍换一批新 prompt。
- `--seed`：同一个种子在分词器相同的引擎上生成完全相同的 prompt，两条线对比时用同一个值。
- `--gpus`：这条线用的显卡编号，只用于每秒采样利用率、显存和功耗，可以不填。

## 测的是什么

- **冷 prefill**：每条 prompt 是不同的随机文本，开头带一个随机会话号，互相没有共同前缀，所以前缀缓存命中为 0。结果里的 `cache_hit_tokens` 用来确认这一点。
- **prefill 合计（`prefill_tok_s`）**：这一组请求的 prompt token 总数 ÷ 最后一条拿到首字的时间。
- **首字延迟（`ttft`）**：从同时发出到每条请求生成第一个 token 的时间，按先后排序。
- **decode（`decode_tok_s`、`decode_agg_tok_s`）**：只统计所有请求都在生成的那段时间，前者是每路速度的平均，后者是各路之和。token 数取自流式返回里的累计用量，不是数 SSE 分片。
- **KV 占用峰值（`kv_peak`）**：这一格期间 `vllm:kv_cache_usage_perc` 的最大值，每秒采样一次。
- **wave**：KV 缓存放不下全部 prompt 时，一部分请求要等别的请求结束才能开始，“所有请求都在生成”的时间段就不存在。`wave` 是第一批一起跑起来的请求数，`wave_*` 字段只按这一批计算。全部放得下时 `wave` 等于并发数。

详细记录（`detail_*.json`，schema 2）里每个格子还有：

- `requests[]`：每条请求的 prompt 和生成 token 数、首字延迟、结束时间、自己的 decode 速度、token 间隔的 p50/p90/p99/最大值，以及完整的逐 token 时间线。
- `server`：引擎自己的计数在这一格里的变化，包括 prompt 和生成 token 数、前缀缓存查询和命中、抢占次数，以及引擎侧统计的首字、排队、prefill、decode、端到端时间（次数和总秒数）。
- `timeline[]`：每秒一次的 KV 占用、运行和排队的请求数，以及每张显卡的利用率、显存和功耗。

## 结果

| 目录 | 内容 |
|---|---|
| `results/2026-09-30-prefix-on-off/` | 本分支开/关前缀缓存，4 并发，8K/16K/32K/64K，两遍。`summary_on_vs_off.json` 是并排汇总，`result_on.json`、`result_off.json` 是每个格子的数据 |
| `results/2026-09-30-fork-vs-official/` | 本分支对官方 `main@d30469863`，同一模型、同一批 prompt，4 并发，8K/16K/32K/64K，两遍。`summary_fork_vs_official.json` 是并排汇总，`detail_*.json` 是详细记录。**注意**：这次官方用的不是官方的最佳参数，结论已作废，见 README 的“历史记录” |
| `results/2026-10-01-fork-vs-official-best/` | 本分支对官方最新，双方最佳参数（README 实测第 1 组）。文件名前缀：`F1` 本分支开前缀缓存（线上配置），`F2` 本分支关前缀缓存，`O1` 官方最佳参数（关前缀缓存），`O2` 官方开前缀缓存，`OM` 官方开 MTP，`FM` 本分支开 MTP。`*_sweep_c<N>.json` 是 8K 输入、贪心、N 并发；`*_long.json` 是 4 并发 8K–64K、默认采样；`*_l128k.json` 是单条 128K；`*_mtp.json` 是自然 prompt 的 MTP 测试；`*_answers.json` 是答题对照；`summary_nomtp.txt` 是不开 MTP 的汇总 |
| `results/2026-10-01-wheel-1001/` | 发行的 wheel（1001）：`detail_wheel_1001_c4.json` 是和第 1 组同样的压测，`answers_*.json` 是 1001 和 0930 的答题对照 |
| `results/2026-09-30-mtp-fp16-vs-e4m3/` | 开 MTP（k=4）时 FP16 KV 和 E4M3 KV。`c1s`、`c1g` 是单请求 8K（采样、贪心），`c4s2k`、`c4g2k` 是 4 并发 2K（采样、贪心），`c4` 是 4 并发 8K/16K/32K；`answers_*.json` 是输出对照，`kv_scale_report.json` 是标定得到的各层 K、V 最大值和 scale |
| `results/2026-10-03-fork-vs-official-v1.5.1/` | 本分支 1003 对官方 v1.5.1（README 实测第 5、6 组）。文件名前缀是测试时的内部简称，对照见下表。`*_sweep_c*` 是 8K 并发扫描，`*_long` 是 4 并发 8K–64K，`*_l128k` 是 128K，`*_mtp*` 是 MTP 测试，`*_answers.json` 是答题，`engine_*.log` 是引擎日志，`compose.*.yaml` 是各配置的 compose。`O1_HEAD_hang_pyspy.txt` 和 `engine_O1_HEAD_hang.log` 是官方 main 起不来的证据。`REPORT_TABLES.md` 是脚本生成的全部对比表 |

`results/2026-10-03-fork-vs-official-v1.5.1/` 的文件名前缀：

| 前缀 | 配置 |
|---|---|
| `F1` | 本分支 1003，不开 MTP，前缀缓存开（线上配置，24 路，显存利用率 0.90） |
| `F2` | 本分支 1003，不开 MTP，前缀缓存关 |
| `FM` | 本分支 1003，开 MTP，默认（显存利用率 0.87） |
| `FM2` | 本分支 1003，开 MTP，打开 KV 稳态预算（显存利用率 0.93） |
| `V1` | 官方 v1.5.1，不开 MTP，前缀缓存关，显存利用率 0.94 |
| `V2` | 官方 v1.5.1，前缀缓存开，显存利用率 0.94（显存溢出，作废） |
| `V2b` | 官方 v1.5.1，前缀缓存开，显存利用率 0.90 |
| `V2c` | 官方 v1.5.1，前缀缓存开，显存利用率 0.92（显存溢出，只有一部分有效） |
| `VM` | 官方 v1.5.1，开 MTP，显存利用率 0.95，每步 prefill 8192（显存溢出，作废） |
| `VMa` | 官方 v1.5.1，开 MTP，显存利用率 0.92，每步 prefill 4096 |
| `F1W` | 发行的 wheel 的验证（README 第 7 组） |
| `*_sweep_c*`、`*_long`、`*_l128k`、`*_mtp*`、`*_answers` | 并发扫描、4 并发长上下文、128K、MTP 测试、答题 |

开/关前缀缓存那一轮用的是脚本的第一版，测法和指标算法相同，只是没有上面“详细记录”里的那些字段。
