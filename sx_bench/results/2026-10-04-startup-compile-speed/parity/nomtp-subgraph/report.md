cache_parity lane=nomtp mode=subgraph: FAIL

| arm | healthy s | model load | compile (rank) | dynamo | cache load | graph capture |
|---|---:|---:|---:|---:|---:|---:|
| off | 593.7 | 197.1 | 165.0 | 20.0 | 0.0 | 131.0 |
| off2 | 382.1 | 195.9 | 88.4 | 20.1 | 0.0 | 54.0 |
| cold | 573.2 | 209.3 | 190.0 | 21.0 | 0.0 | 116.0 |
| warm | 367.0 | 220.3 | 57.0 | 20.5 | 1.8 | 36.0 |

| pair | identical prompts | draft counters | verdict |
|---|---:|---|---|
| off vs cold | 6/10 | - | DIFFERENT |
| off vs warm | 7/10 | - | DIFFERENT |
| cold vs warm | 6/10 | - | DIFFERENT |
| off vs off2 | 10/10 | - | ok |
  off vs cold p01_code_lru: first difference at token 43 of 512/512
  off vs cold p02_math: first difference at token 98 of 256/256
  off vs cold p03_zh: first difference at token 160 of 210/218
  off vs cold p08_summary_16k: first difference at token 7 of 107/108
  off vs warm p01_code_lru: first difference at token 9 of 512/512
  off vs warm p03_zh: first difference at token 43 of 210/200
  off vs warm p08_summary_16k: first difference at token 25 of 107/114
  cold vs warm p01_code_lru: first difference at token 9 of 512/512
  cold vs warm p02_math: first difference at token 98 of 256/256
  cold vs warm p03_zh: first difference at token 43 of 218/200
  cold vs warm p08_summary_16k: first difference at token 7 of 108/114

FAILED:
  - off vs cold: 6/10 prompts identical
  - off vs warm: 7/10 prompts identical
  - cold vs warm: 6/10 prompts identical
