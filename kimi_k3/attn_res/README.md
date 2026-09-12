# attn_res — Attention Residuals (Kimi, arXiv:2603.15031)

| file | what |
|---|---|
| `SURVEY.md` | The main document: the AttnRes idea (following [Su Jianlin's blog](https://spaces.ac.cn/archives/11664)), the unified depth-mixing-matrix view (residual / Highway / HC / mHC / AttnRes), the two-phase inference schedule and its I/O accounting, how SGLang supports this family today (DeepSeek-V4 mHC anatomy) and what an AttnRes integration would need, plus measured B200 costs. |
| `bench_attn_res.py` | Per-layer latency of the residual *mechanism only* (merge + input RMSNorm, excluding the layer function): plain residual vs mHC(m=4, torch path) vs Block-AttnRes naïve vs Block-AttnRes two-phase vs Full-AttnRes naïve. Includes an fp32 exactness check of the online-softmax merge. |
| `results/bench_attn_res.csv` | Raw sweep output (T ∈ {1, 16, 256, 4096, 16384}, d=7168, N=10, S=6, L=54). |

Run:

```bash
python3 attn_res/bench_attn_res.py            # defaults reproduce results/
python3 attn_res/bench_attn_res.py --tokens 1 4096 --d 4096
```
