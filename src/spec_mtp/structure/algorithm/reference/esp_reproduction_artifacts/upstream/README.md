# ESP Reproduction Artifacts

All code, results, logs, and reports for the reproduction of **ESP (Embedding-Space Probing)** from arXiv 2603.17942 (ICML 2026).

## Contents

```
├── code/
│   ├── esp_v5.py          # V5 implementation (2-pass, temp=0.01)
│   ├── esp_v6.py          # V6 implementation (single-pass tree attention, temp=1.0)
│   ├── repro_lemma31.py   # Lemma 3.1 numerical verification
│   └── run.sh             # HF job runner
├── results/
│   ├── esp_v5_results.json    # V5 results (LLaMA3.2-3B)
│   ├── esp_v6_results.json    # V6 results (all 3 models)
│   └── lemma31_results.json   # Lemma 3.1 audit (200 trials, 0 violations)
├── poster/
│   ├── esp_reproduction_poster.pdf  # Research poster (PDF)
│   └── esp_reproduction_poster.png  # Research poster (PNG)
├── logbook/
│   ├── index.md
│   ├── executive-summary/page.md
│   ├── claim-1-*/page.md through claim-5-*/page.md
│   └── conclusion/page.md
└── README.md
```

## Key Results

| Model | BC | τ (Ours) | τ (Paper) |
|---|---|---|---|
| LLaMA3.2-3B | 30 | 1.45 | 1.56 |
| LLaMA3.2-3B | 60 | 1.47 | 1.63 |
| LLaMA3.1-8B | 30 | 2.00 | 1.63 |
| Qwen3-8B | 30 | 2.91 | 1.74 |

## Reproduction Status

- ✅ Lemma 3.1: 0 violations (verified)
- ⚠️ Table 1 τ: τ > 1 confirmed, values within 7-80% of paper
- ❌ 12% quality: Not tested (needs SpecBench)
- ❌ 15-19% throughput: Not reproduced (needs KV cache + FlashAttention)
- ⚠️ Cosine ~0.45 vs ~0.35: Pattern confirmed, values vary by model

## Paper

arXiv 2603.17942 — *Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing*
ICML 2026, OpenReview id `mKj4fO0BU4`
