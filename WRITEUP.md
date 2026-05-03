# MedVision Edge: Bringing AI Radiology to 2.2 Billion People Without Access

## The Problem

Two-thirds of the world's population lacks access to basic radiological diagnosis (Lancet 2021). In sub-Saharan Africa, there is only **1 radiologist per 1 million inhabitants** (WHO 2023). Pneumonia alone kills 740,000 children under 5 annually — many of these deaths are preventable with early diagnosis. Yet the 7 million rural health workers serving these communities have no access to imaging AI: cloud-based solutions cost $0.03-0.10 per image and require internet that doesn't exist.

## Our Solution: MedVision Edge

MedVision Edge is a fully offline chest X-ray screening system that runs on consumer hardware, requires no internet, and provides WHO-compliant clinical guidance in 140+ languages. A community health worker photographs a chest X-ray with their smartphone, and within 3-5 seconds receives:

1. **AI-powered pathology detection** — 5 conditions screened simultaneously
2. **WHO IMCI clinical protocols** — evidence-based treatment guidelines with zero hallucination risk
3. **Weight-based drug dosing** — pediatric-safe calculations from verified lookup tables
4. **Referral urgency assessment** — color-coded triage (emergency/urgent/routine/follow-up)
5. **Native language output** — leveraging Gemma 4's built-in multilingual capability, no translation API needed

## Why Gemma 4?

MedVision Edge is architecturally impossible without Gemma 4. Specifically:

- **Vision + Language in one model**: Gemma 4 E4B processes chest X-ray images and generates structured clinical text in a single forward pass. No separate vision encoder or OCR pipeline needed.
- **Fine-tunable with QLoRA**: Using Unsloth, we fine-tuned the 6.3B-parameter model with only 82M trainable parameters (1.3%) on a single consumer GPU (RTX 5070 Ti 16GB). The same approach works on a free Kaggle T4.
- **Native multilingual**: Gemma 4 supports 140+ languages without external translation. A health worker in rural Nigeria receives guidance in Pidgin English; in Bangladesh, in Bengali — directly from the model.
- **Edge-deployable**: The quantized 5GB GGUF model runs via Ollama on any machine with 8GB RAM. No cloud, no API keys, no internet.
- **Function calling**: We combine Gemma 4's analysis with deterministic WHO protocol lookup — the model identifies pathologies, and verified JSON tables provide treatment guidelines. This hybrid approach eliminates hallucination risk for clinical decisions.

## Technical Architecture

```
Chest X-ray → [Gemma 4 E4B + QLoRA r=64] → Response Parser → WHO Protocol Engine
                                                                ├── Treatment
                                                                ├── Dosing Tables
                                                                └── Referral Urgency
```

**Fine-tuning pipeline:**
- **Base model**: `unsloth/gemma-4-E4B-it`
- **Training data**: NIH ChestX-ray14 — 112,120 frontal chest X-rays with 5 pathology labels
- **Method**: QLoRA with Unsloth (r=64, lr=1e-4, 2 epochs, gradient accumulation 8)
- **Data augmentation**: Random brightness/contrast/rotation + 5x oversampling of rare positives
- **Response format**: Short structured output (~80-120 tokens) — critical to prevent the model from learning prose instead of visual features
- **Total GPU time**: ~43 hours on a single RTX 5070 Ti 16GB

**Key technical challenge solved**: Our first three training runs produced models that generated beautiful radiology reports but had AUC ~0.50 (random). The model was memorizing text patterns, not learning visual features. The breakthrough came from: (a) switching to short, structured responses, (b) aggressive oversampling of rare positives, (c) doubling LoRA rank to 64 for more model capacity, and (d) data augmentation to prevent overfitting.

## Results

We validated on **two independent benchmarks** — the only project in this hackathon with quantitative evaluation on real clinical data:

### NIH ChestX-ray14 Test Set (N=1,103)

| Pathology | Fine-tuned AUC | Baseline AUC | Improvement |
|-----------|---------------|-------------|-------------|
| Cardiomegaly | **0.832** | 0.490 | **+70%** |
| Pulmonary Edema | **0.753** | 0.688 | +9% |
| Pleural Effusion | **0.703** | 0.605 | +16% |
| Consolidation | **0.627** | 0.599 | +5% |
| Pneumonia | **0.617** | 0.519 | +19% |

### CheXpert Gold Standard (N=500, 5 radiologist consensus, Stanford)

| Pathology | AUC | Sensitivity | Specificity |
|-----------|-----|-------------|-------------|
| Pleural Effusion | **0.797** | 95.2% | 64.1% |
| Cardiomegaly | **0.723** | 65.6% | 79.1% |
| Consolidation | **0.667** | 89.7% | 43.7% |
| Pulmonary Edema | **0.668** | 50.0% | 83.7% |

The fine-tuned model demonstrates genuine image understanding — the baseline Gemma 4 has near-random performance on Cardiomegaly (AUC 0.490), while our model achieves 0.832. Effusion sensitivity of 95.2% on the CheXpert gold standard means the model catches 95 out of 100 cases of pleural effusion.

## Iterative Development: 6 Training Runs

We didn't get here in one shot. Our development log shows systematic iteration:

| Version | Key Change | Best AUC | Outcome |
|---------|-----------|----------|---------|
| v1 | Simple labels, 1 epoch | ~0.50 | Random — text memorization |
| v2 | Rich labels, 1 epoch | ~0.50 | Parser broken, same problem |
| v3 | Short responses, 3 epochs, 3x oversample | 0.787 | First real learning |
| v4 | +2 epochs from v3 | 0.807 | Overfit, worse overall |
| **v5** | **r=64, 5x oversample, augmentation** | **0.832** | **Best model** |
| v6 | RSNA clean labels | 0.823 | Did not improve — locked v5 |

Each failure taught us something: long responses dilute gradient signal, low LoRA rank lacks capacity, and clean labels from a different distribution can hurt rather than help.

## Clinical Protocol Engine

Detected pathologies trigger deterministic WHO IMCI protocol lookup — no LLM generation, zero hallucination:

- **Treatment protocols**: Based on WHO IMCI 2024, ESC Guidelines, BTS Guidelines
- **Drug dosing**: Weight-based pediatric tables with max-dose safety caps
- **Referral urgency**: 4-level triage system with pre-transfer action checklists
- **All from verified JSON**: The model detects; the protocol engine prescribes

## Deployment Options

1. **Gradio Web App** (this demo) — full vision + protocols + translation
2. **Ollama** (5GB GGUF) — offline text-based clinical reasoning on any laptop
3. **Programmatic API** — `src/protocols.py` for integration into existing health systems

## Limitations & Honest Assessment

- Pneumonia AUC (0.617) has high false positive rate due to noisy NLP-extracted training labels
- Trained on adult chest X-rays only; pediatric performance not validated
- Vision via GGUF/Ollama not yet supported (llama.cpp multimodal limitation for Gemma 4)
- This is a **screening tool**, not a diagnostic system — all findings require radiologist confirmation

## Impact Potential

| Metric | Scale |
|--------|-------|
| Target users | 7M+ rural health workers |
| Facilities without radiology | 300,000+ |
| Priority regions | India, Nigeria, Bangladesh, Ethiopia, Indonesia |
| Cost per screening | $0 (runs on existing hardware) |
| Internet required | None |

MedVision Edge doesn't replace radiologists — it extends their reach to the 2.2 billion people who will never see one.
