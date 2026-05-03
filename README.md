# MedVision Edge — Offline Chest X-ray Analysis

AI-powered chest X-ray screening for underserved communities, using **Gemma 4 E4B** fine-tuned on 112K+ radiographs. Runs entirely offline on consumer hardware.

## The Problem

- **2.2 billion people** lack access to diagnostic imaging (WHO 2023)
- **1 radiologist per 1M inhabitants** in sub-Saharan Africa
- **740,000 children under 5** die from pneumonia annually, many preventable with early diagnosis
- Cloud-based ML costs $0.03-0.10/image — unaffordable without internet

## Our Solution

MedVision Edge brings AI radiology screening to the 7M+ rural health workers who need it most:

1. **Photograph a chest X-ray** with any smartphone or tablet
2. **Gemma 4 E4B analyzes locally** in 3-5 seconds, no internet required
3. **WHO IMCI protocols** provide evidence-based treatment guidance
4. **140+ languages** supported natively — no translation API needed
5. **Deterministic clinical protocols** — drug dosing and referral urgency with zero hallucination risk

## Results

Validated on two independent benchmarks with real clinical images:

| Pathology | NIH ChestX-ray14 AUC (N=1,103) | CheXpert Gold Standard AUC (N=500) | vs Gemma 4 Baseline |
|-----------|--------------------------------|-------------------------------------|---------------------|
| Cardiomegaly | **0.832** | 0.723 | **+70%** |
| Pleural Effusion | 0.703 | **0.797** | +16% |
| Pulmonary Edema | **0.753** | 0.668 | +9% |
| Consolidation | 0.627 | **0.667** | +5% |
| Pneumonia | 0.617 | — | +19% |

*CheXpert test set uses consensus labels from 5 board-certified radiologists (Stanford).*

### Key achievements

- Fine-tuned model detects pathologies the base model completely misses (Cardiomegaly: base AUC 0.490 vs ours 0.832)
- Sensitivity >0.65 across all pathologies — the model reads images, not just text patterns
- Effusion AUC 0.797 on gold standard benchmark with 95.2% sensitivity
- All results reproducible with provided scripts and public datasets

## Architecture

```
Chest X-ray Image
       |
       v
[Gemma 4 E4B + QLoRA r=64]  ← Fine-tuned vision model (5GB quantized)
       |
       v
[Response Parser]             ← Regex-based YES/NO extraction per pathology
       |
       v
[WHO IMCI Protocol Engine]    ← Deterministic JSON lookup (zero hallucination)
  |         |          |
  v         v          v
Treatment  Dosing   Referral
Protocol   Tables   Urgency
```

## Technical Details

- **Base model**: `unsloth/gemma-4-E4B-it` (6.3B params)
- **Fine-tuning**: QLoRA with Unsloth (r=64, 82M trainable params = 1.3% of total)
- **Training data**: NIH ChestX-ray14 — 112,120 images, 5 pathologies
- **Training**: 2 epochs, lr 1e-4, gradient accumulation 8, data augmentation
- **Hardware**: NVIDIA RTX 5070 Ti 16GB (~43h total GPU time)
- **Deployment**: 5GB GGUF via Ollama (text) or Gradio with transformers (vision)
- **Protocols**: WHO IMCI 2024, ESC Guidelines, BTS Guidelines

## Project Structure

```
medvision-edge/
├── app.py                    # Gradio web interface
├── src/
│   └── protocols.py          # WHO IMCI protocol engine (deterministic)
├── protocols/
│   ├── pneumonia.json        # Clinical guidelines per pathology
│   ├── consolidation.json
│   ├── cardiomegaly.json
│   ├── effusion.json
│   ├── edema.json
│   ├── dosage.json           # Pediatric weight-based dosing tables
│   └── referral.json         # Urgency classification criteria
├── scripts/
│   ├── data_prep.py          # NIH dataset preparation
│   ├── create_conversation_dataset_v4.py  # Training data formatting
│   ├── finetune_gemma4_v4.py # Fine-tuning script (best config)
│   ├── evaluate_model.py     # AUC/sensitivity/specificity evaluation
│   └── eval_chexpert.py      # CheXpert gold standard benchmark
├── results/
│   ├── evaluation_ft.json    # NIH test set metrics
│   ├── chexpert_eval_ft.json # CheXpert benchmark metrics
│   └── threshold_analysis.json
└── requirements.txt
```

## Quick Start

### Gradio Demo (requires GPU)

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:7860
```

### Ollama (text-only, offline)

```bash
ollama run medvision-edge
```

## Training Reproduction

```bash
# 1. Download NIH ChestX-ray14 from Kaggle
# 2. Prepare dataset
python scripts/data_prep.py
python scripts/create_conversation_dataset_v4.py

# 3. Fine-tune (requires 16GB+ VRAM)
python scripts/finetune_gemma4_v4.py

# 4. Evaluate
python scripts/evaluate_model.py --mode ft
python scripts/eval_chexpert.py --mode ft
```

## Target Impact

| Metric | Target Population |
|--------|-------------------|
| Rural health workers | 7M+ globally |
| Health centers without internet | 300,000+ |
| Priority countries | India, Nigeria, Bangladesh, Ethiopia, Indonesia |
| Languages supported | 140+ (native Gemma 4) |

## Limitations

- **AI screening tool only** — not for clinical diagnosis without radiologist confirmation
- Pneumonia detection has high false positive rate (AUC 0.617) due to noisy NLP-extracted labels in training data
- Vision via GGUF/Ollama not yet supported (llama.cpp limitation for multimodal Gemma 4)
- Trained on adult chest X-rays only; pediatric performance not validated

## License

Apache 2.0

## Acknowledgments

- [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) — Training data
- [CheXpert](https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2) — Gold standard evaluation
- [Unsloth](https://github.com/unslothai/unsloth) — Efficient fine-tuning
- [Google Gemma 4](https://ai.google.dev/gemma) — Base model
- WHO IMCI 2024 — Clinical protocols

---

Built for the [Gemma 4 Good Hackathon](https://kaggle.com/competitions/gemma-4-good)
