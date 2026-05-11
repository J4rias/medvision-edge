"""
MedVision Edge — Offline Chest X-ray Analysis
Gradio demo for Gemma 4 E4B fine-tuned on 112K+ chest X-rays.

Deploy: HuggingFace Space (ZeroGPU) or local with `python app.py`
"""

import os
import re
import json
import torch
import gradio as gr
import spaces
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# ── Config ──────────────────────────────────────────────────────
MODEL_PATH = os.environ.get(
    "MEDVISION_MODEL",
    os.path.expanduser("~/ml-projects/medvision/model_output/final_model"),
)
IS_SPACES = os.environ.get("SPACE_ID") is not None
LOAD_IN_4BIT = os.environ.get("MEDVISION_4BIT", "true").lower() == "true" and not IS_SPACES
PATHOLOGIES = ["Pneumonia", "Consolidation", "Cardiomegaly", "Effusion", "Edema"]

LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "Swahili": "sw",
    "Hindi": "hi",
    "Arabic": "ar",
    "Portuguese": "pt",
    "Bengali": "bn",
    "Chinese": "zh",
    "Russian": "ru",
}

EVAL_PROMPT = (
    "Analyze this chest X-ray. For each condition, state YES or NO, "
    "then describe your findings.\n"
    "- Pneumonia\n- Consolidation\n- Cardiomegaly\n- Effusion\n- Edema"
)

TRANSLATE_PROMPT = (
    "Translate the following medical report to {language}. "
    "Keep medical terminology accurate. Translate only, do not add commentary.\n\n{text}"
)

# ── Model loading ─────────────────────────────────────────────
# ZeroGPU: load on CPU at module level, ZeroGPU moves to GPU automatically
# Local: load with device_map="auto" (optionally 4-bit)
print(f"Loading model from {MODEL_PATH} (spaces={IS_SPACES}, 4bit={LOAD_IN_4BIT})...")

if LOAD_IN_4BIT:
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        ),
        device_map="auto",
    )
else:
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
    )

model.eval()
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("Model loaded.")


def parse_response(response: str) -> dict:
    """Extract YES/NO predictions from model response."""
    predictions = {}
    clean = response.replace("**", "").replace("__", "").replace("*", "").replace("_", "")

    for pathology in PATHOLOGIES:
        p1 = rf"(?:[-•*]\s*)?(?:\d+\.\s*)?{pathology}\s*[:—–-]\s*(YES|NO|PRESENT|ABSENT)\b"
        p2 = rf"{pathology}\s*\(\s*(YES|NO|PRESENT|ABSENT)\s*\)"
        p3 = rf"{pathology}.{{0,60}}?\b(YES|NO|PRESENT|ABSENT)\b"
        p4_pos = rf"{pathology}.{{0,80}}?(present|detected|observed|found|identified|consistent with|indicative|suggesting|evidence of)"
        p4_neg = rf"{pathology}.{{0,80}}?(absent|not present|no evidence|not detected|not observed|unremarkable|clear|normal)"
        p4_neg_rev = rf"(no evidence of|no |absent|not |unremarkable|clear ).{{0,40}}?{pathology}"

        matched = False
        for pattern in [p1, p2, p3]:
            match = re.search(pattern, clean, re.IGNORECASE)
            if match:
                val = match.group(1).upper()
                predictions[pathology] = val in ("YES", "PRESENT")
                matched = True
                break

        if not matched:
            if re.search(p4_pos, clean, re.IGNORECASE):
                predictions[pathology] = True
            elif re.search(p4_neg, clean, re.IGNORECASE) or re.search(p4_neg_rev, clean, re.IGNORECASE):
                predictions[pathology] = False
            else:
                predictions[pathology] = False  # Default to negative

    return predictions


def run_inference(image: Image.Image, prompt: str) -> str:
    """Run model inference on a single image."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    print(f"Inference on device: {model.device}, dtype: {model.dtype}")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
        )

    decoded = processor.batch_decode(output, skip_special_tokens=True)[0]
    print(f"Raw decoded (first 500 chars): {decoded[:500]}")
    # Extract only the assistant's response (handle multiple Gemma 4 marker formats)
    for marker in ["<|turn>model", "<start_of_turn>model", "model\n"]:
        if marker in decoded:
            decoded = decoded.split(marker)[-1]
            break
    for marker in ["<turn|>", "<end_of_turn>", "<eos>"]:
        if marker in decoded:
            decoded = decoded.split(marker)[0]
            break
    return decoded.strip()


@spaces.GPU
def analyze_xray(image, language, patient_age, patient_weight):
    """Main analysis function for Gradio interface."""
    if image is None:
        return "Please upload a chest X-ray image.", "{}", "", ""

    # Move model to GPU if not already there (ZeroGPU assigns GPU when entering @spaces.GPU)
    if IS_SPACES and not next(model.parameters()).is_cuda:
        model.to("cuda")
        print(f"Model moved to cuda (first request)")
    elif IS_SPACES:
        print(f"Model already on cuda (cached)")

    # Convert to PIL if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert("RGB")
    else:
        image = image.convert("RGB")

    # 1. Run X-ray analysis
    raw_response = run_inference(image, EVAL_PROMPT)

    # 2. Parse findings
    findings = parse_response(raw_response)
    detected = [p for p, v in findings.items() if v]

    # 3. Generate findings summary
    findings_display = {}
    for p in PATHOLOGIES:
        status = "DETECTED" if findings.get(p) else "Not detected"
        emoji = "🔴" if findings.get(p) else "🟢"
        findings_display[f"{emoji} {p}"] = status

    # 4. Generate clinical protocol
    from src.protocols import generate_clinical_summary
    age = int(patient_age) if patient_age else None
    weight = float(patient_weight) if patient_weight else None
    clinical_summary = generate_clinical_summary(findings, age, weight)

    # 5. Translate if needed
    lang_name = language if language else "English"
    lang_code = LANGUAGES.get(lang_name, "en")

    translated = ""
    if lang_code != "en" and detected:
        translate_prompt = TRANSLATE_PROMPT.format(
            language=lang_name,
            text=clinical_summary,
        )
        # Use the model to translate (Gemma 4 supports 140+ languages natively)
        messages = [
            {"role": "user", "content": [{"type": "text", "text": translate_prompt}]},
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=1024, temperature=0.3, do_sample=True)
        translated = processor.batch_decode(output, skip_special_tokens=True)[0]
        # Strip model response from full output (handles Gemma 4 turn markers)
        # Try multiple marker formats: Gemma 4 uses <|turn>model / <turn|>
        for marker in ["<|turn>model", "<start_of_turn>model", "model\n"]:
            if marker in translated:
                translated = translated.split(marker)[-1]
                break
        for marker in ["<turn|>", "<end_of_turn>", "<eos>"]:
            if marker in translated:
                translated = translated.split(marker)[0]
                break
        # Fallback: if prompt text leaked through, strip it
        if translate_prompt in translated:
            translated = translated.split(translate_prompt)[-1]
        translated = translated.strip()

    findings_text = "\n".join(f"{k}: {v}" for k, v in findings_display.items())
    return raw_response, findings_text, clinical_summary, translated


# ── Gradio UI ───────────────────────────────────────────────────
DESCRIPTION = """
# MedVision Edge — Offline Chest X-ray Analysis

**AI-powered chest X-ray screening** using Gemma 4 E4B fine-tuned on 112K+ radiographs.

Detects 5 pathologies: **Pneumonia, Consolidation, Cardiomegaly, Pleural Effusion, Pulmonary Edema**

### Validated on two independent benchmarks

| Pathology | NIH ChestX-ray14 AUC | CheXpert Gold Standard AUC | vs Baseline |
|-----------|----------------------|---------------------------|-------------|
| Cardiomegaly | **0.832** | 0.723 | +70% |
| Pleural Effusion | 0.703 | **0.797** | +16% |
| Pulmonary Edema | **0.753** | 0.668 | +9% |
| Consolidation | 0.627 | **0.667** | +5% |

*CheXpert test set: 500 images, 5 board-certified radiologist consensus (Stanford)*

> **AI screening tool only.** Not for clinical diagnosis. All findings must be confirmed by a qualified radiologist.
"""

ARTICLE = """
### About MedVision Edge
- **Model**: Gemma 4 E4B-it, fine-tuned with Unsloth QLoRA (r=64, 82M trainable params)
- **Training data**: NIH ChestX-ray14 (112K images, 5 pathologies, 2 epochs)
- **Evaluation**: NIH test set (1,103 images) + CheXpert gold standard (500 images, 5 radiologist consensus)
- **Protocols**: WHO IMCI 2024 clinical guidelines (deterministic function calling, zero hallucination)
- **Languages**: 140+ supported natively by Gemma 4
- **Deployment**: Runs offline on consumer GPU (5GB GGUF via Ollama) or this Gradio demo
- **GPU used**: ~43h on NVIDIA RTX 5070 Ti 16GB (training + evaluation)

Built for the [Gemma 4 Good Hackathon](https://kaggle.com/competitions/gemma-4-good) | Apache 2.0
"""

CUSTOM_CSS = """
/* Replace broken Gradio 6.x spinner SVG with a CSS spinner */
.wrap svg.svelte-1vhirvf {
    display: none !important;
}
.wrap.svelte-1uj8rng:not(.hide)::after {
    content: "";
    width: 28px;
    height: 28px;
    border: 3px solid var(--body-text-color-subdued, #ccc);
    border-top-color: var(--color-accent, #FF7C00);
    border-radius: 50%;
    animation: medvision-spin 0.8s linear infinite;
}
@keyframes medvision-spin {
    to { transform: rotate(360deg); }
}
/* Responsive expected results cards */
.expected-results .card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 8px;
    margin: 8px 0;
}
.expected-results .card {
    background: var(--block-background-fill);
    border: 1px solid var(--block-border-color);
    border-radius: 8px;
    padding: 10px;
    font-size: 0.85em;
}
.expected-results .card strong {
    display: block;
    margin-bottom: 4px;
}
"""

EXTRA_HEAD = """
<script>
// Force rear camera: override getUserMedia to prefer environment facingMode
(function() {
    const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = function(constraints) {
        if (constraints && constraints.video && typeof constraints.video === 'object') {
            if (!constraints.video.deviceId) {
                constraints.video.facingMode = {ideal: 'environment'};
            }
        } else if (constraints && constraints.video === true) {
            constraints.video = {facingMode: {ideal: 'environment'}};
        }
        return original(constraints);
    };
})();
</script>
"""

demo = gr.Blocks(
    title="MedVision Edge",
    theme=gr.themes.Soft(),
    css=CUSTOM_CSS,
    head='<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🏥</text></svg>">' + EXTRA_HEAD,
)

with demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload Chest X-ray",
                sources=["upload", "webcam"],
                elem_id="xray-input",
                webcam_options=gr.WebcamOptions(
                    mirror=False,
                    constraints={"facingMode": {"ideal": "environment"}},
                ),
            )
            language_input = gr.Dropdown(
                choices=list(LANGUAGES.keys()),
                value="English",
                label="Output Language",
            )
            with gr.Row():
                age_input = gr.Number(label="Patient Age (years)", precision=0, minimum=0, maximum=120)
                weight_input = gr.Number(label="Patient Weight (kg)", precision=1, minimum=0, maximum=300)
            analyze_btn = gr.Button("🔍 Analyze X-ray", variant="primary", size="lg")

        with gr.Column(scale=2):
            findings_output = gr.Textbox(label="Findings", lines=6)
            raw_output = gr.Textbox(label="Model Analysis (raw)", lines=10)
            protocol_output = gr.Markdown(label="Clinical Protocol (WHO IMCI)")
            translated_output = gr.Markdown(label="Translated Report")

    analyze_btn.click(
        fn=analyze_xray,
        inputs=[image_input, language_input, age_input, weight_input],
        outputs=[raw_output, findings_output, protocol_output, translated_output],
    )

    gr.HTML("""
<div class="expected-results">
<h3>Example chest X-rays — Expected results</h3>
<p style="font-size:0.85em;color:var(--body-text-color-subdued);">CheXpert test set, radiologist-verified</p>
<div class="card-grid">
  <div class="card"><strong>1. Normal</strong>All clear — no pathology detected</div>
  <div class="card"><strong>2. Cardiomegaly</strong>Cardiomegaly: DETECTED</div>
  <div class="card"><strong>3. Effusion</strong>Effusion: DETECTED</div>
  <div class="card"><strong>4. Edema</strong>Edema: DETECTED</div>
  <div class="card"><strong>5. Multiple</strong>Pneumonia: DETECTED<br>Consolidation: DETECTED<br>Effusion: DETECTED</div>
</div>
</div>
""")

    gr.Examples(
        examples=[
            ["examples/example1_normal.jpg", "English", 35, 70],
            ["examples/example2_cardiomegaly.jpg", "English", 62, 85],
            ["examples/example3_effusion.jpg", "English", 55, 75],
            ["examples/example4_edema.jpg", "Spanish", 70, 80],
            ["examples/example5_multiple.jpg", "English", 58, 72],
        ],
        inputs=[image_input, language_input, age_input, weight_input],
        label="Click an example to load it",
    )

    gr.Markdown(ARTICLE)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
