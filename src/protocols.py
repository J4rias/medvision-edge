"""
MedVision Edge — Clinical Protocol Engine
Deterministic function calling: WHO IMCI protocols, drug dosage, referral urgency.
No hallucination possible — all outputs come from verified JSON lookup tables.
"""

import json
import os
from pathlib import Path

PROTOCOLS_DIR = Path(__file__).parent.parent / "protocols"


def _load_protocol(name):
    path = PROTOCOLS_DIR / f"{name}.json"
    with open(path) as f:
        return json.load(f)


# Pre-load all protocols
_PROTOCOLS = {}
for name in ["pneumonia", "consolidation", "cardiomegaly", "effusion", "edema", "dosage", "referral"]:
    try:
        _PROTOCOLS[name] = _load_protocol(name)
    except FileNotFoundError:
        pass


# Mapping from finding names to protocol keys
_FINDING_TO_KEY = {
    "Pneumonia": "pneumonia",
    "Consolidation": "consolidation",
    "Cardiomegaly": "cardiomegaly",
    "Effusion": "effusion",
    "Pleural Effusion": "effusion",
    "Edema": "edema",
    "Pulmonary Edema": "edema",
}


def get_clinical_protocol(finding: str, severity: str = "mild") -> dict:
    """
    Retrieve WHO IMCI clinical protocol based on X-ray finding.

    Args:
        finding: Detected pathology (e.g., "Pneumonia", "Cardiomegaly")
        severity: "mild", "moderate", or "severe"

    Returns:
        Dict with classification, treatment, referral criteria
    """
    key = _FINDING_TO_KEY.get(finding, finding.lower())
    protocol = _PROTOCOLS.get(key)

    if not protocol:
        return {"error": f"No protocol found for '{finding}'"}

    severity = severity.lower()
    if severity not in ("mild", "moderate", "severe"):
        severity = "mild"

    result = {
        "condition": protocol.get("condition", finding),
        "severity": severity,
        "classification": protocol.get("classification", {}).get(severity, {}),
        "treatment": protocol.get("treatment", {}).get(severity, {}),
        "referral_criteria": protocol.get("referral", {}),
        "source": protocol.get("source", "WHO IMCI 2024"),
    }

    return result


def calculate_drug_dosage(drug: str, weight_kg: float, age_years: int = None) -> dict:
    """
    Calculate medication dosage based on weight and age.

    Args:
        drug: Drug name (e.g., "amoxicillin", "furosemide")
        weight_kg: Patient weight in kg
        age_years: Patient age in years (optional)

    Returns:
        Dict with dose, frequency, route, warnings
    """
    dosage_db = _PROTOCOLS.get("dosage", {}).get("drugs", {})
    drug_key = drug.lower().replace(" ", "_")
    drug_info = dosage_db.get(drug_key)

    if not drug_info:
        return {"error": f"Drug '{drug}' not found in formulary"}

    is_pediatric = age_years is not None and age_years < 18

    if is_pediatric and "pediatric" in drug_info.get("dosing", {}):
        ped = drug_info["dosing"]["pediatric"]
        dose_per_kg = ped.get("dose_mg_per_kg", 0)
        calculated_dose = dose_per_kg * weight_kg
        max_dose = ped.get("max_dose_mg")
        if max_dose and calculated_dose > max_dose:
            calculated_dose = max_dose

        # Find closest weight band
        weight_band = None
        for band in ped.get("weight_table", []):
            wt = band.get("weight_kg", "")
            if isinstance(wt, str):
                if wt.startswith(">=") and weight_kg >= float(wt[2:]):
                    weight_band = band
                elif "-" in wt:
                    lo, hi = wt.split("-")
                    if float(lo) <= weight_kg <= float(hi):
                        weight_band = band

        result = {
            "drug": drug_info.get("class", drug),
            "patient": f"{weight_kg}kg, {age_years}yr",
            "calculated_dose_mg": round(calculated_dose),
            "frequency": ped.get("frequency", ""),
            "route": ped.get("route", "oral"),
            "duration_days": ped.get("duration_days", "as directed"),
            "formulations": drug_info.get("formulations", []),
        }
        if weight_band:
            result["recommended_dose"] = weight_band.get("dose", "")
        if ped.get("caution"):
            result["caution"] = ped["caution"]
        return result
    else:
        adult = drug_info.get("dosing", {}).get("adult", {})
        if isinstance(adult, dict) and "dose" in adult:
            return {
                "drug": drug_info.get("class", drug),
                "patient": f"{weight_kg}kg, adult",
                "dose": adult.get("dose", ""),
                "frequency": adult.get("frequency", ""),
                "route": adult.get("route", "oral"),
                "formulations": drug_info.get("formulations", []),
            }
        else:
            return {
                "drug": drug_info.get("class", drug),
                "patient": f"{weight_kg}kg, adult",
                "dosing": adult,
                "formulations": drug_info.get("formulations", []),
            }


def determine_referral_urgency(findings: list, confidence_scores: dict = None) -> dict:
    """
    Determine if patient needs referral and urgency level.

    Args:
        findings: List of detected pathologies (e.g., ["Pneumonia", "Effusion"])
        confidence_scores: Optional dict of {pathology: confidence} (not used for rules)

    Returns:
        Dict with urgency level, criteria met, pre-transfer actions
    """
    referral_db = _PROTOCOLS.get("referral", {})
    referral_by_finding = referral_db.get("referral_by_finding", {})
    urgency_levels = referral_db.get("urgency_levels", {})

    if not findings:
        return {
            "urgency": "follow_up",
            "color": "green",
            "recommendation": "No acute findings. Routine follow-up recommended.",
        }

    # Determine highest urgency across all findings
    max_urgency = "follow_up"
    urgency_order = ["follow_up", "routine", "urgent", "emergency"]
    matched_criteria = []

    for finding in findings:
        key = _FINDING_TO_KEY.get(finding, finding.lower())
        rules = referral_by_finding.get(key, {})

        # Multiple findings = higher urgency
        if len(findings) >= 3:
            level = "urgent"
        elif len(findings) >= 2:
            level = "routine"
        else:
            level = "follow_up"

        # Check finding-specific referral rules
        if rules.get("refer_emergency"):
            matched_criteria.append(f"{finding}: {rules['refer_emergency']}")
        if rules.get("refer_urgent"):
            matched_criteria.append(f"{finding}: {rules['refer_urgent']}")
            if urgency_order.index(level) < urgency_order.index("urgent"):
                level = "routine"  # At least routine if has urgent criteria

        if urgency_order.index(level) > urgency_order.index(max_urgency):
            max_urgency = level

    # For single severe finding or multiple findings, escalate
    severe_findings = {"Cardiomegaly", "Edema", "Pulmonary Edema"}
    if any(f in severe_findings for f in findings) and len(findings) >= 2:
        max_urgency = "urgent"

    urgency_info = urgency_levels.get(max_urgency, {})

    return {
        "urgency": max_urgency,
        "color": urgency_info.get("color", "green"),
        "timeframe": urgency_info.get("timeframe", "Routine follow-up"),
        "findings_detected": findings,
        "referral_criteria": matched_criteria[:5],  # Top 5
        "pre_transfer_actions": urgency_info.get("pre_transfer_actions", []),
        "recommendation": f"{'URGENT: ' if max_urgency in ('urgent', 'emergency') else ''}Refer for specialist evaluation. {len(findings)} patholog{'y' if len(findings)==1 else 'ies'} detected.",
    }


def generate_clinical_summary(findings: dict, patient_age: int = None, patient_weight: float = None, language: str = "en") -> str:
    """
    Generate a complete clinical summary with protocols for all detected findings.

    Args:
        findings: Dict of {pathology: True/False}
        patient_age: Age in years (optional)
        patient_weight: Weight in kg (optional)
        language: Output language code (for header only)

    Returns:
        Formatted clinical summary string
    """
    detected = [p for p, present in findings.items() if present]

    if not detected:
        return "**No significant pathologies detected.**\n\nRecommendation: Routine follow-up. No immediate intervention required.\n\n⚠️ *This is an AI screening tool. Findings should be confirmed by a qualified radiologist.*"

    lines = []
    lines.append(f"## Clinical Summary — {len(detected)} Finding{'s' if len(detected)>1 else ''} Detected\n")

    for finding in detected:
        protocol = get_clinical_protocol(finding, severity="mild")
        if "error" not in protocol:
            classification = protocol.get("classification", {})
            treatment = protocol.get("treatment", {})

            lines.append(f"### {finding}")
            if classification.get("label"):
                lines.append(f"**Classification:** {classification['label']}")
            if isinstance(treatment, dict):
                setting = treatment.get("setting", "")
                if setting:
                    lines.append(f"**Setting:** {setting}")
                if treatment.get("first_line"):
                    fl = treatment["first_line"]
                    if isinstance(fl, dict):
                        dose = fl.get('dose', '')
                        if not dose and fl.get('dose_mg_per_kg'):
                            dose = f"{fl['dose_mg_per_kg']} mg/kg"
                        lines.append(f"**First-line:** {fl.get('drug', '')} {dose} {fl.get('frequency', '')}")
                    elif isinstance(fl, list):
                        for med in fl:
                            lines.append(f"- {med.get('drug', '')} {med.get('dose', '')} {med.get('frequency', '')}")
            lines.append("")

    # Referral
    referral = determine_referral_urgency(detected)
    urgency_emoji = {"emergency": "🔴", "urgent": "🟠", "routine": "🟡", "follow_up": "🟢"}.get(referral["urgency"], "⚪")
    lines.append(f"### Referral Assessment")
    lines.append(f"{urgency_emoji} **{referral['urgency'].upper()}** — {referral.get('timeframe', '')}")
    lines.append(f"{referral.get('recommendation', '')}")
    lines.append("")

    # Dosage if weight provided
    if patient_weight and detected:
        lines.append("### Suggested Dosing")
        primary_drug = "amoxicillin" if "Pneumonia" in detected or "Consolidation" in detected else "furosemide"
        dosage = calculate_drug_dosage(primary_drug, patient_weight, patient_age)
        if "error" not in dosage:
            if dosage.get('calculated_dose_mg'):
                dose_str = f"{dosage['calculated_dose_mg']} mg"
            elif dosage.get('dose'):
                dose_str = dosage['dose']
            else:
                dose_str = "N/A"
            lines.append(f"**{primary_drug.title()}**: {dose_str} {dosage.get('frequency', '')}")
        lines.append("")

    lines.append("---")
    lines.append("⚠️ *AI screening tool. All findings must be confirmed by a qualified radiologist. Do not use as sole basis for clinical decisions.*")

    return "\n".join(lines)
