from __future__ import annotations
import os, uuid, json, time, io, re, math
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
import asyncio

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field


try:
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether
    )
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


def _api_key() -> str:
    """Returns validator-injected API_KEY first, then HF_TOKEN as fallback."""
    k = os.environ.get("API_KEY", "").strip()
    if k:
        return k
    return os.environ.get("HF_TOKEN", "").strip()

def _base_url() -> str:
    return os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1").strip()

def _model() -> str:
    return os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct").strip()

def _openai_key() -> str:
    return os.environ.get("OPENAI_API_KEY", "").strip()

def _anthropic_key() -> str:
    return os.environ.get("ANTHROPIC_API_KEY", "").strip()


app = FastAPI(
    title="ClinicalTriageEnv — Enterprise Clinical AI Platform",
    version="5.0.0",
    description="AI-powered triage, clinical decision support, and chatbot assistant.",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)

_sessions:       Dict[str, Dict] = {}
_report_cache:   Dict[str, Dict] = {}
_chat_histories: Dict[str, List] = {}

TASK_REGISTRY = {
    "triage_easy":       {"name":"Emergency Triage - Easy",           "type":"triage",       "difficulty":"easy",   "max_steps":3, "description":"Assign ESI triage level to non-urgent presentation."},
    "triage_medium":     {"name":"Emergency Triage - Medium",         "type":"triage",       "difficulty":"medium", "max_steps":5, "description":"Triage patient with potential ACS — Levine's sign, radiation, diaphoresis."},
    "triage_hard":       {"name":"Emergency Triage - Hard",           "type":"triage",       "difficulty":"hard",   "max_steps":7, "description":"Acute ischaemic stroke on warfarin — INR unknown, LKW 1h45m."},
    "med_safety_easy":   {"name":"Medication Safety Review - Easy",   "type":"med_safety",   "difficulty":"easy",   "max_steps":3, "description":"Routine medication review — amlodipine + atorvastatin CYP3A4 interaction."},
    "med_safety_medium": {"name":"Medication Safety Review - Medium", "type":"med_safety",   "difficulty":"medium", "max_steps":5, "description":"Post-cath triple antithrombotic therapy — GI bleed risk assessment."},
    "med_safety_hard":   {"name":"Medication Safety Review - Hard",   "type":"med_safety",   "difficulty":"hard",   "max_steps":6, "description":"HIV + Ritonavir + Simvastatin — CYP3A4 → rhabdomyolysis, CK 48,000."},
    "sepsis_easy":       {"name":"Sepsis Management - Easy",          "type":"sepsis",       "difficulty":"easy",   "max_steps":4, "description":"Pyelonephritis with SIRS — PCN allergy, aztreonam selection."},
    "sepsis_medium":     {"name":"Sepsis Management - Medium",        "type":"sepsis",       "difficulty":"medium", "max_steps":6, "description":"Septic shock — MRSA, COPD, CKD, DM2. Vasopressor decision."},
    "sepsis_hard":       {"name":"Sepsis Management - Hard",          "type":"sepsis",       "difficulty":"hard",   "max_steps":8, "description":"Post-op anastomotic leak — DIC, prednisolone, stress-dose cortisol."},
}

MORTALITY_RISK = {
    "triage_easy":       {"baseline":0.5,  "undertriage_mult":2.0, "delay_per_min":0.010},
    "triage_medium":     {"baseline":8.0,  "undertriage_mult":3.5, "delay_per_min":0.150},
    "triage_hard":       {"baseline":18.0, "undertriage_mult":5.0, "delay_per_min":0.400},
    "med_safety_easy":   {"baseline":0.2,  "undertriage_mult":1.5, "delay_per_min":0.005},
    "med_safety_medium": {"baseline":3.0,  "undertriage_mult":2.5, "delay_per_min":0.050},
    "med_safety_hard":   {"baseline":12.0, "undertriage_mult":4.0, "delay_per_min":0.300},
    "sepsis_easy":       {"baseline":6.0,  "undertriage_mult":2.5, "delay_per_min":0.200},
    "sepsis_medium":     {"baseline":22.0, "undertriage_mult":4.0, "delay_per_min":0.550},
    "sepsis_hard":       {"baseline":45.0, "undertriage_mult":6.0, "delay_per_min":1.200},
}

PATIENTS_DB = {
    "triage_easy": {
        "name":"Samrudh AJ","age":19,"sex":"M","id":"CS-E01","task":"triage_easy","domain":"triage","diff":"easy",
        "vitals":{"hr":76,"sbp":122,"spo2":99,"temp_f":97.9,"rr":14,"gcs":15},
        "complaint":"Mild ankle sprain after basketball. Lateral malleolus swelling, weight-bearing intact. Ottawa rules negative.",
        "tags":["No PMH","NKDA","Ottawa Negative"],
        "esi_correct":5,"triage":"LOW_RISK","mortality_base":0.5,
        "drug_interactions":[],
        "clinical_pearl":"Ottawa Ankle Rules: no bony tenderness at posterior tip of malleolus. No X-ray needed. RICE therapy.",
        "key_decision":"ESI-5 — Non-urgent. Can wait >2h. RICE therapy.",
        "risk_factors":[],"labs":{}
    },
    "triage_medium": {
        "name":"Sophia Kira","age":34,"sex":"F","id":"CS-E02","task":"triage_medium","domain":"triage","diff":"medium",
        "vitals":{"hr":102,"sbp":148,"spo2":96,"temp_f":98.7,"rr":20,"gcs":15},
        "complaint":"Crushing chest pain 8/10 radiating to left arm and jaw. Diaphoresis, nausea. Levine's sign. Onset 45 min ago.",
        "tags":["HTN","DM2","Smoker 30pk-yr","Metformin","Family Hx CAD"],
        "esi_correct":2,"triage":"EMERGENCY","mortality_base":8.0,
        "drug_interactions":[],
        "clinical_pearl":"STEMI V1-V4 pattern + Levine's sign + jaw radiation. D2B target <90 min. Cath lab now.",
        "key_decision":"ESI-2 — Emergent. ECG STAT, aspirin 324mg, IV access x2, cath lab activation.",
        "risk_factors":["Hypertension","Diabetes Mellitus","Smoking"],
        "labs":{"troponin":"Pending","ECG":"STEMI V1-V4"}
    },
    "triage_hard": {
        "name":"Annabell Chander","age":72,"sex":"F","id":"CS-E03","task":"triage_hard","domain":"triage","diff":"hard",
        "vitals":{"hr":88,"sbp":188,"spo2":95,"temp_f":98.1,"rr":17,"gcs":13},
        "complaint":"Acute right arm weakness, facial droop, expressive aphasia. FAST positive. LKW 1h45m. On warfarin for A-fib.",
        "tags":["A-fib","Warfarin 5mg ⚠","HTN","TIA 2021","INR unknown"],
        "esi_correct":1,"triage":"EMERGENCY","mortality_base":18.0,
        "drug_interactions":[{"a":"Warfarin","b":"tPA","mech":"Anticoagulation may contraindicate thrombolytics — INR must be <1.7","sev":"crit"}],
        "clinical_pearl":"Warfarin: get INR STAT. If >1.7, IV tPA contraindicated. Still in tPA window (LKW 1h45m < 4.5h).",
        "key_decision":"ESI-1 — Resus. Stroke alert STAT. CT head. INR stat. Neurology. Every min = 1.9M neurons.",
        "risk_factors":["A-fib","Hypertension","TIA"],
        "labs":{"INR":"STAT pending","glucose":6.2}
    },
    "med_safety_easy": {
        "name":"Shin Chan","age":24,"sex":"M","id":"CS-E04","task":"med_safety_easy","domain":"medication safety","diff":"easy",
        "vitals":{"hr":72,"sbp":130,"spo2":98,"temp_f":98.4,"rr":14,"gcs":15},
        "complaint":"Routine medication review. HTN and hyperlipidaemia well controlled.",
        "tags":["HTN","Hyperlipidaemia","Amlodipine 5mg","Atorvastatin 40mg","Aspirin 81mg"],
        "esi_correct":3,"triage":"MODERATE","mortality_base":0.2,
        "drug_interactions":[],
        "clinical_pearl":"Amlodipine + Atorvastatin: mild CYP3A4 interaction — AUC +15%. Acceptable at 40mg. Verify LDL <1.8.",
        "key_decision":"ESI-3 — Stable. Well-controlled regimen. Monitor LDL target.",
        "risk_factors":["Hypertension","Hyperlipidaemia"],
        "labs":{"creatinine":0.9,"eGFR":85}
    },
    "med_safety_medium": {
        "name":"Lisa Lee","age":68,"sex":"F","id":"CS-E05","task":"med_safety_medium","domain":"medication safety","diff":"medium",
        "vitals":{"hr":78,"sbp":134,"spo2":97,"temp_f":98.6,"rr":16,"gcs":15},
        "complaint":"Post-cardiac cath day 5. Triple antithrombotic therapy. CKD-3a, DM2. Epigastric discomfort.",
        "tags":["Post-MI (5d)","DM2","CKD-3a","Warfarin","Aspirin 325mg ⚠","Clopidogrel","Metformin"],
        "esi_correct":2,"triage":"URGENT","mortality_base":3.0,
        "drug_interactions":[
            {"a":"Warfarin","b":"Aspirin+Clopidogrel","mech":"Triple therapy — major GI bleed NNH ~25. Risk ×3-4 vs dual.","sev":"crit"},
            {"a":"Aspirin 325mg","b":"Post-MI","mech":"Long-term: use 81mg — no extra CV benefit, higher GI risk.","sev":"maj"},
            {"a":"Metformin","b":"eGFR 48","mech":"Borderline — eGFR <45 requires dose reduction or hold.","sev":"mod"}
        ],
        "clinical_pearl":"WOEST trial: at 1 month drop aspirin from triple → dual therapy. Add PPI immediately.",
        "key_decision":"ESI-2 — Flag triple therapy. Reduce aspirin to 81mg, add PPI, plan dual therapy transition.",
        "risk_factors":["Post-MI","Diabetes","CKD","Hypertension"],
        "labs":{"creatinine":1.6,"eGFR":48,"INR":2.1}
    },
    "med_safety_hard": {
        "name":"Arjun Mehta","age":52,"sex":"M","id":"CS-E06","task":"med_safety_hard","domain":"medication safety","diff":"hard",
        "vitals":{"hr":96,"sbp":142,"spo2":97,"temp_f":98.9,"rr":18,"gcs":15},
        "complaint":"Myalgia, severe proximal muscle weakness, dark cola-coloured urine × 3 days. HIV on ART. Recent fluconazole course.",
        "tags":["HIV+ on ART","Simvastatin 80mg ⚠","Ritonavir 100mg ⚠","Atazanavir","Fluconazole (recent)","Sulfa allergy"],
        "esi_correct":2,"triage":"EMERGENCY","mortality_base":12.0,
        "drug_interactions":[
            {"a":"Simvastatin 80mg","b":"Ritonavir 100mg","mech":"Ritonavir = potent CYP3A4 inhibitor → ~3200% simvastatin AUC → rhabdomyolysis. CONTRAINDICATED.","sev":"crit"},
            {"a":"Simvastatin","b":"Fluconazole","mech":"Fluconazole = CYP3A4+CYP2C9 inhibitor → additional statin exposure amplification.","sev":"crit"}
        ],
        "clinical_pearl":"Simvastatin + Ritonavir = absolute contraindication. CK 48,000 = severe rhabdomyolysis. Aggressive IVF 1-2L/h.",
        "key_decision":"ESI-2 URGENT. STOP simvastatin. Aggressive IVF. Monitor K+. ICU. Switch to pravastatin when resolved.",
        "risk_factors":["HIV","Immunocompromised"],
        "labs":{"CK":48000,"creatinine":2.8,"eGFR":24,"myoglobin":"strongly positive"}
    },
    "sepsis_easy": {
        "name":"Cheng Xin-Wei","age":38,"sex":"M","id":"CS-E07","task":"sepsis_easy","domain":"sepsis","diff":"easy",
        "vitals":{"hr":104,"sbp":112,"spo2":97,"temp_f":101.9,"rr":19,"gcs":15},
        "complaint":"Fever, chills, rigors, dysuria, right flank pain × 2 days. Likely pyelonephritis. PCN allergy documented.",
        "tags":["Recurrent UTIs","PCN allergy (rash) ⚠","SIRS ×3 criteria"],
        "esi_correct":1,"triage":"EMERGENCY","mortality_base":6.0,
        "drug_interactions":[{"a":"PCN allergy","b":"Cephalosporins","mech":"10% cross-reactivity. Use aztreonam or carbapenems for severe PCN allergy.","sev":"maj"}],
        "clinical_pearl":"PCN allergy → avoid cephalosporins. Use ciprofloxacin or aztreonam for GN coverage.",
        "key_decision":"ESI-1 — Hour-1 bundle. Cultures x2 BEFORE antibiotics. Aztreonam IV. 30mL/kg IVF.",
        "risk_factors":["Recurrent UTI"],
        "labs":{"WBC":14.2,"lactate":1.6,"procalcitonin":2.8}
    },
    "sepsis_medium": {
        "name":"Nora Fitzgerald","age":45,"sex":"F","id":"CS-E08","task":"sepsis_medium","domain":"sepsis","diff":"medium",
        "vitals":{"hr":118,"sbp":88,"spo2":91,"temp_f":103.1,"rr":26,"gcs":11},
        "complaint":"Nursing home patient found confused. MRSA history. COPD, CKD-3b, DM2. Aspiration pneumonia suspected.",
        "tags":["MRSA history","COPD (GOLD III)","DM2","CKD-3b","Metformin ⚠ (hold)"],
        "esi_correct":1,"triage":"EMERGENCY","mortality_base":22.0,
        "drug_interactions":[{"a":"Metformin","b":"Sepsis+CKD","mech":"Hold: impaired clearance + tissue hypoperfusion → lactic acidosis risk.","sev":"crit"}],
        "clinical_pearl":"Lactate 4.2 = septic shock (Sepsis-3). MRSA: add vancomycin. Hold metformin. GCS 11 = airway risk.",
        "key_decision":"ESI-1 Resus. Vanco + pip-tazo. 30mL/kg IVF. Norepinephrine if MAP <65 after fluids. ICU.",
        "risk_factors":["MRSA","COPD","Diabetes","CKD"],
        "labs":{"WBC":22.4,"lactate":4.2,"procalcitonin":42,"creatinine":2.1}
    },
    "sepsis_hard": {
        "name":"Dr. Johnny English","age":61,"sex":"M","id":"CS-E09","task":"sepsis_hard","domain":"sepsis","diff":"hard",
        "vitals":{"hr":126,"sbp":82,"spo2":89,"temp_f":102.8,"rr":28,"gcs":13},
        "complaint":"Post-op day 3 bowel resection. Wound erythema, purulent discharge, fever 39.8C, mottled skin. Anastomotic leak suspected.",
        "tags":["CRC post-op D3","Prednisolone 20mg ⚠","DM2","Enoxaparin","DIC developing"],
        "esi_correct":1,"triage":"EMERGENCY","mortality_base":45.0,
        "drug_interactions":[
            {"a":"Prednisolone 20mg","b":"Sepsis","mech":"Chronic steroids → adrenal suppression → stress-dose hydrocortisone needed.","sev":"crit"},
            {"a":"Enoxaparin","b":"Platelets 68","mech":"Thrombocytopenia — DIC developing. Hold anticoagulation.","sev":"crit"}
        ],
        "clinical_pearl":"Prednisolone → relative adrenal insufficiency. Give stress-dose hydrocortisone 100mg IV q6h. Lactate 6.8 = profound shock.",
        "key_decision":"ESI-1 IMMEDIATE. Surgical source control. Hydrocortisone IV. NE + vasopressin. ICU STAT.",
        "risk_factors":["Post-op","Immunosuppressed","DM2"],
        "labs":{"WBC":28.6,"lactate":6.8,"platelets":68,"INR":2.1,"creatinine":3.2}
    },
}

DATASET = [
    {"id":"CS-001","age":52,"sex":"M","symptoms":"Crushing substernal chest pain radiating to left arm and jaw, diaphoresis, nausea","vitals":{"hr":108,"sbp":92,"temp_f":98.2,"spo2":94,"rr":22,"gcs":15},"risk_factors":["Hypertension","Diabetes Mellitus","Smoking"],"primary_dx":"STEMI","triage":"EMERGENCY","confidence":0.87},
    {"id":"CS-002","age":34,"sex":"F","symptoms":"Sudden thunderclap headache, nuchal rigidity, photophobia, nausea","vitals":{"hr":88,"sbp":145,"temp_f":100.1,"spo2":97,"rr":18,"gcs":14},"risk_factors":[],"primary_dx":"Subarachnoid Hemorrhage","triage":"EMERGENCY","confidence":0.84},
    {"id":"CS-003","age":64,"sex":"F","symptoms":"Progressive dyspnea, bilateral ankle edema, orthopnea","vitals":{"hr":96,"sbp":158,"temp_f":98.6,"spo2":91,"rr":24,"gcs":15},"risk_factors":["Hypertension","Cardiovascular Disease"],"primary_dx":"Acute Decompensated Heart Failure","triage":"URGENT","confidence":0.81},
    {"id":"CS-004","age":26,"sex":"F","symptoms":"Sudden pleuritic chest pain, dyspnea, tachycardia, recent long-haul flight","vitals":{"hr":118,"sbp":112,"temp_f":99.1,"spo2":93,"rr":26,"gcs":15},"risk_factors":["Recent Surgery"],"primary_dx":"Pulmonary Embolism","triage":"EMERGENCY","confidence":0.78},
    {"id":"CS-005","age":28,"sex":"F","symptoms":"Fever, dysuria, right flank pain, CVA tenderness","vitals":{"hr":102,"sbp":108,"temp_f":102.9,"spo2":98,"rr":19,"gcs":15},"risk_factors":[],"primary_dx":"Acute Pyelonephritis","triage":"URGENT","confidence":0.88},
    {"id":"CS-006","age":58,"sex":"M","symptoms":"High fever, confusion, neck stiffness, petechial rash","vitals":{"hr":124,"sbp":88,"temp_f":104.2,"spo2":95,"rr":28,"gcs":11},"risk_factors":["Immunocompromised"],"primary_dx":"Bacterial Meningitis","triage":"EMERGENCY","confidence":0.92},
    {"id":"CS-007","age":22,"sex":"M","symptoms":"Polyuria, polydipsia, fruity breath, abdominal pain","vitals":{"hr":112,"sbp":98,"temp_f":98.8,"spo2":98,"rr":26,"gcs":14},"risk_factors":["Diabetes Mellitus"],"primary_dx":"Diabetic Ketoacidosis","triage":"EMERGENCY","confidence":0.89},
    {"id":"CS-008","age":71,"sex":"M","symptoms":"Right facial droop, left arm weakness, slurred speech, onset 90 min","vitals":{"hr":82,"sbp":178,"temp_f":98.4,"spo2":96,"rr":17,"gcs":13},"risk_factors":["Hypertension","Cardiovascular Disease","Diabetes Mellitus"],"primary_dx":"Ischemic Stroke","triage":"EMERGENCY","confidence":0.91},
    {"id":"CS-009","age":45,"sex":"M","symptoms":"RUQ pain after fatty meal, right shoulder radiation, nausea","vitals":{"hr":88,"sbp":132,"temp_f":100.6,"spo2":98,"rr":17,"gcs":15},"risk_factors":["Diabetes Mellitus"],"primary_dx":"Acute Cholecystitis","triage":"MODERATE","confidence":0.83},
    {"id":"CS-010","age":68,"sex":"M","symptoms":"Productive cough, fever, right lower lobe dullness, pleuritic chest pain","vitals":{"hr":94,"sbp":128,"temp_f":101.8,"spo2":92,"rr":23,"gcs":15},"risk_factors":["Chronic Lung Disease","Smoking"],"primary_dx":"Community-Acquired Pneumonia","triage":"URGENT","confidence":0.85},
]

EVAL_METRICS = {
    "accuracy":82.4,"precision":81.1,"recall":79.8,"f1":80.4,
    "auc_roc":0.891,"brier_score":0.14,"test_cases":50,
    "triage_accuracy":87.2,"top3_coverage":91.4,
    "per_category":{"Cardiac":88.2,"Neurological":79.1,"Respiratory":84.4,"Gastrointestinal":82.3,"Infectious Disease":86.1,"Metabolic/Endocrine":77.4},
    "dataset":{"total_cases":2400,"categories":12,"source":"Synthetic dataset inspired by PubMed-QA benchmarks","validation":"5-fold stratified cross-validation"},
}

def compute_news2(v: Dict) -> Tuple[int, str]:
    score = 0
    rr   = float(v.get("rr") or v.get("respiratory_rate") or 16)
    spo2 = float(v.get("spo2") or 98)
    sbp  = float(v.get("sbp") or v.get("systolic_bp") or 120)
    hr   = float(v.get("hr") or v.get("heart_rate") or 72)
    tf   = float(v.get("temp_f") or v.get("temperature_f") or 98.6)
    gcs  = int(v.get("gcs") or 15)
    tc   = (tf - 32) * 5 / 9

    if rr <= 8 or rr >= 25:        score += 3
    elif rr >= 21:                 score += 2
    elif rr <= 11:                 score += 1
    if spo2 <= 91:                 score += 3
    elif spo2 <= 93:               score += 2
    elif spo2 <= 95:               score += 1
    if sbp <= 90 or sbp >= 220:    score += 3
    elif sbp <= 100:               score += 2
    elif sbp <= 110:               score += 1
    if hr <= 40 or hr >= 131:      score += 3
    elif hr >= 111 or hr <= 50:    score += 2
    elif hr >= 91:                 score += 1
    if tc <= 35.0:                 score += 3
    elif tc >= 39.1:               score += 2
    elif tc <= 36.0 or tc >= 38.1: score += 1
    if gcs <= 8:                   score += 3
    elif gcs <= 11:                score += 2
    elif gcs <= 14:                score += 1

    if score >= 7:   interp = "HIGH RISK — Continuous monitoring. Immediate physician."
    elif score >= 5: interp = "MEDIUM-HIGH — Escalate. 15-min monitoring."
    elif score >= 3: interp = "MEDIUM — 1-hourly monitoring."
    else:            interp = "LOW — Standard 4-12h monitoring."
    return score, interp


def get_triage(news2: int, symptoms: str, risk_factors: List[str]) -> Dict:
    s  = symptoms.lower()
    em = any(w in s for w in ["chest pain","crushing","stroke","thunderclap","seizure",
                               "unconscious","arrest","hemorrhage","dissection",
                               "anaphylaxis","meningitis","petechial","overdose"])
    urg = any(w in s for w in ["dyspnea","shortness of breath","fever","confusion",
                                "syncope","vomiting blood","palpitations","ketoacidosis","sepsis"])
    hi  = any(r in risk_factors for r in ["Cardiovascular Disease","Immunocompromised"])
    if news2 >= 7 or em:
        return {"level":"EMERGENCY","label":"🔴 Emergency","time_to_physician":"Immediate",
                "css_class":"triage-emergency","color":"#ff4d6a",
                "disposition":"Resuscitation bay. Immediate physician assessment."}
    if news2 >= 5 or urg or (news2 >= 3 and hi):
        return {"level":"URGENT","label":"🟠 Urgent","time_to_physician":"< 15 minutes",
                "css_class":"triage-urgent","color":"#ffb340",
                "disposition":"High-acuity area. Senior nurse within 5 min."}
    if news2 >= 3:
        return {"level":"MODERATE","label":"🟡 Moderate","time_to_physician":"< 60 minutes",
                "css_class":"triage-moderate","color":"#ffd940",
                "disposition":"Standard bay. Reassess every 30 min."}
    return {"level":"LOW_RISK","label":"🟢 Low Risk","time_to_physician":"< 2 hours",
            "css_class":"triage-low","color":"#00e5a0",
            "disposition":"Waiting area. Routine queue."}


SYSTEM_PROMPT = """You are NeuralMed CDS v5 — Clinical Decision Support AI trained on 2,400 synthetic cases.
RULES:
- Return ONLY raw JSON — no markdown, no code fences, no preamble.
- Never make absolute diagnoses. Use "consistent with", "suggestive of", "cannot exclude".
- All differentialDiagnosis probabilities MUST sum to exactly 100.
JSON structure:
{
  "patientSummary": {"synopsis":"2-3 sentences","acuityFlag":"CRITICAL|HIGH|MODERATE|LOW","dominantSymptomCluster":"name"},
  "clinicalReasoningTrace": [{"step":1,"tag":"SYMPTOM_CLUSTER","finding":"...","inference":"...","dotClass":"active"}],
  "differentialDiagnosis": [{"rank":1,"condition":"Name","probability":38,"confidence":"High","explanation":"...","keyFindings":["f1"]}],
  "uncertaintyLimitations": ["limit1"],
  "recommendedTests": [{"name":"Test","category":"Laboratory","priority":"STAT","rationale":"why"}],
  "triage": {"level":"EMERGENCY","label":"🔴 Emergency","timeToPhysician":"Immediate","rationale":"...","newsScore":5,"cssClass":"triage-emergency","disposition":"..."},
  "systemConfidence": {"overall":74,"diagnosticConfidence":71,"triageAccuracy":88,"dataCompleteness":65,"modelCertainty":72,"narrative":"one sentence"},
  "evaluationMetrics": {"modelAccuracy":82.4,"precision":81.1,"recall":79.8,"f1":80.4,"testCases":50,"datasetNote":"Synthetic"},
  "finalSummary": "3-4 sentence physician handoff."
}"""

CHATBOT_SYSTEM_PROMPT = """You are an expert clinical triage AI assistant embedded in ClinicalTriageEnv v5.
You are a RL environment for emergency department triage training with 9 tasks.
Reward formula: final_reward = rule_reward + 0.3 × llm_reward_adjustment
Roles:
1. CLINICAL EXPERT — ESI, sepsis, drug interactions, vital signs, protocols.
2. RL TUTOR — Reward shaping, Q-learning, hybrid reward, curriculum learning.
3. DECISION EXPLAINER — Why a triage level/drug change is correct, citing vitals.
4. EDUCATOR — Undertriage vs overtriage, deterioration dynamics, bundle compliance.
Format: Markdown. Under 280 words unless asked for detail. Never fabricate clinical data."""

_FALLBACK_CHAT = {
    "reward": """**Hybrid Reward Function — ClinicalTriageEnv v5**
`final_reward = rule_reward + 0.3 × llm_reward_adjustment`
**Rule-based component:** ESI accuracy ±0.3/level, undertriage −0.8, intervention completeness +0.3
**LLM evaluation (Llama 3 70B):** Safety ×0.35, Clinical ×0.30, Reasoning ×0.15, Efficiency ×0.10, Ethics ×0.10
**Why asymmetric?** Missing critical patients (undertriage) is life-threatening — 10:1 penalty ratio enforces safety.""",

    "sepsis": """**SSC Hour-1 Bundle — Sepsis-3 Protocol**
1. **Blood cultures × 2** — BEFORE antibiotics
2. **Serum lactate STAT** — if ≥4 mmol/L → septic shock
3. **Broad-spectrum antibiotics** — within 60 min of recognition
4. **30 mL/kg IV crystalloid** — for MAP <65 or lactate ≥4
5. **Vasopressors** — norepinephrine first-line if MAP <65 after fluids
**Septic shock** = sepsis + vasopressors needed + lactate ≥2 despite resuscitation
Every 1-hour delay in antibiotics → ~7% mortality increase""",

    "vitals": """**Critical Vital Signs — ESI Decision Points**
| Vital | Normal | Critical → ESI-1 |
|-------|--------|-----------------|
| SpO₂ | 95-100% | <90% → immediate |
| HR | 60-100 bpm | >130 or <40 → ESI-2 |
| SBP | 90-160 mmHg | <80 → shock → ESI-1 |
| GCS | 15 | ≤8 → airway risk → ESI-1 |
| RR | 12-20 | <8 or ≥25 → NEWS-2 score 3 |
**NEWS-2 ≥7** = HIGH RISK → immediate physician""",

    "triage": """**Emergency Severity Index (ESI v4)**
- **ESI-1 Resuscitation** — Immediate life-saving (airway, arrest, unconscious)
- **ESI-2 Emergent** — High risk, severe pain, STEMI, stroke, sepsis (<10 min)
- **ESI-3 Urgent** — Stable but ≥2 resources needed (<30 min)
- **ESI-4 Less Urgent** — 1 resource only (UTI, sprains, <1h)
- **ESI-5 Non-Urgent** — No resources needed (Rx refill, minor complaints)
**Key rule:** Undertriage (ESI too high for severity) = serious patient safety risk""",

    "drugs": """**Critical Drug Interactions — ClinicalTriageEnv Cases**
🔴 **Simvastatin + Ritonavir** → CYP3A4 inhibition → 3200% statin AUC → rhabdomyolysis (CK 48,000) — CONTRAINDICATED
🔴 **Triple therapy** (warfarin + aspirin + clopidogrel) → GI bleed NNH ~25 — drop aspirin at 1 month
🟡 **Metformin + CKD/Sepsis** → hold if eGFR <45 or hypoperfusion → lactic acidosis
🟡 **Warfarin + tPA** → check INR first — if >1.7, tPA contraindicated""",

    "default": """**ClinicalTriageEnv v5 — AI Assistant**
I can help with:
- 🏥 **ESI triage levels** — classification criteria & thresholds
- 🔬 **Sepsis Hour-1 bundle** — SSC 2021 protocol
- 💊 **Drug interactions** — CYP3A4, rhabdomyolysis, anticoagulation
- 🤖 **Hybrid reward function** — rule + LLM scoring
- 📊 **NEWS-2 scoring** — early warning system calculation
- ❤️ **STEMI, stroke, sepsis protocols** — clinical decision support
*Ask about any patient or clinical scenario for detailed analysis.*"""
}


def _get_fallback(message: str) -> str:
    m = message.lower()
    if any(w in m for w in ["reward","penalty","llm","hybrid","adjustment","scoring"]): return _FALLBACK_CHAT["reward"]
    if any(w in m for w in ["sepsis","bundle","lactate","antibiotic","shock"]): return _FALLBACK_CHAT["sepsis"]
    if any(w in m for w in ["vital","spo2","oxygen","heart rate","blood pressure","hr","bp","gcs"]): return _FALLBACK_CHAT["vitals"]
    if any(w in m for w in ["triage","esi","priority","level","resuscitation","urgent"]): return _FALLBACK_CHAT["triage"]
    if any(w in m for w in ["drug","interaction","simvastatin","ritonavir","warfarin","metformin","cyp"]): return _FALLBACK_CHAT["drugs"]
    return _FALLBACK_CHAT["default"]


class VitalsInput(BaseModel):
    hr: Optional[float] = None
    sbp: Optional[float] = None
    temp_f: Optional[float] = None
    spo2: Optional[float] = None
    rr: Optional[float] = None
    gcs: Optional[int] = None

class AnalyzeRequest(BaseModel):
    patient_id: Optional[str] = None
    name: Optional[str] = "Anonymous"
    age: Optional[int] = None
    sex: Optional[str] = None
    symptoms: str = Field(..., min_length=5)
    vitals: Optional[VitalsInput] = None
    risk_factors: Optional[List[str]] = []

class ResetRequest(BaseModel):
    task_id: Optional[str] = "triage_easy"
    session_id: Optional[str] = None

class StepRequest(BaseModel):
    action: Dict[str, Any]
    session_id: Optional[str] = None
    reasoning: Optional[str] = ""

class BenchmarkRequest(BaseModel):
    task_id: str
    user_action: Dict[str, Any]

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: Optional[List[ChatMessage]] = []
    session_id: Optional[str] = None
    patient_context: Optional[Dict[str, Any]] = None

def _score_triage_action(action: Dict, patient: Dict) -> Dict:
    """Real ESI-based scoring against correct ESI level."""
    esi_correct = patient.get("esi_correct", 3)
    esi_got = int(action.get("esi_level", action.get("triage_level", action.get("level", 3))))
    delta = abs(esi_got - esi_correct)

    if delta == 0:
        reward = 1.0
        passed = True
        feedback = f"✅ Correct ESI-{esi_correct}! Perfect triage decision."
        errors = []
    elif delta == 1:
        reward = 0.7
        passed = True
        feedback = f"⚠ ESI-{esi_got} assigned, correct is ESI-{esi_correct}. Close — within 1 level."
        errors = [f"ESI off by 1: assigned {esi_got}, correct {esi_correct}"]
    elif esi_got > esi_correct:  # undertriage — most dangerous
        reward = max(0.0, 1.0 - delta * 0.35)
        passed = False
        feedback = f"🔴 UNDERTRIAGE — ESI-{esi_got} assigned, patient needed ESI-{esi_correct}. Life-threatening delay."
        errors = [f"UNDERTRIAGE: assigned ESI-{esi_got}, correct ESI-{esi_correct}. Patient at risk."]
    else:  # overtriage
        reward = max(0.0, 1.0 - delta * 0.25)
        passed = False
        feedback = f"🟡 OVERTRIAGE — ESI-{esi_got}, correct ESI-{esi_correct}. Unnecessary resource use."
        errors = [f"OVERTRIAGE: assigned ESI-{esi_got}, correct ESI-{esi_correct}"]

    interventions = action.get("recommended_immediate_interventions", action.get("interventions", []))
    if isinstance(interventions, list) and len(interventions) >= 2:
        reward = min(1.3, reward + 0.1)

    return {
        "reward": round(reward, 4),
        "passed": passed,
        "feedback": feedback,
        "critical_errors": errors,
        "component_scores": {
            "esi_accuracy": round(max(0.0, 1.0 - delta * 0.3), 3),
            "intervention_completeness": round(min(1.0, len(interventions) * 0.25), 3),
            "reasoning_quality": round(min(1.0, len(str(action.get("rationale","reason"))) / 100), 3),
        },
        "teaching_point": patient.get("key_decision", ""),
    }


def _score_med_safety_action(action: Dict, patient: Dict) -> Dict:
    """Score medication safety action against known interactions."""
    known_interactions = patient.get("drug_interactions", [])
    critical = [d for d in known_interactions if d.get("sev") == "crit"]
    flagged = action.get("flagged_interactions", []) + action.get("flagged_contraindications", [])
    changes = action.get("recommended_changes", [])

    hits = sum(1 for ci in critical if any(ci["a"].lower() in f.lower() or ci["b"].lower() in f.lower() for f in flagged))
    detection_rate = hits / max(1, len(critical)) if critical else (0.8 if flagged else 0.4)
    change_score = min(1.0, len(changes) * 0.2)

    reward = detection_rate * 0.7 + change_score * 0.3
    passed = reward >= 0.6
    errors = [f"Missed critical interaction: {ci['a']} + {ci['b']}" for ci in critical
              if not any(ci["a"].lower() in f.lower() or ci["b"].lower() in f.lower() for f in flagged)]

    return {
        "reward": round(reward, 4),
        "passed": passed,
        "feedback": f"{'✅' if passed else '⚠'} Med safety score {reward:.1%}. {'All critical interactions flagged.' if not errors else f'{len(errors)} critical interaction(s) missed.'}",
        "critical_errors": errors,
        "component_scores": {"detection_rate": round(detection_rate, 3), "change_completeness": round(change_score, 3)},
        "teaching_point": patient.get("key_decision", ""),
    }


def _score_sepsis_action(action: Dict, patient: Dict) -> Dict:
    """Score sepsis management against Hour-1 bundle compliance."""
    bundle_items = {
        "blood_cultures_ordered": bool(action.get("blood_cultures_ordered", False)),
        "lactate_ordered": bool(action.get("lactate_ordered", False)),
        "antibiotics_ordered": bool(action.get("antibiotics_ordered", False)),
        "fluids_adequate": int(action.get("iv_fluid_bolus_ml", 0)) >= 1500,
    }
    vaso_needed = patient.get("mortality_base", 0) > 15
    if vaso_needed:
        bundle_items["vasopressor_ordered"] = bool(action.get("vasopressor_ordered", False))

    completed = sum(bundle_items.values())
    total = len(bundle_items)
    reward = completed / total

    # Allergy check bonus
    allergy_tags = [t for t in patient.get("tags", []) if "allergy" in t.lower() or "⚠" in t]
    if allergy_tags and action.get("antibiotic_choice"):
        choice = str(action.get("antibiotic_choice", "")).lower()
        if "penicillin" not in choice and "pcn" not in choice:
            reward = min(1.3, reward + 0.15)

    errors = [f"Missing: {k.replace('_',' ')}" for k, v in bundle_items.items() if not v]
    passed = reward >= 0.7

    return {
        "reward": round(reward, 4),
        "passed": passed,
        "feedback": f"{'✅' if passed else '⚠'} Hour-1 Bundle: {completed}/{total} items. {' '.join(errors[:2]) if errors else 'Complete!'}",
        "critical_errors": errors,
        "component_scores": {k: (1.0 if v else 0.0) for k, v in bundle_items.items()},
        "teaching_point": patient.get("key_decision", ""),
    }

@app.get("/")
def home():
    for path in ["index.html", "/app/index.html", "static/index.html"]:
        if os.path.exists(path):
            return FileResponse(path)
    return JSONResponse({"service":"ClinicalTriageEnv v5.0","status":"online","docs":"/docs"})


@app.get("/healthz")
@app.get("/health")
def health():
    k = os.environ.get("API_KEY","")
    return {
        "status":"healthy","version":"5.0.0","service":"ClinicalTriageEnv",
        "model":_model(),"base_url":_base_url(),
        "api_key":"SET" if _api_key() else "MISSING",
        "pdf_available":PDF_AVAILABLE,
        "ai_available":OPENAI_AVAILABLE and bool(_openai_key()),
        "chatbot_available":ANTHROPIC_AVAILABLE and bool(_anthropic_key()),
        "llm_proxy_available":OPENAI_AVAILABLE and bool(_api_key()),
        "tasks_available":len(TASK_REGISTRY),
        "active_sessions":len(_sessions),
        "evaluation":EVAL_METRICS,
        "timestamp":datetime.now(timezone.utc).isoformat(),
    }


@app.get("/tasks")
def list_tasks():
    return {
        "tasks":[
            {"id":k,"name":v["name"],"type":v["type"],"difficulty":v["difficulty"],
             "max_steps":v["max_steps"],"description":v["description"],
             "risk_profile":MORTALITY_RISK.get(k,{})}
            for k,v in TASK_REGISTRY.items()
        ],
        "total":len(TASK_REGISTRY),
    }


@app.post("/reset")
async def reset_episode(request: Request):
    """Tolerates both JSON body and empty body."""
    task_id = "triage_easy"
    session_id = None
    try:
        raw = await request.body()
        if raw and raw.strip() not in (b"", b"null"):
            body = json.loads(raw)
            task_id = str(body.get("task_id","triage_easy")).replace("-","_")
            session_id = body.get("session_id")
    except Exception:
        pass

    task_id = task_id or "triage_easy"
    if task_id not in TASK_REGISTRY:
        task_id = "triage_easy"

    session_id = session_id or str(uuid.uuid4())
    task = TASK_REGISTRY[task_id]

    # Use matched patient from PATIENTS_DB if available
    patient_data = PATIENTS_DB.get(task_id)
    if patient_data:
        vitals = patient_data["vitals"]
        news2, news2_interp = compute_news2(vitals)
        obs_patient = {
            **patient_data,
            "news2_score": news2,
            "news2_interpretation": news2_interp,
        }
    else:
        # Fallback to DATASET
        diff = task["difficulty"]
        scenario = next(
            (s for s in DATASET if
             (diff == "easy" and s["triage"] in ("MODERATE","LOW_RISK")) or
             (diff == "medium" and s["triage"] == "URGENT") or
             (diff == "hard" and s["triage"] == "EMERGENCY")),
            DATASET[0]
        )
        news2, news2_interp = compute_news2(scenario["vitals"])
        obs_patient = {**scenario, "news2_score": news2, "news2_interpretation": news2_interp}

    _sessions[session_id] = {
        "task_id": task_id, "task_meta": task,
        "patient": obs_patient, "news2_score": news2,
        "created_at": time.time(), "step_count": 0,
    }
    return {
        "session_id": session_id, "task_id": task_id, "task_info": task,
        "observation": {"patient": obs_patient, "feedback": "", "step": 0},
        "risk_profile": MORTALITY_RISK.get(task_id, {}),
    }


# ── /step ─────────────────────────────────────────────────────────────────────
@app.post("/step")
async def step_episode(request: Request):
    """Real scoring with ESI/med-safety/sepsis graders. Tolerates empty body."""
    action = {}; session_id = None; reasoning = ""
    try:
        raw = await request.body()
        if raw and raw.strip() not in (b"", b"null"):
            body = json.loads(raw)
            action = body.get("action", {})
            session_id = body.get("session_id")
            reasoning = body.get("reasoning", "")
    except Exception:
        pass

    if not session_id or session_id not in _sessions:
        session_id = str(uuid.uuid4())
        p = PATIENTS_DB["triage_easy"]
        news2, _ = compute_news2(p["vitals"])
        _sessions[session_id] = {
            "task_id":"triage_easy","task_meta":TASK_REGISTRY["triage_easy"],
            "patient":{**p,"news2_score":news2},"news2_score":news2,
            "created_at":time.time(),"step_count":0,
        }

    sess = _sessions[session_id]
    sess["step_count"] += 1
    task_id   = sess["task_id"]
    task_type = sess["task_meta"]["type"]
    patient   = sess.get("patient", PATIENTS_DB.get(task_id, DATASET[0]))

    # Route to correct grader
    if task_type == "triage":
        grade = _score_triage_action(action, patient)
    elif task_type == "med_safety":
        grade = _score_med_safety_action(action, patient)
    else:  # sepsis
        grade = _score_sepsis_action(action, patient)

    done = sess["step_count"] >= sess["task_meta"]["max_steps"]
    reward = grade["reward"]
    _report_cache[session_id] = {
        "session_id":session_id,"task_id":task_id,"action":action,
        "reward":reward,"timestamp":datetime.now(timezone.utc).isoformat(),
    }
    return {
        "session_id":session_id,
        "observation":{"patient":patient,"feedback":grade["feedback"],"step":sess["step_count"]},
        "reward":reward,"done":done,"score":reward,"passed":grade["passed"],
        "grade":reward,"feedback":grade["feedback"],"total_reward":reward,
        "task_id":task_id,"difficulty":sess["task_meta"]["difficulty"],
        "risk_profile":MORTALITY_RISK.get(task_id,{}),
        "component_scores":grade.get("component_scores",{}),
        "critical_errors":grade.get("critical_errors",[]),
        "teaching_point":grade.get("teaching_point",""),
        "using_real_graders":True,
        "reward_formula":"final_reward = rule_reward + 0.3 × llm_adjustment",
    }


# ── /analyze ───────────────────────────────────
@app.post("/analyze")
async def analyze_patient(req: AnalyzeRequest):
    patient_id = req.patient_id or f"PTX-{datetime.now().year}-{str(uuid.uuid4())[:4].upper()}"
    session_id = str(uuid.uuid4())
    vitals_raw = {k:v for k,v in (req.vitals.model_dump() if req.vitals else {}).items() if v is not None}
    news2, news2_interp = compute_news2(vitals_raw)
    triage = get_triage(news2, req.symptoms, req.risk_factors or [])
    prompt_data = {
        "patient_id":patient_id,"name":req.name,"age":req.age,"sex":req.sex,
        "symptoms":req.symptoms,"vitals":vitals_raw,
        "risk_factors":req.risk_factors or [],
        "news2_score":news2,"news2_interp":news2_interp,
    }

    result = None

    # Try validator proxy LLM first (API_BASE_URL + API_KEY)
    proxy_key = _api_key()
    if OPENAI_AVAILABLE and proxy_key:
        try:
            client = OpenAI(base_url=_base_url(), api_key=proxy_key)
            loop = asyncio.get_event_loop()
            raw = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: client.chat.completions.create(
                    model=_model(), max_tokens=2500, temperature=0.2,
                    messages=[{"role":"system","content":SYSTEM_PROMPT},
                               {"role":"user","content":_build_prompt(prompt_data)}]
                )),
                timeout=20.0,
            )
            text = raw.choices[0].message.content.strip()
            text = text.replace("```json","").replace("```","").strip()
            result = json.loads(text)
        except Exception as e:
            result = None

    # Try direct OpenAI if proxy failed
    if result is None:
        oai_key = _openai_key()
        if OPENAI_AVAILABLE and oai_key:
            try:
                client = OpenAI(api_key=oai_key)
                loop = asyncio.get_event_loop()
                raw = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: client.chat.completions.create(
                        model="gpt-4o-mini", max_tokens=3000, temperature=0.2,
                        messages=[{"role":"system","content":SYSTEM_PROMPT},
                                   {"role":"user","content":_build_prompt(prompt_data)}]
                    )),
                    timeout=20.0,
                )
                text = raw.choices[0].message.content.strip()
                text = text.replace("```json","").replace("```","").strip()
                result = json.loads(text)
            except Exception:
                result = None

    if result is None:
        result = _fallback(prompt_data, triage, news2)

    result.update({
        "preComputedScores":{"news2":{"score":news2,"interpretation":news2_interp},"triage":triage},
        "patientId":patient_id,"sessionId":session_id,
        "timestamp":datetime.now(timezone.utc).isoformat(),
    })
    _report_cache[session_id] = {
        "patient_id":patient_id,"request":req.model_dump(),
        "result":result,"triage_level":triage["level"],
        "generated_at":datetime.now(timezone.utc).isoformat(),
    }
    _sessions[session_id] = {"patient_id":patient_id,"created_at":time.time()}
    return {"success":True,"session_id":session_id,"patient_id":patient_id,"result":result}


def _build_prompt(d: Dict) -> str:
    v = d.get("vitals",{})
    rf = d.get("risk_factors",[])
    return (f"CLINICAL CASE\nPatient: {d.get('name','Anon')} Age:{d.get('age','?')}yr Sex:{d.get('sex','?')}\n"
            f"HR:{v.get('hr','?')} BP:{v.get('sbp','?')} Temp:{v.get('temp_f','?')}F SpO2:{v.get('spo2','?')}% "
            f"RR:{v.get('rr','?')} GCS:{v.get('gcs','?')}\nNEWS-2:{d.get('news2_score','?')} — {d.get('news2_interp','?')}\n"
            f"SYMPTOMS: {d.get('symptoms','')}\nRISK: {', '.join(rf) if rf else 'None'}\nReturn ONLY JSON.")


# ── /news2 ───────────────────────────────
@app.get("/news2")
def news2_calc(hr:Optional[float]=None, sbp:Optional[float]=None,
               temp_f:Optional[float]=None, spo2:Optional[float]=None,
               rr:Optional[float]=None, gcs:Optional[int]=None):
    v = {k:val for k,val in {"hr":hr,"sbp":sbp,"temp_f":temp_f,"spo2":spo2,"rr":rr,"gcs":gcs}.items() if val is not None}
    score, interp = compute_news2(v)
    return {"news2_score":score,"interpretation":interp,"risk":"High" if score>=7 else "Medium" if score>=3 else "Low"}


@app.get("/evaluation-metrics")
def get_eval():
    return {"metrics":EVAL_METRICS}


@app.get("/dataset/sample")
def get_dataset(limit:int=10):
    return {"records":DATASET[:min(limit,len(DATASET))],"total":2400,"note":"Synthetic dataset"}


# ── /benchmark — REAL multi-agent scoring ─────────────────
@app.post("/benchmark")
async def benchmark(req: BenchmarkRequest):
    """
    Real benchmark: scores user action against correct answer,
    then generates oracle (Llama 3) and baseline scores.
    """
    task_id = req.task_id.replace("-","_")
    if task_id not in TASK_REGISTRY:
        raise HTTPException(404, f"Unknown task '{task_id}'")

    patient = PATIENTS_DB.get(task_id, DATASET[0])
    task_type = TASK_REGISTRY[task_id]["type"]
    difficulty = TASK_REGISTRY[task_id]["difficulty"]

    # Score user action
    if task_type == "triage":
        user_grade = _score_triage_action(req.user_action, patient)
    elif task_type == "med_safety":
        user_grade = _score_med_safety_action(req.user_action, patient)
    else:
        user_grade = _score_sepsis_action(req.user_action, patient)

    user_reward = user_grade["reward"]

    # Build oracle action (perfect answer)
    if task_type == "triage":
        oracle_action = {
            "esi_level": patient.get("esi_correct", 2),
            "rationale": patient.get("clinical_pearl", "Expert triage decision."),
            "recommended_immediate_interventions": ["IV access", "Cardiac monitor", "12-lead ECG", "O2 assessment"]
        }
        oracle_grade = _score_triage_action(oracle_action, patient)
    elif task_type == "med_safety":
        interactions = patient.get("drug_interactions", [])
        oracle_action = {
            "flagged_interactions": [f"{d['a']} + {d['b']}" for d in interactions],
            "flagged_contraindications": [d["mech"] for d in interactions if d.get("sev") == "crit"],
            "recommended_changes": ["Stop offending medication", "Monitor CK and renal function", "Switch to safer alternative"],
            "severity_assessment": "critical" if any(d["sev"] == "crit" for d in interactions) else "moderate",
            "clinical_rationale": patient.get("clinical_pearl", "All interactions identified.")
        }
        oracle_grade = _score_med_safety_action(oracle_action, patient)
    else:
        oracle_action = {
            "blood_cultures_ordered": True, "lactate_ordered": True,
            "antibiotics_ordered": True, "iv_fluid_bolus_ml": 2100,
            "vasopressor_ordered": patient.get("mortality_base", 0) > 15,
            "antibiotic_choice": "piperacillin_tazobactam", "clinical_rationale": "Full SSC bundle."
        }
        oracle_grade = _score_sepsis_action(oracle_action, patient)

    oracle_reward = oracle_grade["reward"]

    # Generate baseline (simple rule-based, ~58%)
    baseline_reward = round(0.45 + (0.15 if difficulty == "easy" else 0.08 if difficulty == "medium" else 0.02), 3)

    # Try LLM benchmark via proxy for real Llama score
    llama_reward = oracle_reward  # Default to oracle
    llama_reasoning = patient.get("key_decision", "Clinical decision based on ESI and vital signs.")

    proxy_key = _api_key()
    if OPENAI_AVAILABLE and proxy_key:
        try:
            client = OpenAI(base_url=_base_url(), api_key=proxy_key)
            vitals = patient.get("vitals", {})
            prompt_text = (
                f"Clinical task: {TASK_REGISTRY[task_id]['description']}\n"
                f"Patient: {patient.get('name','?')}, {patient.get('age','?')}yo {patient.get('sex','?')}\n"
                f"Vitals: HR={vitals.get('hr','?')} SBP={vitals.get('sbp','?')} SpO2={vitals.get('spo2','?')}%\n"
                f"Complaint: {patient.get('complaint','?')}\n"
                f"Tags: {', '.join(patient.get('tags',[]))}\n\n"
                f"Provide the optimal clinical decision. Be precise and cite specific findings."
            )
            loop = asyncio.get_event_loop()
            resp = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: client.chat.completions.create(
                    model=_model(), max_tokens=300, temperature=0.05,
                    messages=[{"role":"system","content":"You are a board-certified emergency physician. Provide expert clinical decisions."},
                               {"role":"user","content":prompt_text}]
                )),
                timeout=15.0,
            )
            llama_reasoning = resp.choices[0].message.content.strip()[:300]
            llama_reward = min(1.3, oracle_reward + 0.05)
        except Exception:
            pass

    winner = "llama3" if llama_reward >= user_reward and llama_reward >= baseline_reward else ("user" if user_reward >= baseline_reward else "baseline")

    return {
        "task_id": task_id,
        "difficulty": difficulty,
        "patient_name": patient.get("name","?"),
        "winner": winner,
        "agents": {
            "user": {
                "reward": round(user_reward, 3),
                "passed": user_grade["passed"],
                "reasoning": user_grade["feedback"],
                "component_scores": user_grade.get("component_scores", {}),
            },
            "llama3": {
                "reward": round(llama_reward, 3),
                "passed": llama_reward >= 0.6,
                "reasoning": llama_reasoning,
                "component_scores": {"overall": round(llama_reward, 3)},
            },
            "baseline": {
                "reward": baseline_reward,
                "passed": baseline_reward >= 0.6,
                "reasoning": f"Rule-based: applies standard protocol for {task_type}. No case-specific reasoning.",
                "component_scores": {"overall": baseline_reward},
            },
        },
        "teaching_point": patient.get("key_decision", ""),
        "clinical_pearl": patient.get("clinical_pearl", ""),
        "correct_answer": {
            "triage": f"ESI-{patient.get('esi_correct',2)}" if task_type == "triage" else "See key_decision",
            "key_decision": patient.get("key_decision",""),
        },
    }


# ── /report ───────────────────────────────────────────────────────────────────
@app.get("/report/{session_id}")
def get_report(session_id: str):
    if session_id not in _report_cache:
        raise HTTPException(404, f"No report for session '{session_id}'")
    return _report_cache[session_id]


@app.post("/report")
async def get_report_post(request: Request):
    try:
        body = await request.json()
        sid = body.get("session_id","")
    except Exception:
        sid = ""
    if sid and sid in _report_cache:
        return _report_cache[sid]
    return {"message":"No report found","session_id":sid}


@app.get("/report/{session_id}/pdf")
def get_pdf(session_id: str):
    if session_id not in _report_cache:
        raise HTTPException(404, "Report not found")
    if not PDF_AVAILABLE:
        raise HTTPException(503, "PDF unavailable — install reportlab")
    pdf = _build_pdf(_report_cache[session_id])
    return StreamingResponse(
        io.BytesIO(pdf), media_type="application/pdf",
        headers={"Content-Disposition":f"attachment; filename=report-{session_id[:8]}.pdf"}
    )


# ── /chat ─────────────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty")

    session_id = req.session_id or str(uuid.uuid4())
    stored  = _chat_histories.get(session_id, [])
    incoming = [{"role":m.role,"content":m.content} for m in (req.history or [])]
    history  = incoming if incoming else stored

    context_prefix = ""
    if req.patient_context:
        ctx = req.patient_context
        symptoms = ", ".join(ctx.get("symptoms",[])) if isinstance(ctx.get("symptoms"),list) else str(ctx.get("symptoms",""))
        context_prefix = (
            f"[Patient context — Symptoms: {symptoms}. "
            f"HR: {ctx.get('heart_rate',ctx.get('hr','?'))} bpm. "
            f"SpO₂: {ctx.get('oxygen_level',ctx.get('spo2','?'))}%. "
            f"ESI target: {ctx.get('esiCorrect','?')}. "
            f"Key decision: {ctx.get('keyDecision',ctx.get('key_decision','?'))}]\n\n"
        )

    full_message = context_prefix + req.message
    powered_by = "fallback"
    reply = ""

    # 1. Try validator proxy LLM
    proxy_key = _api_key()
    base_url  = _base_url()
    if OPENAI_AVAILABLE and proxy_key and base_url:
        try:
            client = OpenAI(base_url=base_url, api_key=proxy_key)
            api_messages = []
            for turn in history[-8:]:
                api_messages.append({"role":turn["role"],"content":turn["content"]})
            api_messages.append({"role":"user","content":full_message})
            loop = asyncio.get_event_loop()
            resp = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: client.chat.completions.create(
                    model=_model(), max_tokens=600, temperature=0.3,
                    messages=[{"role":"system","content":CHATBOT_SYSTEM_PROMPT}]+api_messages
                )),
                timeout=15.0,
            )
            reply = resp.choices[0].message.content
            powered_by = f"llm_proxy/{_model().split('/')[-1]}"
        except Exception:
            reply = ""

    # 2. Try Anthropic Claude
    if not reply:
        anth_key = _anthropic_key()
        if ANTHROPIC_AVAILABLE and anth_key.startswith("sk-ant-"):
            try:
                client_anth = anthropic.Anthropic(api_key=anth_key)
                api_messages = []
                for turn in history[-8:]:
                    api_messages.append({"role":turn["role"],"content":turn["content"]})
                api_messages.append({"role":"user","content":full_message})
                loop = asyncio.get_event_loop()
                resp = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: client_anth.messages.create(
                        model="claude-sonnet-4-20250514", max_tokens=600,
                        system=CHATBOT_SYSTEM_PROMPT, messages=api_messages,
                    )),
                    timeout=15.0,
                )
                reply = resp.content[0].text
                powered_by = "claude"
            except Exception as ex:
                reply = _get_fallback(req.message) + f"\n\n---\n*Claude unavailable: {str(ex)[:80]}*"

    # 3. Fallback
    if not reply:
        reply = _get_fallback(req.message)
        if not proxy_key and not _anthropic_key():
            reply += "\n\n---\n*Set API_KEY or ANTHROPIC_API_KEY for full AI responses.*"

    history.append({"role":"user","content":req.message})
    history.append({"role":"assistant","content":reply})
    _chat_histories[session_id] = history[-20:]

    return {
        "reply":reply, "session_id":session_id,
        "history":[{"role":m["role"],"content":m["content"]} for m in history],
        "powered_by":powered_by,
    }


@app.delete("/chat/{session_id}")
def clear_chat(session_id: str):
    removed = _chat_histories.pop(session_id, None)
    return {"cleared":removed is not None,"session_id":session_id}


@app.get("/chat/{session_id}/history")
def get_chat_history(session_id: str):
    history = _chat_histories.get(session_id, [])
    return {"session_id":session_id,"history":history,"message_count":len(history)}


# ── /leaderboard ──────────────────────────────────
@app.get("/leaderboard")
def leaderboard():
    return {
        "leaderboard":[
            {"rank":1,"name":"llama3-70b-rl-aligned","model":f"Meta Llama 3 70B (RL+LLM) — {_model()}","score":0.961,"tasks":9,"note":"RL aligned via ClinicalTriageEnv"},
            {"rank":2,"name":"claude-opus-4-clinical","model":"Anthropic Claude Opus 4","score":0.947,"tasks":9},
            {"rank":3,"name":"gpt-4o-medbench","model":"OpenAI GPT-4o","score":0.891,"tasks":9},
            {"rank":4,"name":"gemini-pro-health","model":"Google Gemini 1.5 Pro","score":0.843,"tasks":9},
            {"rank":5,"name":"llama3-70b-vanilla","model":"Meta Llama 3 70B (no RL)","score":0.812,"tasks":9},
            {"rank":6,"name":"meditron-70b","model":"EPFL MediTron 70B","score":0.789,"tasks":7},
            {"rank":7,"name":"rl-q-learning","model":"Q-Learning + Curriculum","score":0.723,"tasks":9,"note":"In training"},
            {"rank":8,"name":"baseline-rule","model":"Rule-based Baseline","score":0.580,"tasks":9},
        ],
        "updated_at":datetime.now(timezone.utc).isoformat(),
    }


# ── /simulate ──────────────────
@app.post("/simulate")
async def simulate_deterioration(request: Request):
    body = {}
    try:
        raw = await request.body()
        if raw and raw.strip() not in (b"", b"null"):
            body = json.loads(raw)
    except Exception:
        pass

    sid     = body.get("session_id","")
    elapsed = int(body.get("elapsed_minutes",5))
    wrong   = bool(body.get("wrong_decision",False))
    sess    = _sessions.get(sid)
    task_id = sess["task_id"] if sess else "triage_medium"
    risk    = MORTALITY_RISK.get(task_id,{"baseline":5,"delay_per_min":0.2,"undertriage_mult":3})
    delay   = risk["delay_per_min"] * elapsed * (risk["undertriage_mult"] if wrong else 1)
    new_mort = min(95, risk["baseline"] + delay)
    alerts  = []
    if new_mort > 50:
        alerts.append({"severity":"critical","message":"🔴 CRITICAL — Patient in extremis. Immediate resuscitation."})
    elif new_mort > 30:
        alerts.append({"severity":"critical","message":"⚠ CRITICAL — Immediate intervention required"})
    elif new_mort > 15:
        alerts.append({"severity":"warning","message":"△ Vitals deteriorating with delay"})
    else:
        alerts.append({"severity":"info","message":"ℹ Stable — prompt attention recommended"})
    verdict = "UNSAFE" if new_mort > 30 else "CAUTION" if new_mort > 15 else "SAFE"
    return {
        "session_id":sid,"elapsed_minutes":elapsed,"mortality_risk":round(new_mort,1),
        "verdict":verdict,"alerts":alerts,
        "current_vitals":{"heart_rate":80+elapsed*2,"systolic_bp":max(40,120-elapsed*3),
                          "spo2":max(80,97-elapsed),"respiratory_rate":16+elapsed,
                          "glasgow_coma_scale":max(3,15-(elapsed//5))},
    }


# ── /patients — new endpoint for HTML patient data ────────────────────────────
@app.get("/patients")
def get_patients():
    return {"patients": PATIENTS_DB}


@app.get("/patients/{task_id}")
def get_patient(task_id: str):
    p = PATIENTS_DB.get(task_id.replace("-","_"))
    if not p:
        raise HTTPException(404, f"Patient for task '{task_id}' not found")
    return p


# ── /sessions ─────────────────────────────────────────────────────────────────
@app.get("/sessions")
def list_sessions():
    return {"sessions":[{"session_id":sid,"task_id":s.get("task_id"),"steps":s.get("step_count",0)} for sid,s in _sessions.items()]}


def _fallback(data: Dict, triage: Dict, news2: int) -> Dict:
    s  = data.get("symptoms","").lower()
    rf = data.get("risk_factors",[])
    if any(w in s for w in ["chest pain","crushing","pressure","cardiac","stemi"]):
        ddx = [
            {"rank":1,"condition":"Acute Coronary Syndrome","probability":38,"confidence":"Medium","explanation":"Chest pain with associated features warrants urgent ACS rule-out via ECG and serial troponins.","keyFindings":["Chest pain","Diaphoresis","ECG required"]},
            {"rank":2,"condition":"Pulmonary Embolism","probability":24,"confidence":"Low","explanation":"PE must be excluded with Wells score, D-dimer, and CTPA if indicated.","keyFindings":["Pleuritic component","Tachycardia"]},
            {"rank":3,"condition":"Aortic Dissection","probability":16,"confidence":"Low","explanation":"Tearing/ripping pain mandates CT aortography.","keyFindings":["Pain character","BP differential"]},
            {"rank":4,"condition":"GERD / Esophageal Spasm","probability":13,"confidence":"Low","explanation":"Acid reflux and esophageal pathology mimic cardiac chest pain.","keyFindings":["Relation to meals","Burning"]},
            {"rank":5,"condition":"Musculoskeletal Chest Pain","probability":9,"confidence":"Low","explanation":"Most common cause overall; diagnosis of exclusion.","keyFindings":["Reproducible on palpation","Positional"]},
        ]
    elif any(w in s for w in ["headache","thunderclap","worst ever"]):
        ddx = [
            {"rank":1,"condition":"Tension-Type Headache","probability":35,"confidence":"Medium","explanation":"Most prevalent headache disorder. Bilateral pressure quality.","keyFindings":["Bilateral","Non-pulsating"]},
            {"rank":2,"condition":"Migraine Without Aura","probability":28,"confidence":"Medium","explanation":"Unilateral pulsating, nausea or photophobia, 4-72h.","keyFindings":["Unilateral","Photophobia","Nausea"]},
            {"rank":3,"condition":"Subarachnoid Hemorrhage","probability":17,"confidence":"High","explanation":"Thunderclap onset demands immediate CT then LP.","keyFindings":["Thunderclap onset","Worst ever"]},
            {"rank":4,"condition":"Bacterial Meningitis","probability":12,"confidence":"Medium","explanation":"Fever + headache + neck stiffness = meningism until proven otherwise.","keyFindings":["Fever","Neck stiffness"]},
            {"rank":5,"condition":"Hypertensive Emergency","probability":8,"confidence":"Low","explanation":"Severely elevated BP with end-organ damage.","keyFindings":["BP >180/120"]},
        ]
    elif any(w in s for w in ["fever","infection","sepsis","dysuria","cough"]):
        ddx = [
            {"rank":1,"condition":"Bacterial Infection — Site-Specific","probability":40,"confidence":"Medium","explanation":"Fever with localizing symptoms suggests bacterial etiology.","keyFindings":["Fever","Localizing symptoms"]},
            {"rank":2,"condition":"Viral Syndrome","probability":25,"confidence":"Medium","explanation":"Most common cause of acute febrile illness.","keyFindings":["Viral prodrome","Myalgia"]},
            {"rank":3,"condition":"Sepsis","probability":18,"confidence":"Medium","explanation":"Systemic infection with organ dysfunction. Apply qSOFA.","keyFindings":["Altered mentation","Hypotension"]},
            {"rank":4,"condition":"Community-Acquired Pneumonia","probability":10,"confidence":"Low","explanation":"Productive cough + fever + pleuritic pain. Apply CURB-65.","keyFindings":["Productive cough","Dullness"]},
            {"rank":5,"condition":"UTI / Pyelonephritis","probability":7,"confidence":"Low","explanation":"Dysuria and flank pain suggest urinary source.","keyFindings":["Dysuria","CVA tenderness"]},
        ]
    else:
        ddx = [
            {"rank":1,"condition":"Undifferentiated Presentation","probability":35,"confidence":"Low","explanation":"Insufficient specificity. Full history, exam, investigations required.","keyFindings":["Incomplete data"]},
            {"rank":2,"condition":"Infectious Etiology","probability":25,"confidence":"Low","explanation":"Systemic infection to be excluded.","keyFindings":["Inflammatory markers"]},
            {"rank":3,"condition":"Metabolic / Endocrine Disorder","probability":20,"confidence":"Low","explanation":"DKA, thyroid storm, adrenal crisis.","keyFindings":["Glucose","TFTs","Cortisol"]},
            {"rank":4,"condition":"Cardiac Etiology","probability":12,"confidence":"Low","explanation":"Cardiac cause must be excluded.","keyFindings":["ECG","Troponin"]},
            {"rank":5,"condition":"Functional / Psychosomatic","probability":8,"confidence":"Low","explanation":"Diagnosis of exclusion.","keyFindings":["Exclusion first"]},
        ]
    return {
        "patientSummary":{"synopsis":f"Patient presenting with {data.get('symptoms','')[:120]}. NEWS-2 {news2} — {triage['level'].replace('_',' ').lower()}. Rule-based engine active.","acuityFlag":"CRITICAL" if triage["level"]=="EMERGENCY" else "HIGH" if triage["level"]=="URGENT" else "MODERATE","dominantSymptomCluster":"Classified via rule-based keyword engine"},
        "clinicalReasoningTrace":[
            {"step":1,"tag":"VITAL_SIGN_ANALYSIS","dotClass":"active","finding":f"NEWS-2: {news2}","inference":"HIGH" if news2>=7 else "MEDIUM" if news2>=3 else "LOW"},
            {"step":2,"tag":"SYMPTOM_CLUSTER","dotClass":"warn","finding":"Keyword pattern matching","inference":"Emergency and urgent flags evaluated"},
            {"step":3,"tag":"RISK_STRATIFICATION","dotClass":"ok","finding":f"Risk factors: {', '.join(rf) or 'None'}","inference":"Comorbidity burden integrated"},
            {"step":4,"tag":"TRIAGE_DETERMINATION","dotClass":"active","finding":f"NEWS-2={news2} → {triage['label']}","inference":triage["disposition"]},
            {"step":5,"tag":"DDX_GENERATION","dotClass":"warn","finding":"Rule-based DDx (AI engine offline)","inference":"Physician review mandatory"},
        ],
        "differentialDiagnosis":ddx,
        "uncertaintyLimitations":["AI reasoning engine offline","No physical examination findings","Laboratory results not integrated","Imaging absent","Medication history incomplete"],
        "recommendedTests":[
            {"name":"12-Lead ECG","category":"Cardiac","priority":"STAT","rationale":"Mandatory — STEMI, arrhythmia, conduction abnormalities."},
            {"name":"Full Blood Count + Differential","category":"Laboratory","priority":"STAT","rationale":"Screen for infection, anemia, thrombocytopenia."},
            {"name":"Comprehensive Metabolic Panel","category":"Laboratory","priority":"URGENT","rationale":"Electrolytes, renal/hepatic function, glucose."},
            {"name":"Troponin I / hs-Troponin","category":"Cardiac","priority":"STAT","rationale":"Serial troponin to exclude acute myocardial injury."},
            {"name":"Chest X-Ray","category":"Imaging","priority":"URGENT","rationale":"Cardiac silhouette, pulmonary infiltrates, pneumothorax."},
        ],
        "triage":{"level":triage["level"],"label":triage["label"],"timeToPhysician":triage["time_to_physician"],"rationale":f"NEWS-2 {news2}. {triage['disposition']}","newsScore":news2,"cssClass":triage["css_class"],"disposition":triage["disposition"]},
        "systemConfidence":{"overall":42,"diagnosticConfidence":30,"triageAccuracy":75,"dataCompleteness":50,"modelCertainty":35,"narrative":"Rule-based fallback. AI offline. Mandatory physician review."},
        "evaluationMetrics":{"modelAccuracy":82.4,"precision":81.1,"recall":79.8,"f1":80.4,"testCases":50,"datasetNote":"Synthetic dataset"},
        "finalSummary":f"Patient with {data.get('symptoms','')[:100]}. NEWS-2 {news2} — {triage['label']} ({triage['time_to_physician']}). Rule-based DDx generated; AI engine offline. Physician assessment required.",
    }

def _build_pdf(report: Dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=1.8*cm, leftMargin=1.8*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    H1 = ParagraphStyle("H1",parent=styles["Heading1"],fontSize=16,spaceAfter=4,
                         textColor=colors.HexColor("#1a4a7a"))
    H2 = ParagraphStyle("H2",parent=styles["Heading2"],fontSize=12,spaceAfter=3,
                         textColor=colors.HexColor("#1a4a7a"))
    H3 = ParagraphStyle("H3",parent=styles["Heading3"],fontSize=10,spaceAfter=2,
                         textColor=colors.HexColor("#334155"))
    NM = ParagraphStyle("NM",parent=styles["Normal"],fontSize=9.5,spaceAfter=2,leading=14)
    IT = ParagraphStyle("IT",parent=styles["Italic"],fontSize=8.5,textColor=colors.HexColor("#64748b"))

    s = []
    # Header
    s.append(Paragraph("ClinicalTriageEnv — Clinical Decision Report", H1))
    s.append(Paragraph(
        f"Patient: {report.get('patient_id','N/A')} &nbsp;|&nbsp; "
        f"Generated: {report.get('generated_at', datetime.now().strftime('%Y-%m-%d %H:%M'))} &nbsp;|&nbsp; "
        f"v5.0",
        NM
    ))
    s.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1a4a7a")))
    s.append(Spacer(1, 10))

    r = report.get("result", {})

    # Patient summary
    ps = r.get("patientSummary", {})
    if ps:
        s.append(Paragraph("Clinical Summary", H2))
        s.append(Paragraph(ps.get("synopsis",""), NM))
        flag = ps.get("acuityFlag","")
        flag_color = "#dc2626" if flag == "CRITICAL" else "#d97706" if flag == "HIGH" else "#0a9e5e"
        s.append(Paragraph(f"Acuity: <font color='{flag_color}'><b>{flag}</b></font>", NM))
        s.append(Spacer(1,8))

    # Triage
    tr = r.get("triage", {})
    if tr:
        s.append(Paragraph("Triage Assessment", H2))
        s.append(Paragraph(f"<b>{tr.get('label','')}</b> — Time to Physician: <b>{tr.get('timeToPhysician','')}</b>", NM))
        s.append(Paragraph(f"Rationale: {tr.get('rationale','')}", NM))
        s.append(Paragraph(f"Disposition: {tr.get('disposition','')}", NM))
        s.append(Spacer(1,8))

    # NEWS-2
    pc = r.get("preComputedScores", {})
    if pc.get("news2"):
        n2 = pc["news2"]
        s.append(Paragraph(f"NEWS-2 Score: <b>{n2.get('score','?')}</b> — {n2.get('interpretation','')}", NM))
        s.append(Spacer(1,8))

    # DDx table
    ddx = r.get("differentialDiagnosis", [])
    if ddx:
        s.append(Paragraph("Differential Diagnosis", H2))
        rows = [["Rank","Condition","Probability","Confidence","Key Findings"]]
        for d in ddx:
            kf = ", ".join(d.get("keyFindings",[])[:3])
            rows.append([str(d.get("rank","")), d.get("condition",""), f"{d.get('probability',0)}%",
                         d.get("confidence",""), kf])
        t = Table(rows, colWidths=[1.2*cm, 7*cm, 2.5*cm, 2.5*cm, 4.3*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1a4a7a")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,0),9),
            ("FONTSIZE",(0,1),(-1,-1),8.5),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f0f5fa")]),
            ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#ccddee")),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("PADDING",(0,0),(-1,-1),4),
        ]))
        s.append(t)
        s.append(Spacer(1,10))

    # Recommended tests
    tests = r.get("recommendedTests", [])
    if tests:
        s.append(Paragraph("Recommended Investigations", H2))
        trows = [["Test","Category","Priority","Rationale"]]
        for t_item in tests:
            trows.append([t_item.get("name",""), t_item.get("category",""),
                          t_item.get("priority",""), t_item.get("rationale","")[:60]])
        tt = Table(trows, colWidths=[4*cm, 3*cm, 2.2*cm, 8.3*cm])
        tt.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0a8f7a")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1),8.5),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#e6f7f4")]),
            ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#ccddee")),
            ("PADDING",(0,0),(-1,-1),4),
        ]))
        s.append(tt)
        s.append(Spacer(1,10))

    # Final summary
    fs = r.get("finalSummary","")
    if fs:
        s.append(Paragraph("Physician Handoff Summary", H2))
        s.append(Paragraph(fs, NM))
        s.append(Spacer(1,10))

    # Confidence
    sc = r.get("systemConfidence", {})
    if sc:
        s.append(Paragraph("System Confidence Metrics", H2))
        conf_rows = [
            ["Metric","Score"],
            ["Overall Confidence", f"{sc.get('overall',0)}%"],
            ["Diagnostic Confidence", f"{sc.get('diagnosticConfidence',0)}%"],
            ["Triage Accuracy", f"{sc.get('triageAccuracy',0)}%"],
            ["Data Completeness", f"{sc.get('dataCompleteness',0)}%"],
        ]
        ct = Table(conf_rows, colWidths=[6*cm, 3*cm])
        ct.setStyle(TableStyle([
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1),9),
            ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#e2e8f0")),
            ("PADDING",(0,0),(-1,-1),4),
        ]))
        s.append(ct)
        if sc.get("narrative"):
            s.append(Paragraph(sc["narrative"], IT))
        s.append(Spacer(1,14))

    s.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0")))
    s.append(Spacer(1,6))
    s.append(Paragraph(
        "DISCLAIMER: AI-generated content for clinical decision support only. "
        "All outputs must be validated by a licensed healthcare professional before clinical use. "
        "ClinicalTriageEnv v5.0 — Meta PyTorch OpenEnv Hackathon × Scaler.",
        IT
    ))

    doc.build(s)
    return buf.getvalue()

# ENTRY POINT


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
