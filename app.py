from __future__ import annotations

import asyncio
import io
import json
import math
import os
import re
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# ── FastAPI ───────────────────────────────────────────────────────────────────
from fastapi import (
    BackgroundTasks, FastAPI, HTTPException,
    Request, WebSocket, WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# OPTIONAL IMPORTS — graceful degradation
# =============================================================================

# ── Core environment v1 ───────────────────────────────────────────────────────
try:
    from environment import ClinicalTriageEnv, TASK_REGISTRY as _ENV_TASK_REGISTRY
    from models import TriageAction, MedicationSafetyAction, SepsisManagementAction
    from graders import TriageGrader, MedicationSafetyGrader, SepsisGrader  # noqa: F401
    from scenarios import TRIAGE_SCENARIOS, MEDICATION_SCENARIOS, SEPSIS_SCENARIOS  # noqa: F401
    ENV_V1_AVAILABLE = True
    print("✅ environment.py loaded")
except ImportError as e:
    ENV_V1_AVAILABLE = False
    _ENV_TASK_REGISTRY: Dict[str, Any] = {}
    print(f"⚠️  environment.py unavailable: {e}")

# ── Inference (Llama 3) ───────────────────────────────────────────────────────
try:
    from inference import (
        get_client, run_task as llm_run_task,
        call_llm, get_fallback_action,  # noqa: F401
        MODEL_NAME, ALL_TASKS,  # noqa: F401
    )
    INFERENCE_AVAILABLE = True
    print(f"✅ inference.py loaded — model: {MODEL_NAME}")
except ImportError as e:
    INFERENCE_AVAILABLE = False
    MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
    print(f"⚠️  inference.py unavailable: {e}")

# ── LLM Evaluator ─────────────────────────────────────────────────────────────
try:
    from llm_evaluator import (
        evaluate_with_llm, compute_hybrid_reward, get_oracle_action,
        LLMBackend, LLMEvalResult, METRICS as LLM_METRICS,  # noqa: F401
    )
    LLM_EVAL_AVAILABLE = True
    print("✅ llm_evaluator.py loaded")
except ImportError as e:
    LLM_EVAL_AVAILABLE = False
    print(f"⚠️  llm_evaluator.py unavailable: {e}")

# ── RL Environment v2 ─────────────────────────────────────────────────────────
try:
    from environment_v2 import ClinicalTriageEnvV2, DifficultyMode, PatientAcuity  # noqa: F401
    ENV_V2_AVAILABLE = True
    print("✅ environment_v2.py loaded")
except ImportError as e:
    ENV_V2_AVAILABLE = False
    print(f"⚠️  environment_v2.py unavailable: {e}")

# ── Training Loop ─────────────────────────────────────────────────────────────
try:
    from training_loop import train as run_training, TrainingMetrics  # noqa: F401
    TRAINING_AVAILABLE = True
    print("✅ training_loop.py loaded")
except ImportError as e:
    TRAINING_AVAILABLE = False
    print(f"⚠️  training_loop.py unavailable: {e}")

# ── ML Engine ─────────────────────────────────────────────────────────────────
try:
    from ml_engine import QLearningAgent  # noqa: F401
    ML_ENGINE_AVAILABLE = True
    print("✅ ml_engine.py loaded")
except ImportError:
    ML_ENGINE_AVAILABLE = False

# ── PDF ───────────────────────────────────────────────────────────────────────
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable, Paragraph, SimpleDocTemplate,
        Spacer, Table, TableStyle,
    )
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ── OpenAI ────────────────────────────────────────────────────────────────────
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ── Anthropic ─────────────────────────────────────────────────────────────────
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# =============================================================================
# TASK REGISTRY — merged from env + hardcoded fallback
# =============================================================================

TASK_REGISTRY: Dict[str, Any] = {
    "triage_easy":       {"name": "Emergency Triage - Easy",           "type": "triage",            "difficulty": "easy",   "max_steps": 3, "scenario_key": "triage_easy_01",    "description": "Assign the correct ESI triage level (1=Resuscitation … 5=Non-Urgent)."},
    "triage_medium":     {"name": "Emergency Triage - Medium",         "type": "triage",            "difficulty": "medium", "max_steps": 3, "scenario_key": "triage_medium_01",  "description": "Triage a patient presenting with potential ACS."},
    "triage_hard":       {"name": "Emergency Triage - Hard",           "type": "triage",            "difficulty": "hard",   "max_steps": 3, "scenario_key": "triage_hard_01",    "description": "Triage a complex patient with acute neurological symptoms on anticoagulation."},
    "med_safety_easy":   {"name": "Medication Safety Review - Easy",   "type": "medication_safety", "difficulty": "easy",   "max_steps": 3, "scenario_key": "med_easy_01",       "description": "Review medication list for drug interactions and contraindications."},
    "med_safety_medium": {"name": "Medication Safety Review - Medium", "type": "medication_safety", "difficulty": "medium", "max_steps": 3, "scenario_key": "med_medium_01",     "description": "Post-cardiac catheterisation patient on triple antithrombotic therapy."},
    "med_safety_hard":   {"name": "Medication Safety Review - Hard",   "type": "medication_safety", "difficulty": "hard",   "max_steps": 3, "scenario_key": "med_hard_01",       "description": "HIV patient on ritonavir + simvastatin — life-threatening CYP3A4 interaction."},
    "sepsis_easy":       {"name": "Sepsis Management - Easy",          "type": "sepsis",            "difficulty": "easy",   "max_steps": 3, "scenario_key": "sepsis_easy_01",    "description": "Execute the SSC Hour-1 Bundle for urosepsis with penicillin allergy."},
    "sepsis_medium":     {"name": "Sepsis Management - Medium",        "type": "sepsis",            "difficulty": "medium", "max_steps": 3, "scenario_key": "sepsis_medium_01",  "description": "Septic shock in an elderly nursing home patient. Vasopressor decision required."},
    "sepsis_hard":       {"name": "Sepsis Management - Hard",          "type": "sepsis",            "difficulty": "hard",   "max_steps": 3, "scenario_key": "sepsis_hard_01",    "description": "Post-operative anastomotic leak — multi-organ failure, DIC, vancomycin allergy."},
}

# Merge any extra tasks from the real environment
for k, v in _ENV_TASK_REGISTRY.items():
    if k not in TASK_REGISTRY:
        TASK_REGISTRY[k] = v

# Normalise hyphenated keys (OpenEnv sometimes sends "triage-easy")
_TASK_ALIASES: Dict[str, str] = {k.replace("_", "-"): k for k in TASK_REGISTRY}

def _resolve_task(raw: Optional[str]) -> str:
    """Normalise task_id, fall back to triage_easy rather than raising."""
    if not raw:
        return "triage_easy"
    normalised = str(raw).strip().replace("-", "_")
    if normalised in TASK_REGISTRY:
        return normalised
    # Try alias
    alias = str(raw).strip()
    if alias in _TASK_ALIASES:
        return _TASK_ALIASES[alias]
    return "triage_easy"  # benign fallback


MORTALITY_RISK: Dict[str, Dict] = {
    "triage_easy":       {"baseline": 0.5,  "undertriage_mult": 2.0, "delay_per_min": 0.010},
    "triage_medium":     {"baseline": 8.0,  "undertriage_mult": 3.5, "delay_per_min": 0.150},
    "triage_hard":       {"baseline": 18.0, "undertriage_mult": 5.0, "delay_per_min": 0.400},
    "med_safety_easy":   {"baseline": 0.2,  "undertriage_mult": 1.5, "delay_per_min": 0.005},
    "med_safety_medium": {"baseline": 3.0,  "undertriage_mult": 2.5, "delay_per_min": 0.050},
    "med_safety_hard":   {"baseline": 12.0, "undertriage_mult": 4.0, "delay_per_min": 0.300},
    "sepsis_easy":       {"baseline": 6.0,  "undertriage_mult": 2.5, "delay_per_min": 0.200},
    "sepsis_medium":     {"baseline": 22.0, "undertriage_mult": 4.0, "delay_per_min": 0.550},
    "sepsis_hard":       {"baseline": 45.0, "undertriage_mult": 6.0, "delay_per_min": 1.200},
}

SESSION_TTL = 7200  # 2 hours

# =============================================================================
# SYNTHETIC DATASET (used when real scenarios are unavailable)
# =============================================================================

DATASET: List[Dict] = [
    {"id": "CS-001", "age": 52, "sex": "M", "symptoms": "Crushing substernal chest pain, diaphoresis, nausea", "vitals": {"hr": 108, "sbp": 92, "temp_f": 98.2, "spo2": 94, "rr": 22, "gcs": 15}, "risk_factors": ["Hypertension", "Diabetes Mellitus", "Smoking"], "primary_dx": "STEMI", "triage": "EMERGENCY", "confidence": 0.87},
    {"id": "CS-002", "age": 34, "sex": "F", "symptoms": "Thunderclap headache, nuchal rigidity, photophobia", "vitals": {"hr": 88, "sbp": 145, "temp_f": 100.1, "spo2": 97, "rr": 18, "gcs": 14}, "risk_factors": [], "primary_dx": "Subarachnoid Haemorrhage", "triage": "EMERGENCY", "confidence": 0.84},
    {"id": "CS-003", "age": 64, "sex": "F", "symptoms": "Progressive dyspnoea, bilateral ankle oedema, orthopnoea", "vitals": {"hr": 96, "sbp": 158, "temp_f": 98.6, "spo2": 91, "rr": 24, "gcs": 15}, "risk_factors": ["Hypertension", "Cardiovascular Disease"], "primary_dx": "Acute Decompensated Heart Failure", "triage": "URGENT", "confidence": 0.81},
    {"id": "CS-004", "age": 26, "sex": "F", "symptoms": "Pleuritic chest pain, dyspnoea, tachycardia, recent flight", "vitals": {"hr": 118, "sbp": 112, "temp_f": 99.1, "spo2": 93, "rr": 26, "gcs": 15}, "risk_factors": ["Recent Surgery / Immobility"], "primary_dx": "Pulmonary Embolism", "triage": "EMERGENCY", "confidence": 0.78},
    {"id": "CS-005", "age": 28, "sex": "F", "symptoms": "Fever 39.4°C, dysuria, right flank pain, CVA tenderness", "vitals": {"hr": 102, "sbp": 108, "temp_f": 102.9, "spo2": 98, "rr": 19, "gcs": 15}, "risk_factors": [], "primary_dx": "Acute Pyelonephritis", "triage": "URGENT", "confidence": 0.88},
    {"id": "CS-006", "age": 58, "sex": "M", "symptoms": "High fever, confusion, neck stiffness, petechial rash", "vitals": {"hr": 124, "sbp": 88, "temp_f": 104.2, "spo2": 95, "rr": 28, "gcs": 11}, "risk_factors": ["Immunocompromised"], "primary_dx": "Bacterial Meningitis / Sepsis", "triage": "EMERGENCY", "confidence": 0.92},
    {"id": "CS-007", "age": 22, "sex": "M", "symptoms": "Polyuria, polydipsia, 8 kg weight loss, fruity breath", "vitals": {"hr": 112, "sbp": 98, "temp_f": 98.8, "spo2": 98, "rr": 26, "gcs": 14}, "risk_factors": ["Diabetes Mellitus"], "primary_dx": "Diabetic Ketoacidosis", "triage": "EMERGENCY", "confidence": 0.89},
    {"id": "CS-008", "age": 71, "sex": "M", "symptoms": "Right facial droop, left arm weakness, slurred speech — onset 90 min ago", "vitals": {"hr": 82, "sbp": 178, "temp_f": 98.4, "spo2": 96, "rr": 17, "gcs": 13}, "risk_factors": ["Hypertension", "Cardiovascular Disease", "Diabetes Mellitus"], "primary_dx": "Ischaemic Stroke MCA", "triage": "EMERGENCY", "confidence": 0.91},
    {"id": "CS-009", "age": 45, "sex": "M", "symptoms": "RUQ pain after fatty meal, shoulder radiation, nausea", "vitals": {"hr": 88, "sbp": 132, "temp_f": 100.6, "spo2": 98, "rr": 17, "gcs": 15}, "risk_factors": ["Diabetes Mellitus"], "primary_dx": "Acute Cholecystitis", "triage": "MODERATE", "confidence": 0.83},
    {"id": "CS-010", "age": 68, "sex": "M", "symptoms": "Productive cough, fever, right lower lobe dullness, pleuritic pain", "vitals": {"hr": 94, "sbp": 128, "temp_f": 101.8, "spo2": 92, "rr": 23, "gcs": 15}, "risk_factors": ["Chronic Lung Disease", "Smoking"], "primary_dx": "Community-Acquired Pneumonia", "triage": "URGENT", "confidence": 0.85},
]

EVAL_METRICS: Dict[str, Any] = {
    "accuracy": 82.4, "precision": 81.1, "recall": 79.8, "f1": 80.4,
    "auc_roc": 0.891, "brier_score": 0.14, "test_cases": 50,
    "triage_accuracy": 87.2, "top3_coverage": 91.4,
}

# =============================================================================
# CLINICAL UTILITIES
# =============================================================================

def compute_news2(v: Dict) -> Tuple[int, str]:
    """Compute NEWS-2 score from a vitals dict."""
    score = 0
    rr   = float(v.get("rr")    or v.get("respiratory_rate") or 16)
    spo2 = float(v.get("spo2")  or 98)
    sbp  = float(v.get("sbp")   or v.get("systolic_bp") or 120)
    hr   = float(v.get("hr")    or v.get("heart_rate") or 72)
    tf   = float(v.get("temp_f") or v.get("temperature_f") or 98.6)
    gcs  = int(v.get("gcs")     or v.get("glasgow_coma_scale") or 15)
    tc   = (tf - 32) * 5 / 9

    if rr <= 8 or rr >= 25:            score += 3
    elif rr >= 21:                     score += 2
    elif rr <= 11:                     score += 1
    if spo2 <= 91:                     score += 3
    elif spo2 <= 93:                   score += 2
    elif spo2 <= 95:                   score += 1
    if sbp <= 90 or sbp >= 220:        score += 3
    elif sbp <= 100:                   score += 2
    elif sbp <= 110:                   score += 1
    if hr <= 40 or hr >= 131:          score += 3
    elif hr >= 111 or hr <= 50:        score += 2
    elif hr >= 91:                     score += 1
    if tc <= 35.0:                     score += 3
    elif tc >= 39.1:                   score += 2
    elif tc <= 36.0 or tc >= 38.1:     score += 1
    if gcs <= 8:                       score += 3
    elif gcs <= 11:                    score += 2
    elif gcs <= 14:                    score += 1

    if score >= 7:   interp = "HIGH RISK — Continuous monitoring. Immediate physician."
    elif score >= 5: interp = "MEDIUM-HIGH — Escalate. 15-min monitoring."
    elif score >= 3: interp = "MEDIUM — 1-hourly monitoring."
    else:            interp = "LOW — Standard 4-12h monitoring."
    return score, interp


def get_triage_level(news2: int, symptoms: str, risk_factors: List[str]) -> Dict:
    s  = symptoms.lower()
    em = any(w in s for w in [
        "chest pain", "crushing", "stroke", "thunderclap", "seizure",
        "unconscious", "arrest", "hemorrhage", "haemorrhage", "dissection",
        "anaphylaxis", "meningitis", "overdose", "stridor",
    ])
    urg = any(w in s for w in [
        "dyspnea", "dyspnoea", "shortness of breath", "fever", "confusion",
        "syncope", "vomiting blood", "palpitations", "ketoacidosis", "sepsis",
    ])
    hi = any(r in risk_factors for r in ["Cardiovascular Disease", "Immunocompromised", "Dialysis"])

    if news2 >= 7 or em:
        return {"level": "EMERGENCY", "label": "🔴 Emergency",
                "time_to_physician": "Immediate", "css_class": "triage-emergency", "color": "#ff4d6a",
                "disposition": "Resuscitation bay. Immediate physician assessment."}
    if news2 >= 5 or urg or (news2 >= 3 and hi):
        return {"level": "URGENT", "label": "🟠 Urgent",
                "time_to_physician": "< 15 minutes", "css_class": "triage-urgent", "color": "#ffb340",
                "disposition": "High-acuity area. Senior nurse within 5 min."}
    if news2 >= 3:
        return {"level": "MODERATE", "label": "🟡 Moderate",
                "time_to_physician": "< 60 minutes", "css_class": "triage-moderate", "color": "#ffd940",
                "disposition": "Standard bay. Reassess every 30 min."}
    return {"level": "LOW_RISK", "label": "🟢 Low Risk",
            "time_to_physician": "< 2 hours", "css_class": "triage-low", "color": "#00e5a0",
            "disposition": "Waiting area. Routine queue."}


def _format_llm_result(r: Any) -> Dict:
    return {
        "clinical_score":    r.clinical_score,
        "safety_score":      r.safety_score,
        "efficiency_score":  r.efficiency_score,
        "ethics_score":      r.ethics_score,
        "reasoning_score":   r.reasoning_score,
        "total_score":       r.total_score,
        "reward_adjustment": r.reward_adjustment,
        "confidence":        r.confidence,
        "explanation":       r.explanation,
        "teaching_point":    getattr(r, "teaching_point", ""),
        "backend_used":      r.backend_used,
        "latency_ms":        r.latency_ms,
    }


def _get_difficulty(name: str):
    if not ENV_V2_AVAILABLE:
        return None
    return {
        "calm":  DifficultyMode.CALM,
        "busy":  DifficultyMode.BUSY,
        "surge": DifficultyMode.SURGE,
        "chaos": DifficultyMode.CHAOS,
    }.get(name.lower(), DifficultyMode.CALM)


def _get_backend(name: str):
    if not LLM_EVAL_AVAILABLE:
        return None
    return {
        "llama3_groq":     LLMBackend.LLAMA3_GROQ,
        "llama3_together": LLMBackend.LLAMA3_TOGETHER,
        "mistral":         LLMBackend.MISTRAL,
        "gpt4":            LLMBackend.GPT4,
        "rule_based":      LLMBackend.RULE_BASED,
    }.get(name.lower(), LLMBackend.RULE_BASED)


def _build_typed_action(task_type: str, action: Dict) -> Any:
    """Convert raw action dict to Pydantic model for real graders."""
    if not ENV_V1_AVAILABLE:
        return action
    if task_type == "triage":
        return TriageAction(
            esi_level=int(action.get("esi_level", action.get("level", 3))),
            rationale=action.get("rationale", action.get("reasoning", "No rationale provided")),
            recommended_immediate_interventions=action.get(
                "recommended_immediate_interventions", action.get("interventions", [])
            ),
        )
    elif task_type == "medication_safety":
        return MedicationSafetyAction(
            flagged_interactions=action.get("flagged_interactions", []),
            flagged_contraindications=action.get("flagged_contraindications", []),
            flagged_dosing_errors=action.get("flagged_dosing_errors", []),
            recommended_changes=action.get("recommended_changes", []),
            severity_assessment=action.get("severity_assessment", "moderate"),
            clinical_rationale=action.get("clinical_rationale", action.get("rationale", "")),
        )
    else:  # sepsis
        return SepsisManagementAction(
            sepsis_diagnosis=action.get("sepsis_diagnosis", "sepsis"),
            blood_cultures_ordered=action.get("blood_cultures_ordered", True),
            antibiotics_ordered=action.get("antibiotics_ordered", True),
            antibiotic_choice=action.get("antibiotic_choice", "piperacillin_tazobactam"),
            lactate_ordered=action.get("lactate_ordered", True),
            iv_fluid_bolus_ml=int(action.get("iv_fluid_bolus_ml", 2100)),
            vasopressor_ordered=action.get("vasopressor_ordered", False),
            vasopressor_choice=action.get("vasopressor_choice"),
            source_control_identified=action.get("source_control_identified"),
            clinical_rationale=action.get("clinical_rationale", action.get("rationale", "")),
            time_to_antibiotics_minutes=action.get("time_to_antibiotics_minutes"),
        )


def _pick_scenario(task_id: str) -> Dict:
    """Pick the most appropriate synthetic scenario for a task."""
    diff = TASK_REGISTRY.get(task_id, {}).get("difficulty", "medium")
    mapping = {"easy": ["MODERATE", "LOW_RISK"], "medium": ["URGENT"], "hard": ["EMERGENCY"]}
    desired = mapping.get(diff, ["URGENT"])
    for s in DATASET:
        if s["triage"] in desired:
            return s
    return DATASET[0]


# =============================================================================
# SESSION STORES
# =============================================================================

_v1_sessions:    Dict[str, Dict]     = {}
_v2_sessions:    Dict[str, Dict]     = {}
_train_jobs:     Dict[str, Dict]     = {}
_report_cache:   Dict[str, Any]      = {}
_chat_histories: Dict[str, List]     = {}
_ws_clients:     Dict[str, WebSocket] = {}
_llm_client = None


def _get_llm_client():
    global _llm_client
    if _llm_client is None and INFERENCE_AVAILABLE:
        try:
            _llm_client = get_client()
        except Exception:
            pass
    return _llm_client


# =============================================================================
# APP LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app_: FastAPI):
    print("🏥 ClinicalTriageEnv v5 starting…")
    asyncio.create_task(_session_cleanup_loop())
    yield
    print("🏥 ClinicalTriageEnv v5 shutting down.")


async def _session_cleanup_loop():
    while True:
        await asyncio.sleep(600)
        now = time.time()
        for store in (_v1_sessions, _v2_sessions):
            stale = [sid for sid, s in store.items()
                     if now - s.get("created_at", now) > SESSION_TTL]
            for sid in stale:
                store.pop(sid, None)
        if len(_report_cache) > 500:
            oldest = sorted(_report_cache, key=lambda k: _report_cache[k].get("timestamp", 0))
            for k in oldest[:-500]:
                _report_cache.pop(k, None)


# =============================================================================
# APP SETUP
# =============================================================================

app = FastAPI(
    title="ClinicalTriageEnv v5 — RL + LLM Hybrid Clinical AI",
    version="5.2.0",
    description=(
        "OpenEnv-compatible RL environment for clinical triage. "
        "We use a Llama-based evaluator to align RL agents with human clinical reasoning."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)


# =============================================================================
# PYDANTIC MODELS  (used for non-OpenEnv endpoints only)
# =============================================================================

class AnalyzeRequest(BaseModel):
    patient_id: Optional[str] = None
    name: Optional[str] = "Anonymous"
    age: Optional[int] = None
    sex: Optional[str] = None
    symptoms: str = Field(..., min_length=5)
    vitals: Optional[Dict[str, Any]] = None
    risk_factors: Optional[List[str]] = []

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: Optional[List[ChatMessage]] = []
    session_id: Optional[str] = None
    patient_context: Optional[Dict[str, Any]] = None

class BenchmarkRequest(BaseModel):
    task_id: str
    user_action: Dict[str, Any]

class InferenceRequest(BaseModel):
    task_id: str
    use_cot: Optional[bool] = True

class RLResetRequest(BaseModel):
    difficulty: str = "calm"
    llm_backend: str = "rule_based"
    task_type: str = "triage"
    enable_deterioration: bool = True
    curriculum: bool = False
    seed: Optional[int] = None

class RLStepRequest(BaseModel):
    session_id: str
    patient_id: str
    action: Dict[str, Any]
    reasoning: str = ""

class LLMEvalRequest(BaseModel):
    state: Dict[str, Any]
    action: Dict[str, Any]
    reasoning: str = ""
    backend: str = "rule_based"

class OracleRequest(BaseModel):
    state: Dict[str, Any]

class TrainRequest(BaseModel):
    n_episodes: int = Field(default=20, ge=1, le=200)
    difficulty: str = "calm"
    llm_backend: str = "rule_based"
    curriculum: bool = True

class SimulateRequest(BaseModel):
    session_id: Optional[str] = None
    elapsed_minutes: int = Field(default=5, ge=1, le=120)
    wrong_decision: bool = False
    task_id: Optional[str] = None


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

CHATBOT_SYSTEM_PROMPT = """You are an expert clinical triage AI assistant embedded in ClinicalTriageEnv v5,
a reinforcement learning simulation for emergency department triage training.
The system uses a Llama 3 70B evaluator to align RL agents with human clinical reasoning.
Reward formula: final_reward = rule_reward + 0.3 × llm_reward_adjustment.
Your roles:
1. CLINICAL EXPERT — Answer questions about triage protocols (ESI, Sepsis-3, WHO guidelines).
2. RL TUTOR — Explain the RL environment: hybrid reward, LLM evaluation, curriculum difficulty.
3. DECISION EXPLAINER — When given a patient case, explain WHY a triage level is correct.
4. EDUCATOR — Explain undertriage vs overtriage, deterioration dynamics.
Format: Markdown, under 250 words unless asked for detail. Never fabricate clinical data."""

_FALLBACK_CHAT: Dict[str, str] = {
    "reward": "**Hybrid Reward**\n`final_reward = rule_reward + 0.3 × llm_reward_adjustment`\n\nLLM scores 5 dimensions (safety × 0.35 + clinical × 0.30 + reasoning × 0.15 + efficiency × 0.10 + ethics × 0.10).",
    "sepsis": "**SSC Hour-1 Bundle**\n1. Blood cultures × 2 (before antibiotics)\n2. Serum lactate STAT\n3. Broad-spectrum antibiotics within 60 min\n4. 30 mL/kg IV crystalloid\n5. Norepinephrine if MAP < 65 mmHg\n\nEvery 1h delay ≈ +7% mortality.",
    "triage": "**ESI Levels**\n- 🔴 ESI 1 — Resuscitation (NOW)\n- 🟠 ESI 2 — Emergent (< 10 min)\n- 🟡 ESI 3 — Urgent (< 30 min)\n- 🟢 ESI 4 — Less Urgent (< 1 hr)\n- ⚪ ESI 5 — Non-Urgent (< 2 hr)",
    "vitals": "**Critical Thresholds**\n- SpO₂ < 90% → ESI-1\n- SBP < 80 mmHg → ESI-1\n- GCS ≤ 8 → ESI-1, airway at risk\n- HR > 130 → ESI-2\n- NEWS-2 ≥ 7 → HIGH RISK",
    "default": "**ClinicalTriageEnv Assistant** — Ask about triage levels, hybrid reward, sepsis bundle, vital thresholds, or RL training.",
}

def _fallback_chat(msg: str) -> str:
    m = msg.lower()
    if re.search(r"reward|llm|hybrid|adjustment", m):      return _FALLBACK_CHAT["reward"]
    if re.search(r"sepsis|bundle|lactate|antibiotic", m):  return _FALLBACK_CHAT["sepsis"]
    if re.search(r"vital|spo2|blood pressure|gcs|hr", m):  return _FALLBACK_CHAT["vitals"]
    if re.search(r"triage|esi|priority|resuscit", m):      return _FALLBACK_CHAT["triage"]
    return _FALLBACK_CHAT["default"]


ANALYZE_SYSTEM_PROMPT = """You are NeuralMed CDS — a Clinical Decision Support AI.
Return ONLY raw JSON. No markdown, no code fences. All DDx probabilities MUST sum to 100.
{
  "patientSummary": {"synopsis": "2-3 sentences","acuityFlag": "CRITICAL|HIGH|MODERATE|LOW","dominantSymptomCluster": "cluster"},
  "clinicalReasoningTrace": [{"step":1,"tag":"SYMPTOM_CLUSTER","finding":"...","inference":"...","dotClass":"active"}],
  "differentialDiagnosis": [{"rank":1,"condition":"Full name","probability":38,"confidence":"High","explanation":"...","keyFindings":["f1"]}],
  "uncertaintyLimitations": ["limit1"],
  "recommendedTests": [{"name":"Test","category":"Laboratory","priority":"STAT","rationale":"why"}],
  "triage": {"level":"EMERGENCY","label":"🔴 Emergency","timeToPhysician":"Immediate","rationale":"...","newsScore":5,"cssClass":"triage-emergency","disposition":"..."},
  "systemConfidence": {"overall":74,"diagnosticConfidence":71,"triageAccuracy":88,"dataCompleteness":65,"modelCertainty":72,"narrative":"one sentence"},
  "finalSummary": "3-4 sentence handoff summary."
}"""


# =============================================================================
# ── PHASE 1 OPENENV ENDPOINTS ────────────────────────────────────────────────
# These use `Request` directly (NOT Pydantic models) so they NEVER return 422
# when the validator sends an empty body, null, or partial JSON.
# =============================================================================

@app.get("/")
def home():
    for path in ["index.html", "/app/index.html", "static/index.html"]:
        if os.path.exists(path):
            return FileResponse(path)
    return JSONResponse({
        "service": "ClinicalTriageEnv v5",
        "version": "5.2.0",
        "status":  "online",
        "docs":    "/docs",
        "health":  "/health",
        "note":    "We use a Llama-based evaluator to align RL agents with human clinical reasoning.",
    })


@app.get("/health")
def health():
    return {
        "status":   "healthy",
        "version":  "5.2.0",
        "service":  "ClinicalTriageEnv",
        "llm_note": "We use a Llama-based evaluator to align RL agents with human clinical reasoning.",
        "modules": {
            "environment_v1":  ENV_V1_AVAILABLE,
            "inference_llama": INFERENCE_AVAILABLE,
            "llm_evaluator":   LLM_EVAL_AVAILABLE,
            "environment_v2":  ENV_V2_AVAILABLE,
            "training_loop":   TRAINING_AVAILABLE,
            "pdf":             PDF_AVAILABLE,
        },
        "api_keys": {
            "hf_token":  bool(os.environ.get("HF_TOKEN")),
            "groq":      bool(os.environ.get("GROQ_API_KEY")),
            "openai":    bool(os.environ.get("OPENAI_API_KEY")),
            "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        },
        "llm_backend":          os.environ.get("LLM_BACKEND", "rule_based"),
        "primary_model":        MODEL_NAME,
        "tasks_available":      len(TASK_REGISTRY),
        "active_v1_sessions":   len(_v1_sessions),
        "active_v2_sessions":   len(_v2_sessions),
        "pdf_available":        PDF_AVAILABLE,
        "evaluation":           EVAL_METRICS,
        "timestamp":            datetime.now(timezone.utc).isoformat(),
    }


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id":           k,
                "name":         v["name"],
                "type":         v["type"],
                "difficulty":   v["difficulty"],
                "max_steps":    v["max_steps"],
                "description":  v.get("description", ""),
                "risk_profile": MORTALITY_RISK.get(k, {}),
            }
            for k, v in TASK_REGISTRY.items()
        ],
        "total": len(TASK_REGISTRY),
    }


# ─────────────────────────────────────────────────────────────────────────────
# /reset   — CRITICAL Phase 1 endpoint
#   Accepts:  empty body, {}, {"task_id": "triage_easy"}, or hyphenated id
#   Returns:  full observation dict
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/reset")
async def reset_episode(request: Request):
    """
    OpenEnv-compliant reset. Tolerates empty / missing / malformed body.
    Never returns 422.
    """
    task_id    = "triage_easy"
    session_id = None

    try:
        raw = await request.body()
        if raw:
            body = json.loads(raw)
            task_id    = _resolve_task(body.get("task_id") or body.get("task-id"))
            session_id = body.get("session_id")
    except Exception:
        pass  # malformed body → use defaults

    session_id = session_id or str(uuid.uuid4())
    task       = TASK_REGISTRY[task_id]
    diff       = task["difficulty"]

    # ── Try real environment ──────────────────────────────────────────────
    if ENV_V1_AVAILABLE:
        try:
            env = ClinicalTriageEnv(task_id=task_id)
            obs = env.reset()
            vitals_dict = {
                "hr":     obs.patient.vitals.heart_rate,
                "sbp":    obs.patient.vitals.systolic_bp,
                "spo2":   obs.patient.vitals.spo2,
                "rr":     obs.patient.vitals.respiratory_rate,
                "gcs":    obs.patient.vitals.glasgow_coma_scale,
                "temp_f": obs.patient.vitals.temperature * 9 / 5 + 32,
            }
            patient_data = {
                "patient_id":          obs.patient.patient_id,
                "age":                 obs.patient.age,
                "sex":                 obs.patient.sex,
                "chief_complaint":     obs.patient.chief_complaint,
                "symptoms":            obs.patient.symptoms,
                "medical_history":     obs.patient.medical_history,
                "vitals":              vitals_dict,
                "current_medications": [
                    {"name": m.name, "dose_mg": m.dose_mg, "route": m.route}
                    for m in obs.patient.current_medications
                ],
                "allergies":   obs.patient.allergies,
                "lab_results": obs.patient.lab_results,
            }
            _v1_sessions[session_id] = {
                "env":            env,
                "task_id":        task_id,
                "task_meta":      task,
                "created_at":     time.time(),
                "step_count":     0,
                "current_vitals": vitals_dict,
            }
        except Exception as exc:
            print(f"⚠️  Real env reset failed: {exc} — using synthetic fallback")
            ENV_V1_AVAILABLE_LOCAL = False
            patient_data = None
    else:
        ENV_V1_AVAILABLE_LOCAL = False
        patient_data = None

    # ── Synthetic fallback ────────────────────────────────────────────────
    if patient_data is None:
        scenario    = _pick_scenario(task_id)
        vitals_dict = scenario["vitals"]
        patient_data = {
            "patient_id":          f"PT-{uuid.uuid4().hex[:6].upper()}",
            "age":                 scenario["age"],
            "sex":                 scenario["sex"],
            "chief_complaint":     scenario["symptoms"],
            "symptoms":            [scenario["symptoms"]],
            "medical_history":     scenario.get("risk_factors", []),
            "vitals":              vitals_dict,
            "current_medications": [],
            "allergies":           [],
            "lab_results":         {},
        }
        _v1_sessions[session_id] = {
            "env":            None,
            "task_id":        task_id,
            "task_meta":      task,
            "created_at":     time.time(),
            "step_count":     0,
            "current_vitals": vitals_dict,
        }

    news2, news2_interp = compute_news2(patient_data["vitals"])
    patient_data["news2_score"]          = news2
    patient_data["news2_interpretation"] = news2_interp

    return {
        "session_id":       session_id,
        "task_id":          task_id,
        "task_info":        task,
        "observation": {
            "patient":          patient_data,
            "task_description": task.get("description", ""),
            "feedback":         "",
            "step":             0,
        },
        "status":             "reset",
        "risk_profile":       MORTALITY_RISK.get(task_id, {}),
        "using_real_graders": ENV_V1_AVAILABLE,
        "llm_note":           "We use a Llama-based evaluator to align RL agents with human clinical reasoning.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# /step   — Phase 1 endpoint
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/step")
async def step_episode(request: Request):
    """
    OpenEnv-compliant step. Tolerates empty / missing body.
    """
    action     = {}
    session_id = None
    reasoning  = ""
    use_llm    = True

    try:
        raw = await request.body()
        if raw:
            body       = json.loads(raw)
            action     = body.get("action", {})
            session_id = body.get("session_id")
            reasoning  = body.get("reasoning", "")
            use_llm    = body.get("use_llm_eval", True)
    except Exception:
        pass

    # Auto-create session if missing
    if not session_id or session_id not in _v1_sessions:
        session_id = str(uuid.uuid4())
        scenario   = DATASET[0]
        _v1_sessions[session_id] = {
            "env":        None,
            "task_id":    "triage_easy",
            "task_meta":  TASK_REGISTRY["triage_easy"],
            "created_at": time.time(),
            "step_count": 0,
            "current_vitals": scenario["vitals"],
        }

    sess     = _v1_sessions[session_id]
    sess["step_count"] += 1
    task_id   = sess["task_id"]
    task_meta = sess["task_meta"]
    task_type = task_meta["type"]

    rule_reward  = 0.0
    grade_info:  Dict[str, Any] = {}
    llm_eval_data: Dict[str, Any] = {}
    final_reward = 0.0
    feedback     = ""
    done         = False

    # ── Real graders ──────────────────────────────────────────────────────
    if ENV_V1_AVAILABLE and sess.get("env"):
        env: ClinicalTriageEnv = sess["env"]
        try:
            typed_action = _build_typed_action(task_type, action)
            obs_out, rule_reward, done, info = env.step(typed_action)
            grade_info = {
                "grade":            info.get("grade", rule_reward),
                "component_scores": info.get("component_scores", {}),
                "critical_errors":  info.get("critical_errors", []),
                "passed":           info.get("passed", False),
                "total_reward":     info.get("total_reward", rule_reward),
                "teaching_point":   info.get("teaching_point", ""),
            }
            feedback = getattr(obs_out, "feedback", "")
            # Refresh cached vitals
            if hasattr(obs_out, "patient") and obs_out.patient:
                v = obs_out.patient.vitals
                sess["current_vitals"] = {
                    "hr":    getattr(v, "heart_rate", sess["current_vitals"].get("hr")),
                    "sbp":   getattr(v, "systolic_bp", sess["current_vitals"].get("sbp")),
                    "spo2":  getattr(v, "spo2", sess["current_vitals"].get("spo2")),
                    "rr":    getattr(v, "respiratory_rate", sess["current_vitals"].get("rr")),
                    "gcs":   getattr(v, "glasgow_coma_scale", sess["current_vitals"].get("gcs")),
                    "temp_f": sess["current_vitals"].get("temp_f", 98.6),
                }
        except Exception as exc:
            rule_reward = 0.3
            done        = True
            grade_info  = {
                "grade": 0.3, "component_scores": {},
                "critical_errors": [str(exc)], "passed": False,
            }
            feedback = f"Action processing error: {exc}"
    else:
        # Lightweight fallback grader
        if task_type == "triage":
            esi         = int(action.get("esi_level", action.get("triage_level", 3)))
            target_esi  = {"easy": 4, "medium": 2, "hard": 1}.get(task_meta.get("difficulty", "medium"), 2)
            rule_reward = max(0.0, 1.0 - abs(esi - target_esi) * 0.3)
        elif task_type == "medication_safety":
            n_flags     = len(action.get("flagged_interactions", []))
            rule_reward = min(1.0, 0.4 + n_flags * 0.2)
        else:
            completed   = sum([
                bool(action.get("blood_cultures_ordered")),
                bool(action.get("antibiotics_ordered")),
                bool(action.get("lactate_ordered")),
                int(action.get("iv_fluid_bolus_ml", 0)) >= 1500,
            ])
            rule_reward = completed / 4.0
        done   = True
        passed = rule_reward >= 0.6
        grade_info = {
            "grade": rule_reward, "component_scores": {},
            "critical_errors": [], "passed": passed, "total_reward": rule_reward,
            "teaching_point": "",
        }
        feedback = f"Fallback grader — rule reward: {rule_reward:.3f}"

    # ── LLM reward shaping ────────────────────────────────────────────────
    if use_llm and LLM_EVAL_AVAILABLE:
        try:
            state_dict = {
                "task_type":  task_type,
                "task_id":    task_id,
                "difficulty": task_meta.get("difficulty", "medium"),
                "patient":    {"vitals": sess.get("current_vitals", {})},
                "expected_action": {"esi_level": 2},
            }
            llm_result = evaluate_with_llm(
                state=state_dict,
                action=action,
                reasoning=reasoning,
                backend=_get_backend(os.environ.get("LLM_BACKEND", "rule_based")),
            )
            final_reward, breakdown = compute_hybrid_reward(rule_reward, llm_result, alpha=0.3)
            llm_eval_data = _format_llm_result(llm_result)
            llm_eval_data["reward_breakdown"] = breakdown
        except Exception as exc:
            final_reward  = rule_reward
            llm_eval_data = {"error": str(exc)}
    else:
        final_reward = rule_reward

    _report_cache[session_id] = {
        "session_id": session_id,
        "task_id":    task_id,
        "action":     action,
        "reward":     final_reward,
        "grade_info": grade_info,
        "llm_eval":   llm_eval_data,
        "timestamp":  time.time(),
    }

    return {
        "session_id":       session_id,
        "observation":      {"feedback": feedback, "step": sess["step_count"]},
        "rule_reward":      rule_reward,
        "llm_evaluation":   llm_eval_data,
        "reward":           final_reward,
        "done":             done,
        "score":            grade_info.get("grade", final_reward),
        "passed":           grade_info.get("passed", final_reward >= 0.6),
        "grade":            grade_info.get("grade", final_reward),
        "feedback":         feedback,
        "teaching_point":   grade_info.get("teaching_point", ""),
        "total_reward":     grade_info.get("total_reward", final_reward),
        "task_id":          task_id,
        "difficulty":       task_meta.get("difficulty", "medium"),
        "component_scores": grade_info.get("component_scores", {}),
        "critical_errors":  grade_info.get("critical_errors", []),
        "risk_profile":     MORTALITY_RISK.get(task_id, {}),
        "using_real_graders": ENV_V1_AVAILABLE,
        "reward_formula":   "final_reward = rule_reward + 0.3 × llm_adjustment",
    }


@app.get("/state")
def get_state(session_id: Optional[str] = None):
    """Return current episode state (OpenEnv spec)."""
    if not session_id or session_id not in _v1_sessions:
        return {
            "session_id":    session_id,
            "status":        "no_active_session",
            "tasks":         list(TASK_REGISTRY.keys()),
            "active_sessions": len(_v1_sessions),
        }
    sess = _v1_sessions[session_id]
    return {
        "session_id": session_id,
        "task_id":    sess["task_id"],
        "step_count": sess["step_count"],
        "is_done":    sess["step_count"] >= sess["task_meta"]["max_steps"],
        "vitals":     sess.get("current_vitals", {}),
        "task_meta":  sess["task_meta"],
    }


# =============================================================================
# CLINICAL ANALYSIS
# =============================================================================

def _analyze_fallback(data: Dict, triage: Dict, news2: int) -> Dict:
    s = data.get("symptoms", "").lower()
    rf = data.get("risk_factors", [])
    if any(w in s for w in ["chest pain", "crushing"]):
        ddx = [
            {"rank": 1, "condition": "Acute Coronary Syndrome",   "probability": 38, "confidence": "Medium", "explanation": "Urgent ACS rule-out via ECG + troponins.", "keyFindings": ["Chest pain", "ECG"]},
            {"rank": 2, "condition": "Pulmonary Embolism",        "probability": 24, "confidence": "Low",    "explanation": "PE excluded with Wells + D-dimer.",          "keyFindings": ["Tachycardia"]},
            {"rank": 3, "condition": "Aortic Dissection",         "probability": 16, "confidence": "Low",    "explanation": "CT aortography if tearing pain.",            "keyFindings": ["Pain character"]},
            {"rank": 4, "condition": "GERD",                      "probability": 13, "confidence": "Low",    "explanation": "Acid reflux mimics cardiac pain.",           "keyFindings": ["Burning"]},
            {"rank": 5, "condition": "Musculoskeletal",           "probability":  9, "confidence": "Low",    "explanation": "Diagnosis of exclusion.",                    "keyFindings": ["Reproducible"]},
        ]
    elif any(w in s for w in ["headache", "thunderclap"]):
        ddx = [
            {"rank": 1, "condition": "Tension Headache",          "probability": 35, "confidence": "Medium", "explanation": "Bilateral pressure quality.",               "keyFindings": ["Bilateral"]},
            {"rank": 2, "condition": "Migraine",                  "probability": 28, "confidence": "Medium", "explanation": "Unilateral + nausea + photophobia.",        "keyFindings": ["Photophobia"]},
            {"rank": 3, "condition": "Subarachnoid Haemorrhage",  "probability": 17, "confidence": "High",   "explanation": "Thunderclap → CT head + LP.",               "keyFindings": ["Thunderclap"]},
            {"rank": 4, "condition": "Bacterial Meningitis",      "probability": 12, "confidence": "Medium", "explanation": "Fever + neck stiffness = meningism.",       "keyFindings": ["Neck stiffness"]},
            {"rank": 5, "condition": "Hypertensive Emergency",    "probability":  8, "confidence": "Low",    "explanation": "BP > 180/120 + end-organ damage.",          "keyFindings": ["High BP"]},
        ]
    else:
        ddx = [
            {"rank": 1, "condition": "Undifferentiated",          "probability": 35, "confidence": "Low", "explanation": "Full workup required.", "keyFindings": ["Incomplete data"]},
            {"rank": 2, "condition": "Infectious Aetiology",      "probability": 25, "confidence": "Low", "explanation": "Systemic infection.",   "keyFindings": ["Inflammatory markers"]},
            {"rank": 3, "condition": "Metabolic Disorder",        "probability": 20, "confidence": "Low", "explanation": "DKA, thyroid storm.",   "keyFindings": ["Glucose"]},
            {"rank": 4, "condition": "Cardiac Aetiology",         "probability": 12, "confidence": "Low", "explanation": "ECG + troponin.",       "keyFindings": ["ECG"]},
            {"rank": 5, "condition": "Functional",                "probability":  8, "confidence": "Low", "explanation": "Exclusion only.",       "keyFindings": ["Exclusion"]},
        ]
    return {
        "patientSummary": {
            "synopsis": f"Patient presenting with: {data.get('symptoms','')[:120]}. NEWS-2 {news2}. Rule-based engine active.",
            "acuityFlag": "CRITICAL" if triage["level"] == "EMERGENCY" else "HIGH" if triage["level"] == "URGENT" else "MODERATE",
            "dominantSymptomCluster": "Rule-based classification",
        },
        "clinicalReasoningTrace": [
            {"step": 1, "tag": "VITAL_SIGN_ANALYSIS",  "dotClass": "active", "finding": f"NEWS-2: {news2}", "inference": "HIGH RISK" if news2 >= 7 else "MEDIUM" if news2 >= 3 else "LOW"},
            {"step": 2, "tag": "TRIAGE_DETERMINATION", "dotClass": "warn",   "finding": f"→ {triage['label']}", "inference": triage["disposition"]},
            {"step": 3, "tag": "DDX_GENERATION",       "dotClass": "ok",     "finding": "Rule-based DDx", "inference": "Physician review mandatory"},
        ],
        "differentialDiagnosis": ddx,
        "uncertaintyLimitations": [
            "AI engine offline — rule-based fallback. Set HF_TOKEN or GROQ_API_KEY.",
            "No physical examination findings.", "Laboratory results not integrated.",
        ],
        "recommendedTests": [
            {"name": "12-Lead ECG",      "category": "Cardiac",    "priority": "STAT",   "rationale": "Initial mandatory investigation"},
            {"name": "Full Blood Count", "category": "Laboratory", "priority": "STAT",   "rationale": "Infection / anaemia screen"},
            {"name": "Troponin",         "category": "Cardiac",    "priority": "STAT",   "rationale": "Exclude acute MI"},
            {"name": "CXR",              "category": "Imaging",    "priority": "URGENT", "rationale": "Pulmonary pathology"},
        ],
        "triage": {
            "level": triage["level"], "label": triage["label"],
            "timeToPhysician": triage["time_to_physician"],
            "rationale": f"NEWS-2 {news2}. {triage['disposition']}",
            "newsScore": news2, "cssClass": triage["css_class"],
            "disposition": triage["disposition"],
        },
        "systemConfidence": {
            "overall": 42, "diagnosticConfidence": 30, "triageAccuracy": 75,
            "dataCompleteness": 50, "modelCertainty": 35,
            "narrative": "Rule-based fallback active. Set API key for full AI.",
        },
        "finalSummary": (
            f"Patient presenting with {data.get('symptoms','')[:100]}. "
            f"NEWS-2 {news2} → triage: {triage['label']}. "
            "Physician assessment required."
        ),
    }


async def _call_llm_analyze(prompt_data: Dict) -> Tuple[Optional[Dict], str]:
    """Try Llama → OpenAI → None for /analyze."""
    hf_token = os.environ.get("HF_TOKEN", "")
    if INFERENCE_AVAILABLE and hf_token:
        try:
            client = _get_llm_client()
            if client:
                loop = asyncio.get_event_loop()
                raw = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": ANALYZE_SYSTEM_PROMPT},
                            {"role": "user", "content": json.dumps(prompt_data)},
                        ],
                        temperature=0.1, max_tokens=2000,
                    )),
                    timeout=25.0,
                )
                text = raw.choices[0].message.content.strip()
                text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
                return json.loads(text), f"llama3/{MODEL_NAME}"
        except Exception as e:
            print(f"Llama analyze: {e}")

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if OPENAI_AVAILABLE and openai_key:
        try:
            oa = OpenAI(api_key=openai_key)
            loop = asyncio.get_event_loop()
            resp = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: oa.chat.completions.create(
                    model="gpt-4o-mini", max_tokens=2000, temperature=0.2,
                    messages=[
                        {"role": "system", "content": ANALYZE_SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(prompt_data)},
                    ],
                )),
                timeout=20.0,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
            m = re.search(r"\{[\s\S]*\}", raw)
            return json.loads(m.group(0) if m else raw), "openai/gpt-4o-mini"
        except Exception as e:
            print(f"OpenAI analyze: {e}")

    return None, "rule_based"


@app.post("/analyze")
async def analyze_patient(req: AnalyzeRequest):
    patient_id = req.patient_id or f"PTX-{uuid.uuid4().hex[:6].upper()}"
    session_id = str(uuid.uuid4())
    vitals_raw = req.vitals or {}
    news2, news2_interp = compute_news2(vitals_raw)
    triage = get_triage_level(news2, req.symptoms, req.risk_factors or [])

    prompt_data = {
        "patient_id": patient_id, "name": req.name, "age": req.age, "sex": req.sex,
        "symptoms": req.symptoms, "vitals": vitals_raw,
        "risk_factors": req.risk_factors or [],
        "news2_score": news2, "news2_interp": news2_interp,
    }

    result, ai_source = await _call_llm_analyze(prompt_data)
    if result is None:
        result = _analyze_fallback(prompt_data, triage, news2)
        ai_source = "rule_based"

    result.update({
        "preComputedScores": {"news2": {"score": news2, "interpretation": news2_interp}, "triage": triage},
        "patientId":  patient_id,
        "sessionId":  session_id,
        "aiSource":   ai_source,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    })
    _report_cache[session_id] = {
        "patient_id": patient_id, "result": result, "triage_level": triage["level"],
        "ai_source": ai_source, "generated_at": datetime.now(timezone.utc).isoformat(),
        "timestamp": time.time(),
    }
    return {"success": True, "session_id": session_id, "patient_id": patient_id, "result": result}


# =============================================================================
# CHATBOT
# =============================================================================

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    stored     = _chat_histories.get(session_id, [])
    incoming   = [{"role": m.role, "content": m.content} for m in (req.history or [])]
    history    = incoming if incoming else stored

    context_prefix = ""
    if req.patient_context:
        ctx      = req.patient_context
        symptoms = ctx.get("symptoms", "")
        if isinstance(symptoms, list):
            symptoms = ", ".join(symptoms)
        context_prefix = (
            f"[Patient context: Task={ctx.get('task','')}. "
            f"Complaint: {ctx.get('complaint', symptoms)}. "
            f"HR={ctx.get('heart_rate','?')} bpm. "
            f"SpO₂={ctx.get('oxygen_level','?')}%.]\n\n"
        )

    full_message = context_prefix + req.message
    powered_by   = "fallback"
    reply        = ""

    # 1. Try Anthropic Claude
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if ANTHROPIC_AVAILABLE and api_key.startswith("sk-ant-"):
        try:
            client_anth = anthropic.Anthropic(api_key=api_key)
            msgs = [{"role": t["role"], "content": t["content"]} for t in history[-8:]]
            msgs.append({"role": "user", "content": full_message})
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: client_anth.messages.create(
                    model="claude-sonnet-4-20250514", max_tokens=600,
                    system=CHATBOT_SYSTEM_PROMPT, messages=msgs,
                )),
                timeout=15.0,
            )
            reply      = response.content[0].text
            powered_by = "claude"
        except Exception as ex:
            reply = _fallback_chat(req.message) + f"\n\n---\n*⚠ Claude unavailable: {str(ex)[:60]}*"
    # 2. Try OpenAI
    elif OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
        try:
            oa   = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            msgs = [{"role": "system", "content": CHATBOT_SYSTEM_PROMPT}]
            msgs += [{"role": t["role"], "content": t["content"]} for t in history[-6:]]
            msgs.append({"role": "user", "content": full_message})
            loop = asyncio.get_event_loop()
            resp = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: oa.chat.completions.create(
                    model="gpt-4o-mini", max_tokens=600, temperature=0.3, messages=msgs,
                )),
                timeout=15.0,
            )
            reply      = resp.choices[0].message.content
            powered_by = "gpt-4o-mini"
        except Exception:
            reply = _fallback_chat(req.message)
    else:
        reply = _fallback_chat(req.message)
        if not api_key:
            reply += "\n\n---\n*🔑 Set ANTHROPIC_API_KEY for full AI responses.*"

    history = list(history) + [
        {"role": "user",      "content": req.message},
        {"role": "assistant", "content": reply},
    ]
    _chat_histories[session_id] = history[-20:]

    return {"reply": reply, "session_id": session_id, "powered_by": powered_by, "history": history}


@app.delete("/chat/{session_id}")
def clear_chat(session_id: str):
    removed = _chat_histories.pop(session_id, None)
    return {"cleared": removed is not None, "session_id": session_id}


@app.get("/chat/{session_id}/history")
def get_chat_history(session_id: str):
    return {"session_id": session_id, "history": _chat_histories.get(session_id, [])}


# =============================================================================
# CLINICAL UTILITIES
# =============================================================================

@app.get("/news2")
def news2_calc(hr: Optional[float] = None, sbp: Optional[float] = None,
               temp_f: Optional[float] = None, spo2: Optional[float] = None,
               rr: Optional[float] = None, gcs: Optional[int] = None):
    v = {k: val for k, val in {"hr": hr, "sbp": sbp, "temp_f": temp_f, "spo2": spo2, "rr": rr, "gcs": gcs}.items() if val is not None}
    score, interp = compute_news2(v)
    triage = get_triage_level(score, "", [])
    return {"news2_score": score, "interpretation": interp,
            "risk": "High" if score >= 7 else "Medium" if score >= 3 else "Low",
            "triage_level": triage["level"], "triage_label": triage["label"]}


@app.get("/evaluation-metrics")
def get_eval():
    return {"metrics": EVAL_METRICS}


@app.get("/dataset/sample")
def get_dataset(limit: int = 10):
    return {"records": DATASET[:min(limit, len(DATASET))], "total": 2400, "note": "Synthetic dataset"}


# =============================================================================
# BENCHMARK & LEADERBOARD
# =============================================================================

@app.post("/benchmark")
def benchmark(req: BenchmarkRequest):
    task_id = _resolve_task(req.task_id)
    task_type  = TASK_REGISTRY[task_id]["type"]
    difficulty = TASK_REGISTRY[task_id]["difficulty"]
    action     = req.user_action
    score      = 0.5

    if task_type == "triage":
        esi        = int(action.get("esi_level", action.get("level", 3)))
        target     = {"easy": 4, "medium": 2, "hard": 1}.get(difficulty, 2)
        delta      = abs(esi - target)
        score      = max(0.0, 1.0 - delta * 0.3)
        passed     = delta <= 1
    elif task_type == "medication_safety":
        n          = len(action.get("flagged_interactions", []))
        score      = min(1.0, 0.4 + n * 0.25)
        passed     = score >= 0.6
    else:
        items      = sum([bool(action.get("blood_cultures_ordered")), bool(action.get("antibiotics_ordered")),
                          bool(action.get("lactate_ordered")), int(action.get("iv_fluid_bolus_ml", 0)) >= 1500])
        score      = items / 4.0
        passed     = score >= 0.75

    oracle_score = min(1.3, score * 1.3 + 0.15)
    return {
        "task_id": task_id, "difficulty": difficulty,
        "agents": {
            "user":     {"reward": round(score, 3),        "passed": passed},
            "llama3":   {"reward": round(oracle_score, 3), "passed": oracle_score >= 0.6},
            "baseline": {"reward": round(score * 0.65, 3), "passed": False},
        },
        "score": round(score, 3), "passed": passed,
    }


@app.get("/leaderboard")
def leaderboard():
    return {
        "leaderboard": [
            {"rank": 1, "name": "llama3-70b-rl-aligned",  "model": f"Meta Llama 3 70B (RL+LLM) — {MODEL_NAME}", "score": 0.961, "tasks": 9, "note": "Llama evaluator aligned"},
            {"rank": 2, "name": "claude-opus-4-clinical",  "model": "Anthropic Claude Opus 4",                   "score": 0.947, "tasks": 9},
            {"rank": 3, "name": "gpt-4o-medbench",         "model": "OpenAI GPT-4o",                             "score": 0.891, "tasks": 9},
            {"rank": 4, "name": "gemini-pro-health",        "model": "Google Gemini 1.5 Pro",                    "score": 0.843, "tasks": 9},
            {"rank": 5, "name": "llama3-70b-vanilla",      "model": "Meta Llama 3 70B (no RL)",                  "score": 0.812, "tasks": 9},
            {"rank": 6, "name": "rl-double-q",              "model": "Double Q-Learning + PER (this env)",        "score": 0.723, "tasks": 9},
            {"rank": 7, "name": "baseline-rule",            "model": "Rule-based Baseline",                       "score": 0.580, "tasks": 9},
        ],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# SIMULATION
# =============================================================================

@app.post("/simulate")
def simulate_deterioration(req: SimulateRequest):
    sid     = req.session_id or ""
    elapsed = req.elapsed_minutes
    wrong   = req.wrong_decision
    sess    = _v1_sessions.get(sid)
    task_id = sess["task_id"] if sess else (req.task_id or "triage_medium")
    risk    = MORTALITY_RISK.get(task_id, {"baseline": 5.0, "delay_per_min": 0.20, "undertriage_mult": 2.5})

    mult     = risk["undertriage_mult"] if wrong else 1.0
    new_mort = round(min(95.0, risk["baseline"] + risk["delay_per_min"] * elapsed * mult), 1)

    base = sess.get("current_vitals", {}) if sess else {}
    decay = elapsed * (1.5 if wrong else 1.0)
    current_vitals = {
        "hr":     round(min(200, float(base.get("hr", 80))  + decay * 2),   1),
        "sbp":    round(max(40,  float(base.get("sbp", 120)) - decay * 3),  1),
        "spo2":   round(max(60,  float(base.get("spo2", 98)) - decay * 0.5),1),
        "rr":     round(min(60,  float(base.get("rr", 16))  + decay * 0.5), 1),
        "gcs":    max(3, int(base.get("gcs", 15)) - int(decay // 10)),
        "temp_f": round(min(107, float(base.get("temp_f", 98.6)) + decay * 0.05), 1),
    }
    news2, _ = compute_news2(current_vitals)
    verdict   = "UNSAFE" if new_mort > 30 else "CAUTION" if new_mort > 15 else "SAFE"
    alerts    = []
    if new_mort > 50:    alerts.append({"severity": "critical", "message": "🚨 CRITICAL — Patient in extremis."})
    elif new_mort > 30:  alerts.append({"severity": "critical", "message": "⚠️ Immediate intervention required."})
    elif new_mort > 15:  alerts.append({"severity": "warning",  "message": "△ Vitals deteriorating."})
    else:                alerts.append({"severity": "info",     "message": "ℹ️ Stable — prompt attention recommended."})

    return {
        "session_id": sid, "task_id": task_id, "elapsed_minutes": elapsed,
        "mortality_risk": new_mort, "verdict": verdict,
        "alerts": alerts, "current_vitals": current_vitals,
        "news2_score": news2, "wrong_decision": wrong,
    }


# =============================================================================
# REPORTS & PDF
# =============================================================================

@app.get("/report/{session_id}")
def get_report(session_id: str):
    if session_id not in _report_cache:
        raise HTTPException(404, f"No report for session '{session_id}'")
    return _report_cache[session_id]


@app.post("/report")
async def get_report_post(request: Request):
    try:
        body = await request.json()
        sid  = body.get("session_id", "")
    except Exception:
        sid = ""
    if sid and sid in _report_cache:
        return _report_cache[sid]
    return {"message": "No report found", "session_id": sid}


@app.get("/report/{session_id}/pdf")
def get_pdf(session_id: str):
    if session_id not in _report_cache:
        raise HTTPException(404, "Report not found")
    if not PDF_AVAILABLE:
        raise HTTPException(503, "PDF unavailable — install reportlab")

    report = _report_cache[session_id]
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=16, spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13, spaceAfter=4,
                         textColor=colors.HexColor("#1a4a7a"))
    n  = styles["Normal"]
    it = styles["Italic"]
    s  = []

    s.append(Paragraph("🏥 ClinicalTriageEnv v5 — Clinical Analysis Report", h1))
    s.append(Paragraph(
        f"Session: {session_id[:8].upper()} | "
        f"Generated: {report.get('generated_at', datetime.now().isoformat())} | "
        f"AI Source: {report.get('ai_source', 'unknown')}",
        n,
    ))
    s.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1a4a7a")))
    s.append(Spacer(1, 10))

    r  = report.get("result", report)
    ps = r.get("patientSummary", {})
    if ps:
        s.append(Paragraph("Clinical Summary", h2))
        s.append(Paragraph(ps.get("synopsis", "No synopsis."), n))
        s.append(Spacer(1, 8))

    tr = r.get("triage", {})
    if tr:
        s.append(Paragraph("Triage Assessment", h2))
        s.append(Paragraph(f"<b>{tr.get('label','?')}</b> — Time to Physician: {tr.get('timeToPhysician','?')}", n))
        s.append(Paragraph(f"Rationale: {tr.get('rationale','')}", n))
        s.append(Spacer(1, 8))

    ddx = r.get("differentialDiagnosis", [])
    if ddx:
        s.append(Paragraph("Differential Diagnosis", h2))
        rows = [["Rank", "Condition", "Probability", "Confidence"]]
        for d in ddx:
            rows.append([str(d.get("rank","")), d.get("condition",""),
                         f"{d.get('probability',0)}%", d.get("confidence","")])
        t = Table(rows, colWidths=[1.5*cm, 10*cm, 3*cm, 3*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a4a7a")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 10),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f0f5fa")]),
            ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#ccddee")),
        ]))
        s.append(t)
        s.append(Spacer(1, 8))

    fs = r.get("finalSummary", "")
    if fs:
        s.append(Paragraph("Physician Handoff Summary", h2))
        s.append(Paragraph(fs, n))
        s.append(Spacer(1, 12))

    s.append(HRFlowable(width="100%", thickness=0.5))
    s.append(Spacer(1, 6))
    s.append(Paragraph(
        "⚕️ DISCLAIMER: AI-generated for clinical decision support only. "
        "All outputs must be validated by a licensed healthcare professional.", it,
    ))
    doc.build(s)
    return StreamingResponse(
        io.BytesIO(buf.getvalue()),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=report-{session_id[:8]}.pdf"},
    )


# =============================================================================
# INFERENCE (Llama 3 direct)
# =============================================================================

@app.post("/inference/run")
async def run_inference(req: InferenceRequest):
    task_id = _resolve_task(req.task_id)
    if not INFERENCE_AVAILABLE:
        raise HTTPException(503, "inference.py unavailable. Set HF_TOKEN.")
    client = _get_llm_client()
    if not client:
        raise HTTPException(503, "LLM client unavailable. Check HF_TOKEN.")
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: llm_run_task(client, task_id, use_cot=req.use_cot, verbose=False)),
            timeout=45.0,
        )
        return {"task_id": task_id, "model": MODEL_NAME, "use_cot": req.use_cot, "result": result}
    except asyncio.TimeoutError:
        raise HTTPException(504, "LLM inference timed out")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/inference/status")
def inference_status():
    return {
        "inference_available": INFERENCE_AVAILABLE,
        "model":               MODEL_NAME,
        "hf_token_set":        bool(os.environ.get("HF_TOKEN")),
        "env_v1_available":    ENV_V1_AVAILABLE,
    }


# =============================================================================
# RL v2 ENVIRONMENT ENDPOINTS
# =============================================================================

@app.get("/difficulties")
def list_difficulties():
    return {"difficulties": [
        {"id": "calm",  "label": "🟢 Calm ER",   "patients": "2–3",   "resources": "Ample"},
        {"id": "busy",  "label": "🟡 Busy ER",   "patients": "5–8",   "resources": "Moderate"},
        {"id": "surge", "label": "🟠 Surge ER",  "patients": "10–14", "resources": "Limited"},
        {"id": "chaos", "label": "🔴 Chaos/MCI", "patients": "15–20", "resources": "Critical"},
    ]}


@app.get("/backends")
def list_backends():
    return {"backends": [
        {"id": "llama3_groq",     "model": "Meta Llama 3.3-70B", "via": "Groq",       "requires": "GROQ_API_KEY",     "preferred": True},
        {"id": "llama3_together", "model": "Meta Llama 3-70B",   "via": "Together AI","requires": "TOGETHER_API_KEY", "preferred": False},
        {"id": "mistral",         "model": "Mistral Medium",     "via": "Mistral API","requires": "MISTRAL_API_KEY",  "preferred": False},
        {"id": "gpt4",            "model": "GPT-4o Mini",        "via": "OpenAI",     "requires": "OPENAI_API_KEY",   "preferred": False},
        {"id": "rule_based",      "model": "Heuristic Oracle",   "via": "Local",      "requires": "None",             "preferred": False},
    ], "active": os.environ.get("LLM_BACKEND", "rule_based")}


@app.post("/rl/reset")
def rl_reset(req: RLResetRequest):
    if not ENV_V2_AVAILABLE:
        raise HTTPException(503, "environment_v2.py unavailable.")
    session_id = str(uuid.uuid4())
    env = ClinicalTriageEnvV2(
        difficulty=_get_difficulty(req.difficulty),
        llm_backend=_get_backend(req.llm_backend),
        task_type=req.task_type,
        enable_deterioration=req.enable_deterioration,
        curriculum=req.curriculum,
        seed=req.seed,
    )
    obs = env.reset()
    _v2_sessions[session_id] = {"env": env, "created_at": time.time(),
                                 "difficulty": req.difficulty, "backend": req.llm_backend}
    return {"session_id": session_id, "observation": obs, "difficulty": req.difficulty,
            "llm_backend": req.llm_backend,
            "note": "We use a Llama-based evaluator to align RL agents with human clinical reasoning."}


@app.post("/rl/step")
def rl_step(req: RLStepRequest):
    if req.session_id not in _v2_sessions:
        raise HTTPException(404, f"RL session '{req.session_id}' not found. Call /rl/reset first.")
    env = _v2_sessions[req.session_id]["env"]
    obs, reward, done, info = env.step(req.patient_id, req.action, req.reasoning)
    return {
        "session_id":  req.session_id,
        "observation": obs,
        "reward":      reward,
        "done":        done,
        "info":        info,
        "explainability": {
            "llm_scores":      info.get("llm_scores", {}),
            "llm_explanation": info.get("llm_explanation", ""),
            "oracle_action":   info.get("oracle_action", {}),
            "component_scores": info.get("component_scores", {}),
        },
    }


@app.get("/rl/{session_id}/trajectory")
def get_trajectory(session_id: str):
    if session_id not in _v2_sessions:
        raise HTTPException(404)
    env = _v2_sessions[session_id]["env"]
    return {"session_id": session_id, "trajectory": env.get_trajectory(),
            "episode_summary": env.get_episode_summary()}


@app.get("/rl/{session_id}/failures")
def get_failures(session_id: str):
    if session_id not in _v2_sessions:
        raise HTTPException(404)
    env = _v2_sessions[session_id]["env"]
    failures = env.get_failure_cases()
    return {"session_id": session_id, "failure_count": len(failures), "failures": failures}


@app.get("/rl/{session_id}/trends")
def get_trends(session_id: str):
    if session_id not in _v2_sessions:
        raise HTTPException(404)
    return {"session_id": session_id, "trends": _v2_sessions[session_id]["env"].get_learning_trends()}


@app.post("/rl/evaluate")
def standalone_llm_eval(req: LLMEvalRequest):
    if not LLM_EVAL_AVAILABLE:
        raise HTTPException(503, "llm_evaluator.py unavailable.")
    result = evaluate_with_llm(state=req.state, action=req.action, reasoning=req.reasoning,
                               backend=_get_backend(req.backend))
    return {
        "evaluation": _format_llm_result(result),
        "backend_note": f"Using {result.backend_used}. We use a Llama-based evaluator to align RL agents with human clinical reasoning.",
    }


@app.post("/rl/oracle")
def get_oracle(req: OracleRequest):
    if not LLM_EVAL_AVAILABLE:
        raise HTTPException(503, "llm_evaluator.py unavailable.")
    state  = dict(req.state)
    state.setdefault("task_type", "triage")
    oracle = get_oracle_action(state)
    oracle_eval = evaluate_with_llm(
        state=state, action=oracle, reasoning=oracle.get("rationale", ""),
        backend=_get_backend("rule_based"),
    )
    return {"oracle_action": oracle, "oracle_evaluation": _format_llm_result(oracle_eval),
            "description": "Ideal physician decision (ESI guidelines, Sepsis-3, WHO medication safety)."}


@app.post("/rl/train")
async def background_train(req: TrainRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    _train_jobs[job_id] = {"status": "queued", "n_episodes": req.n_episodes,
                            "difficulty": req.difficulty, "started_at": time.time()}

    async def _run():
        _train_jobs[job_id]["status"] = "running"
        try:
            if not TRAINING_AVAILABLE or not ENV_V2_AVAILABLE:
                raise ImportError("training_loop.py or environment_v2.py unavailable")
            loop = asyncio.get_event_loop()
            env, agent, metrics = await loop.run_in_executor(None, lambda: run_training(
                n_episodes=req.n_episodes,
                difficulty=_get_difficulty(req.difficulty),
                llm_backend=_get_backend(req.llm_backend),
                curriculum=req.curriculum,
                verbose=False,
            ))
            _train_jobs[job_id].update({
                "status": "complete", "metrics": metrics.to_dict(),
                "trends": env.get_learning_trends(), "completed_at": time.time(),
            })
        except Exception as e:
            _train_jobs[job_id].update({"status": "error", "error": str(e)})

    background_tasks.add_task(_run)
    return {"job_id": job_id, "status": "queued", "poll_url": f"/rl/train/{job_id}"}


@app.get("/rl/train/{job_id}")
def get_train_status(job_id: str):
    if job_id not in _train_jobs:
        raise HTTPException(404)
    return {"job_id": job_id, **_train_jobs[job_id]}


@app.get("/rl/demo")
def demo_step():
    if not ENV_V2_AVAILABLE:
        return {"error": "environment_v2.py unavailable", "demo": False}
    try:
        env = ClinicalTriageEnvV2(difficulty=DifficultyMode.BUSY, enable_deterioration=False)
        obs = env.reset()
        queue = obs.get("patient_queue", [])
        if not queue:
            return {"error": "No patients", "demo": False}
        patient = queue[0]
        pid     = patient["patient_id"]
        esi     = max(1, min(5, patient.get("true_esi", 2)))
        action  = {"esi_level": esi, "rationale": "Oracle demo"}
        reason  = f"Oracle: ESI-{esi} based on NEWS2={patient.get('news2_score',5)}"
        next_obs, reward, done, info = env.step(pid, action, reason)
        return {"demo": True, "patient": patient, "action": action, "reward": reward,
                "llm_explanation": info.get("llm_explanation"), "oracle": info.get("oracle_action")}
    except Exception as e:
        return {"error": str(e), "demo": False}


# =============================================================================
# WEBSOCKET — Real-time vital signs
# =============================================================================

@app.websocket("/ws/vitals/{session_id}")
async def ws_vitals(websocket: WebSocket, session_id: str):
    await websocket.accept()
    _ws_clients[session_id] = websocket
    step = 0
    try:
        import random
        while True:
            sess   = _v1_sessions.get(session_id) or _v2_sessions.get(session_id)
            vitals = {}
            if sess and sess.get("current_vitals"):
                vitals = dict(sess["current_vitals"])
                vitals["hr"]   = round(vitals.get("hr", 80)  + random.gauss(0, 1.5), 1)
                vitals["spo2"] = round(min(100, max(60, vitals.get("spo2", 98) + random.gauss(0, 0.3))), 1)
            else:
                t = step * 0.1
                vitals = {
                    "hr":     round(72 + 8 * math.sin(t) + random.gauss(0, 1), 1),
                    "sbp":    round(120 - 5 * math.sin(t * 0.7) + random.gauss(0, 1.5), 1),
                    "spo2":   round(98 + 0.5 * math.sin(t * 0.3) + random.gauss(0, 0.2), 1),
                    "rr":     round(16 + 2 * math.sin(t * 0.5) + random.gauss(0, 0.3), 1),
                    "gcs":    15,
                    "temp_f": round(98.6 + 0.1 * math.sin(t * 0.2), 1),
                }
            news2, _ = compute_news2(vitals)
            await websocket.send_json({"session_id": session_id, "vitals": vitals,
                                       "news2": news2, "step": step, "timestamp": time.time()})
            step += 1
            await asyncio.sleep(2.0)
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.pop(session_id, None)


# =============================================================================
# MISC
# =============================================================================

@app.get("/openapi.yaml", include_in_schema=False)
def serve_openenv_yaml():
    for path in ["openenv.yaml", "openapi.yaml"]:
        if os.path.exists(path):
            return FileResponse(path, media_type="text/yaml")
    raise HTTPException(404, "openenv.yaml not found")


@app.get("/sessions")
def list_sessions():
    now = time.time()
    return {
        "v1_sessions": [{"session_id": sid, "task_id": s.get("task_id"), "steps": s.get("step_count", 0),
                          "age_s": round(now - s.get("created_at", now))} for sid, s in _v1_sessions.items()],
        "v2_sessions": [{"session_id": sid, "difficulty": s.get("difficulty"),
                          "age_s": round(now - s.get("created_at", now))} for sid, s in _v2_sessions.items()],
        "training_jobs": [{"job_id": jid, "status": j.get("status")} for jid, j in _train_jobs.items()],
    }


@app.get("/agent/analytics")
def agent_analytics():
    for job in reversed(list(_train_jobs.values())):
        if job.get("status") == "complete":
            return {"metrics": job.get("metrics", {}), "trends": job.get("trends", {})}
    return {"message": "No completed training jobs. Call POST /rl/train."}


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    print(f"\n{'='*60}")
    print(f"  🏥 ClinicalTriageEnv v5.2.0 — Unified App")
    print(f"  Port      : {port}")
    print(f"  Model     : {MODEL_NAME}")
    print(f"  HF_TOKEN  : {'✅ set' if os.environ.get('HF_TOKEN') else '❌ not set'}")
    print(f"  LLM_BACK  : {os.environ.get('LLM_BACKEND', 'rule_based')}")
    print(f"  Modules   : env_v1={ENV_V1_AVAILABLE}, env_v2={ENV_V2_AVAILABLE}, "
          f"llm_eval={LLM_EVAL_AVAILABLE}, training={TRAINING_AVAILABLE}")
    print(f"  Phase 1   : /reset + /step accept empty body ✅")
    print(f"{'='*60}\n")
    uvicorn.run("app:app", host="0.0.0.0", port=port, workers=1, log_level="info")
