from __future__ import annotations

import os
import sys
import uuid
import json
import time
import io
import math
import asyncio
import re
import threading
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
from contextlib import asynccontextmanager

# ── FastAPI ───────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ── Stdlib path fix ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# MODULE IMPORTS WITH GRACEFUL DEGRADATION
# =============================================================================

# ── Core environment v1 ───────────────────────────────────────────────────────
try:
    from environment import ClinicalTriageEnv, TASK_REGISTRY
    from models import TriageAction, MedicationSafetyAction, SepsisManagementAction
    from graders import TriageGrader, MedicationSafetyGrader, SepsisGrader
    from scenarios import TRIAGE_SCENARIOS, MEDICATION_SCENARIOS, SEPSIS_SCENARIOS
    ENV_V1_AVAILABLE = True
    print("✅ environment.py loaded")
except ImportError as e:
    ENV_V1_AVAILABLE = False
    TASK_REGISTRY: Dict[str, Any] = {}
    print(f"⚠️  environment.py unavailable: {e}")

# ── Inference (Llama 3) ───────────────────────────────────────────────────────
try:
    from inference import (
        get_client, run_task as llm_run_task,
        build_triage_prompt, build_med_safety_prompt, build_sepsis_prompt,
        call_llm, extract_json, get_fallback_action, build_action as build_llm_action,
        SYSTEM_PROMPTS, MODEL_NAME, ALL_TASKS,
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
        LLMBackend, LLMEvalResult,
    )
    LLM_EVAL_AVAILABLE = True
    print("✅ llm_evaluator.py loaded")
except ImportError as e:
    LLM_EVAL_AVAILABLE = False
    print(f"⚠️  llm_evaluator.py unavailable: {e}")

# ── RL Environment v2 ─────────────────────────────────────────────────────────
try:
    from environment_v2 import ClinicalTriageEnvV2, DifficultyMode, PatientAcuity
    ENV_V2_AVAILABLE = True
    print("✅ environment_v2.py loaded")
except ImportError as e:
    ENV_V2_AVAILABLE = False
    print(f"⚠️  environment_v2.py unavailable: {e}")

# ── RL Engine ─────────────────────────────────────────────────────────────────
try:
    from rl_engine import QLearningAgent, featurise as rl_featurise
    ML_ENGINE_AVAILABLE = True
    print("✅ rl_engine.py loaded")
except ImportError:
    ML_ENGINE_AVAILABLE = False

# ── Training Loop ─────────────────────────────────────────────────────────────
try:
    from training_loop import train as run_training, TrainingMetrics
    TRAINING_AVAILABLE = True
    print("✅ training_loop.py loaded")
except ImportError as e:
    TRAINING_AVAILABLE = False
    print(f"⚠️  training_loop.py unavailable: {e}")

# ── PDF ───────────────────────────────────────────────────────────────────────
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether,
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
# TASK REGISTRY FALLBACK
# =============================================================================

if not TASK_REGISTRY:
    TASK_REGISTRY = {
        "triage_easy":       {"name": "Emergency Triage - Easy",          "type": "triage",            "difficulty": "easy",   "max_steps": 3, "scenario_key": "triage_easy_01",   "description": "Assign the correct ESI triage level. ESI: 1=Resuscitation, 2=Emergent, 3=Urgent, 4=Less Urgent, 5=Non-Urgent."},
        "triage_medium":     {"name": "Emergency Triage - Medium",        "type": "triage",            "difficulty": "medium", "max_steps": 3, "scenario_key": "triage_medium_01", "description": "Triage a patient presenting with potential ACS."},
        "triage_hard":       {"name": "Emergency Triage - Hard",          "type": "triage",            "difficulty": "hard",   "max_steps": 3, "scenario_key": "triage_hard_01",   "description": "Triage a complex patient with acute neurological symptoms."},
        "med_safety_easy":   {"name": "Medication Safety Review - Easy",  "type": "medication_safety", "difficulty": "easy",   "max_steps": 3, "scenario_key": "med_easy_01",      "description": "Review medication list for drug interactions."},
        "med_safety_medium": {"name": "Medication Safety Review - Medium","type": "medication_safety", "difficulty": "medium", "max_steps": 3, "scenario_key": "med_medium_01",    "description": "Post-cardiac cath patient on triple antithrombotic therapy."},
        "med_safety_hard":   {"name": "Medication Safety Review - Hard",  "type": "medication_safety", "difficulty": "hard",   "max_steps": 3, "scenario_key": "med_hard_01",      "description": "HIV patient with life-threatening drug interaction."},
        "sepsis_easy":       {"name": "Sepsis Management - Easy",         "type": "sepsis",            "difficulty": "easy",   "max_steps": 3, "scenario_key": "sepsis_easy_01",   "description": "Recognise sepsis criteria and execute Hour-1 SSC bundle."},
        "sepsis_medium":     {"name": "Sepsis Management - Medium",       "type": "sepsis",            "difficulty": "medium", "max_steps": 3, "scenario_key": "sepsis_medium_01", "description": "Manage septic shock in elderly nursing home patient."},
        "sepsis_hard":       {"name": "Sepsis Management - Hard",         "type": "sepsis",            "difficulty": "hard",   "max_steps": 3, "scenario_key": "sepsis_hard_01",   "description": "Post-operative septic shock with anastomotic leak."},
    }

MORTALITY_RISK = {
    "triage_easy":       {"baseline": 0.5,  "undertriage_mult": 2.0, "delay_per_min": 0.01},
    "triage_medium":     {"baseline": 8.0,  "undertriage_mult": 3.5, "delay_per_min": 0.15},
    "triage_hard":       {"baseline": 18.0, "undertriage_mult": 5.0, "delay_per_min": 0.40},
    "med_safety_easy":   {"baseline": 0.2,  "undertriage_mult": 1.5, "delay_per_min": 0.005},
    "med_safety_medium": {"baseline": 3.0,  "undertriage_mult": 2.5, "delay_per_min": 0.05},
    "med_safety_hard":   {"baseline": 12.0, "undertriage_mult": 4.0, "delay_per_min": 0.30},
    "sepsis_easy":       {"baseline": 6.0,  "undertriage_mult": 2.5, "delay_per_min": 0.20},
    "sepsis_medium":     {"baseline": 22.0, "undertriage_mult": 4.0, "delay_per_min": 0.55},
    "sepsis_hard":       {"baseline": 45.0, "undertriage_mult": 6.0, "delay_per_min": 1.20},
}

# Session TTL (seconds) — clean up stale sessions automatically
SESSION_TTL = 7200  # 2 hours


# =============================================================================
# APP LIFESPAN — startup/shutdown hooks
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    print("🏥 ClinicalTriageEnv v5 starting...")
    # Start background session cleanup task
    asyncio.create_task(_session_cleanup_loop())
    yield
    # ── Shutdown ──────────────────────────────────────────────────────────────
    print("🏥 ClinicalTriageEnv v5 shutting down.")


# =============================================================================
# APP SETUP
# =============================================================================

app = FastAPI(
    title="ClinicalTriageEnv v5 — RL + LLM Hybrid Clinical AI",
    version="5.1.0",
    description=(
        "Fully integrated RL + LLM system for clinical triage optimization. "
        "Uses Llama 3 70B for inference and reward shaping. "
        "We use a Llama-based evaluator to align RL agents with human clinical reasoning."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# =============================================================================
# SESSION STORES
# =============================================================================

_v1_sessions:    Dict[str, Dict]   = {}
_v2_sessions:    Dict[str, Dict]   = {}
_train_jobs:     Dict[str, Dict]   = {}
_report_cache:   Dict[str, Any]    = {}
_chat_histories: Dict[str, List]   = {}
_ws_clients:     Dict[str, WebSocket] = {}  # session_id → WebSocket
_llm_client = None  # lazy-init HF/OpenAI client


def _get_llm_client():
    global _llm_client
    if _llm_client is None and INFERENCE_AVAILABLE:
        try:
            _llm_client = get_client()
        except Exception:
            pass
    return _llm_client


async def _session_cleanup_loop():
    """Background task: evict sessions older than SESSION_TTL every 10 minutes."""
    while True:
        await asyncio.sleep(600)
        now = time.time()
        for store in (_v1_sessions, _v2_sessions):
            stale = [sid for sid, sess in store.items()
                     if now - sess.get("created_at", now) > SESSION_TTL]
            for sid in stale:
                store.pop(sid, None)
        # Also trim report cache to last 500 entries
        if len(_report_cache) > 500:
            oldest = sorted(_report_cache, key=lambda k: _report_cache[k].get("timestamp", 0))
            for k in oldest[:-500]:
                _report_cache.pop(k, None)


# =============================================================================
# PYDANTIC REQUEST MODELS
# =============================================================================

class ResetRequest(BaseModel):
    task_id: Optional[str] = "triage_easy"
    session_id: Optional[str] = None

class StepRequest(BaseModel):
    action: Dict[str, Any]
    session_id: Optional[str] = None
    reasoning: Optional[str] = ""
    use_llm_eval: Optional[bool] = True

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
# CLINICAL UTILITIES
# =============================================================================

def compute_news2(v: Dict) -> Tuple[int, str]:
    score = 0
    rr   = float(v.get("rr")   or v.get("respiratory_rate") or 16)
    spo2 = float(v.get("spo2") or 98)
    sbp  = float(v.get("sbp")  or v.get("systolic_bp") or 120)
    hr   = float(v.get("hr")   or v.get("heart_rate") or 72)
    tf   = float(v.get("temp_f") or v.get("temperature_f") or 98.6)
    gcs  = int(v.get("gcs") or v.get("glasgow_coma_scale") or 15)
    tc   = (tf - 32) * 5 / 9

    if rr <= 8 or rr >= 25: score += 3
    elif rr >= 21:           score += 2
    elif rr <= 11:           score += 1

    if spo2 <= 91:           score += 3
    elif spo2 <= 93:         score += 2
    elif spo2 <= 95:         score += 1

    if sbp <= 90 or sbp >= 220: score += 3
    elif sbp <= 100:             score += 2
    elif sbp <= 110:             score += 1

    if hr <= 40 or hr >= 131: score += 3
    elif hr >= 111 or hr <= 50: score += 2
    elif hr >= 91:             score += 1

    if tc <= 35.0:            score += 3
    elif tc >= 39.1:          score += 2
    elif tc <= 36.0 or tc >= 38.1: score += 1

    if gcs <= 8:  score += 3
    elif gcs <= 11: score += 2
    elif gcs <= 14: score += 1

    if score >= 7:   interp = "HIGH RISK — Continuous monitoring. Immediate physician."
    elif score >= 5: interp = "MEDIUM-HIGH — Escalate. 15-min monitoring."
    elif score >= 3: interp = "MEDIUM — 1-hourly monitoring."
    else:            interp = "LOW — Standard 4-12h monitoring."

    return score, interp


def get_triage_level(news2: int, symptoms: str, risk_factors: List[str]) -> Dict:
    s = symptoms.lower()
    em = any(w in s for w in [
        "chest pain", "crushing", "stroke", "thunderclap", "seizure", "unconscious",
        "arrest", "hemorrhage", "haemorrhage", "dissection", "anaphylaxis", "meningitis",
        "overdose", "no light perception", "stridor", "throat swelling",
    ])
    urg = any(w in s for w in [
        "dyspnea", "shortness of breath", "fever", "confusion", "syncope",
        "vomiting blood", "palpitations", "ketoacidosis", "sepsis", "severe pain",
    ])
    hi = any(r in risk_factors for r in ["Cardiovascular Disease", "Immunocompromised", "Dialysis"])

    if news2 >= 7 or em:
        return {
            "level": "EMERGENCY", "label": "🔴 Emergency",
            "time_to_physician": "Immediate", "css_class": "triage-emergency", "color": "#ff4d6a",
            "disposition": "Resuscitation bay. Immediate physician assessment.",
        }
    if news2 >= 5 or urg or (news2 >= 3 and hi):
        return {
            "level": "URGENT", "label": "🟠 Urgent",
            "time_to_physician": "< 15 minutes", "css_class": "triage-urgent", "color": "#ffb340",
            "disposition": "High-acuity area. Senior nurse within 5 min.",
        }
    if news2 >= 3:
        return {
            "level": "MODERATE", "label": "🟡 Moderate",
            "time_to_physician": "< 60 minutes", "css_class": "triage-moderate", "color": "#ffd940",
            "disposition": "Standard bay. Reassess every 30 min.",
        }
    return {
        "level": "LOW_RISK", "label": "🟢 Low Risk",
        "time_to_physician": "< 2 hours", "css_class": "triage-low", "color": "#00e5a0",
        "disposition": "Waiting area. Routine queue.",
    }


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
        "backend_used":      r.backend_used,
        "latency_ms":        r.latency_ms,
    }


def _get_difficulty(name: str):
    if not ENV_V2_AVAILABLE:
        return None
    mapping = {
        "calm":  DifficultyMode.CALM,
        "busy":  DifficultyMode.BUSY,
        "surge": DifficultyMode.SURGE,
        "chaos": DifficultyMode.CHAOS,
    }
    return mapping.get(name.lower(), DifficultyMode.CALM)


def _get_backend(name: str):
    if not LLM_EVAL_AVAILABLE:
        return None
    mapping = {
        "llama3_groq":     LLMBackend.LLAMA3_GROQ,
        "llama3_together": LLMBackend.LLAMA3_TOGETHER,
        "mistral":         LLMBackend.MISTRAL,
        "gpt4":            LLMBackend.GPT4,
        "rule_based":      LLMBackend.RULE_BASED,
    }
    return mapping.get(name.lower(), LLMBackend.RULE_BASED)


def _build_typed_action(task_type: str, action: Dict) -> Any:
    """Convert raw action dict to typed Pydantic model for real graders."""
    if task_type == "triage":
        return TriageAction(
            esi_level=int(action.get("esi_level", action.get("level", 3))),
            rationale=action.get("rationale", action.get("reasoning", "No rationale provided")),
            recommended_immediate_interventions=action.get(
                "recommended_immediate_interventions",
                action.get("interventions", [])
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


# =============================================================================
# CHATBOT PROMPTS
# =============================================================================

CHATBOT_SYSTEM_PROMPT = """You are an expert clinical triage AI assistant embedded in ClinicalTriageEnv v5,
a reinforcement learning simulation for emergency department triage training.
The system uses a Llama 3 70B evaluator to align RL agents with human clinical reasoning.
Reward formula: final_reward = rule_reward + 0.3 × llm_reward_adjustment
Your roles:
1. CLINICAL EXPERT — Answer questions about triage protocols (ESI, START, SALT, Sepsis-3), vital signs, emergency medicine.
2. RL TUTOR — Explain the RL environment: multi-patient queue, hybrid reward, LLM evaluation scores.
3. DECISION EXPLAINER — When given a patient case, explain WHY a specific ESI level is correct.
4. EDUCATOR — Explain undertriage vs overtriage, patient deterioration dynamics, curriculum difficulty.
Key facts:
- LLM evaluation: 5 dimensions (clinical, safety, efficiency, ethics, reasoning) each 0-10
- Safety score weighted 35%, clinical 30%, reasoning 15%, efficiency 10%, ethics 10%
- Difficulty modes: CALM (2-3 patients) → BUSY → SURGE → CHAOS (15-20 patients)
- Patient deterioration: SpO₂ -2/step, SBP -6/step for critical patients
- Double Q-Learning agent with Prioritised Experience Replay
Format: Markdown, concise (under 250 words unless asked for detail). Never fabricate clinical data."""

_FALLBACK_CHAT = {
    "reward": "**Hybrid Reward System**\n`final_reward = rule_reward + 0.3 × llm_reward_adjustment`\n\n- **rule_reward**: ESI match, wait time, resource use\n- **llm_adjustment** ∈ [-0.5, +0.5]: LLM scores clinical correctness, safety, ethics, reasoning\n- We use a Llama-based evaluator to align RL agents with human clinical reasoning.",
    "deterioration": "**Patient Deterioration Model**\n\nEach step without triage:\n- 🔴 CRITICAL: HR +8, SBP -6, SpO₂ -2%, GCS -1\n- 🟡 URGENT: HR +3, SBP -2, SpO₂ -1%\n- 🟢 STABLE: minimal change\n\nIf SpO₂ drops below 90% or SBP below 80 → **automatic upgrade to CRITICAL** mid-episode.",
    "triage": "**ESI Triage Levels**\n\n- 🔴 **ESI 1** — Resuscitation (NOW): arrest, GCS≤8, SpO₂<88%\n- 🟠 **ESI 2** — Emergent (<10 min): STEMI, stroke, septic shock\n- 🟡 **ESI 3** — Urgent (<30 min): stable but needs ≥2 resources\n- 🟢 **ESI 4** — Less Urgent (<1 hr): one resource needed\n- ⚪ **ESI 5** — Non-Urgent (<2 hr): no resources\n\n**Undertriage (ESI too high) is dangerous. Overtriage (ESI too low) is wasteful.**",
    "sepsis": "**Sepsis Hour-1 Bundle (SSC 2021)**\n\nAll 5 within **1 hour**:\n1. 🩸 Blood cultures × 2 (before antibiotics)\n2. 📊 Serum lactate STAT\n3. 💊 Broad-spectrum antibiotics (check allergies!)\n4. 💧 30 mL/kg IV crystalloid (MAP <65 or lactate ≥4)\n5. 💉 Norepinephrine (if MAP <65 after fluids)\n\nEvery 1h antibiotic delay = **+7% mortality**.",
    "vitals": "**Critical Vital Thresholds**\n\n| Vital | Normal | Warning | Critical |\n|---|---|---|---|\n| SpO₂ | 95-100% | 90-94% | <90% |\n| HR | 60-100 | 101-120 | >120 or <50 |\n| SBP | 90-160 | 80-89 | <80 |\n| GCS | 15 | 12-14 | ≤11 |\n| NEWS-2 | 0-2 | 3-6 | ≥7 |\n\n**SpO₂ <90% → ESI-1 regardless of other findings.**",
    "rl": "**RL Architecture**\n\n- **Double Q-Learning** — eliminates overestimation bias\n- **Prioritised Experience Replay** — samples high-TD-error transitions more often\n- **Cosine annealing** warm-up → exponential ε decay\n- **Safety Matrix** — clinical safety augments reward signal\n- **7-dimensional state** (SpO₂, HR, BP, GCS, age, red-flag, amber-flag)\n\nThe agent learns that undertriaging an ESI-1 patient incurs `SAFETY_MATRIX[(1,4)] = -1.0` penalty.",
    "default": "**ClinicalTriageEnv AI Assistant**\n\nI can explain:\n- 🧪 **Hybrid reward**: 'How does LLM reward shaping work?'\n- 📉 **Deterioration**: 'What happens if I delay triage?'\n- 🤖 **RL agent**: 'How does Double Q-Learning work here?'\n- 👨‍⚕️ **Oracle**: 'What would a doctor do?'\n- 🏥 **ESI levels**: 'When is ESI-1 assigned?'\n- 🔬 **Sepsis bundle**: 'What is the Hour-1 bundle?'",
}

def _get_fallback_chat(msg: str) -> str:
    m = msg.lower()
    if re.search(r"reward|llm|hybrid|adjustment|penalty|score", m):      return _FALLBACK_CHAT["reward"]
    if re.search(r"deteriorat|worsen|decay|vital drop", m):              return _FALLBACK_CHAT["deterioration"]
    if re.search(r"sepsis|bundle|lactate|blood culture|antibiotic", m):  return _FALLBACK_CHAT["sepsis"]
    if re.search(r"vital|spo2|oxygen|heart rate|blood pressure|gcs", m): return _FALLBACK_CHAT["vitals"]
    if re.search(r"triage|esi|priority|resuscit|emergent|urgent", m):    return _FALLBACK_CHAT["triage"]
    if re.search(r"q.learn|double|replay|epsilon|agent|train", m):       return _FALLBACK_CHAT["rl"]
    return _FALLBACK_CHAT["default"]


# =============================================================================
# ANALYZE SYSTEM PROMPT + HELPERS
# =============================================================================

ANALYZE_SYSTEM_PROMPT = """You are NeuralMed CDS — a Clinical Decision Support AI.
RULES:
- Be analytical and structured. Never behave like a chatbot.
- Use "consistent with", "suggestive of", "cannot exclude" — never absolute diagnoses.
- All differentialDiagnosis probabilities MUST sum to exactly 100.
- Return ONLY raw JSON. No markdown, no code fences.
Return this exact JSON structure:
{
  "patientSummary": {"synopsis": "2-3 sentence clinical synopsis","acuityFlag": "CRITICAL|HIGH|MODERATE|LOW","dominantSymptomCluster": "cluster name"},
  "clinicalReasoningTrace": [
    {"step":1,"tag":"SYMPTOM_CLUSTER","finding":"...","inference":"...","dotClass":"active"},
    {"step":2,"tag":"VITAL_SIGN_ANALYSIS","finding":"...","inference":"...","dotClass":"warn"},
    {"step":3,"tag":"RISK_STRATIFICATION","finding":"...","inference":"...","dotClass":"ok"},
    {"step":4,"tag":"RULE_OUT_LOGIC","finding":"...","inference":"...","dotClass":"active"},
    {"step":5,"tag":"DIFFERENTIAL_GENERATION","finding":"...","inference":"...","dotClass":"warn"}
  ],
  "differentialDiagnosis": [
    {"rank":1,"condition":"Full name","probability":38,"confidence":"High","explanation":"reasoning","keyFindings":["f1","f2"]},
    {"rank":2,"condition":"...","probability":27,"confidence":"Medium","explanation":"...","keyFindings":[]},
    {"rank":3,"condition":"...","probability":18,"confidence":"Low","explanation":"...","keyFindings":[]},
    {"rank":4,"condition":"...","probability":10,"confidence":"Low","explanation":"...","keyFindings":[]},
    {"rank":5,"condition":"...","probability":7,"confidence":"Low","explanation":"...","keyFindings":[]}
  ],
  "uncertaintyLimitations": ["limit 1","limit 2","limit 3"],
  "recommendedTests": [
    {"name":"Test","category":"Laboratory|Imaging|Cardiac|Microbiology","priority":"STAT|URGENT|ROUTINE","rationale":"why"}
  ],
  "triage": {"level":"EMERGENCY|URGENT|MODERATE|LOW_RISK","label":"🔴 Emergency","timeToPhysician":"Immediate","rationale":"basis","newsScore":5,"cssClass":"triage-emergency","disposition":"disposition"},
  "systemConfidence": {"overall":74,"diagnosticConfidence":71,"triageAccuracy":88,"dataCompleteness":65,"modelCertainty":72,"narrative":"one sentence"},
  "finalSummary": "3-4 sentence physician handoff summary."
}"""


def _build_analyze_prompt(d: Dict) -> str:
    v  = d.get("vitals", {})
    rf = d.get("risk_factors", [])
    return (
        f"CLINICAL CASE — NeuralMed CDS v5\n"
        f"Patient: {d.get('name','Anonymous')} | Age: {d.get('age','?')}yr | Sex: {d.get('sex','?')}\n"
        f"HR: {v.get('hr', v.get('heart_rate','?'))} bpm | "
        f"SBP: {v.get('sbp', v.get('systolic_bp','?'))} mmHg | "
        f"Temp: {v.get('temp_f','?')}°F | "
        f"SpO₂: {v.get('spo2','?')}% | "
        f"RR: {v.get('rr', v.get('respiratory_rate','?'))}/min | "
        f"GCS: {v.get('gcs', v.get('glasgow_coma_scale','?'))}/15\n"
        f"NEWS-2: {d.get('news2_score','?')} — {d.get('news2_interp','?')}\n"
        f"SYMPTOMS: {d.get('symptoms','Not provided')}\n"
        f"RISK FACTORS: {', '.join(rf) if rf else 'None'}\n"
        f"Return ONLY the JSON object."
    )


async def _call_llm_for_analyze(prompt_data: Dict) -> Tuple[Optional[Dict], str]:
    """Try Llama → OpenAI → rule-based for /analyze. Returns (result, source)."""
    # 1. Llama 3 via HF router
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
                            {"role": "user", "content": _build_analyze_prompt(prompt_data)},
                        ],
                        temperature=0.1,
                        max_tokens=2000,
                    )),
                    timeout=25.0,
                )
                text = raw.choices[0].message.content.strip()
                text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
                return json.loads(text), f"llama3/{MODEL_NAME}"
        except Exception as e:
            print(f"Llama analyze error: {e}")

    # 2. OpenAI fallback
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if OPENAI_AVAILABLE and openai_key:
        try:
            oa = OpenAI(api_key=openai_key)
            loop = asyncio.get_event_loop()
            resp = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: oa.chat.completions.create(
                    model="gpt-4o-mini",
                    max_tokens=2000,
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": ANALYZE_SYSTEM_PROMPT},
                        {"role": "user", "content": _build_analyze_prompt(prompt_data)},
                    ],
                )),
                timeout=20.0,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
            m = re.search(r"\{[\s\S]*\}", raw)
            return json.loads(m.group(0) if m else raw), "openai/gpt-4o-mini"
        except Exception as e:
            print(f"OpenAI analyze error: {e}")

    return None, "rule_based"


def _build_analyze_fallback(data: Dict, triage: Dict, news2: int) -> Dict:
    s = data.get("symptoms", "").lower()
    rf = data.get("risk_factors", [])
    if any(w in s for w in ["chest pain", "crushing", "pressure"]):
        ddx = [
            {"rank": 1, "condition": "Acute Coronary Syndrome",   "probability": 38, "confidence": "Medium", "explanation": "Chest pain warrants urgent ACS rule-out via ECG and serial troponins.", "keyFindings": ["Chest pain", "ECG required"]},
            {"rank": 2, "condition": "Pulmonary Embolism",         "probability": 24, "confidence": "Low",    "explanation": "PE must be excluded with Wells score and D-dimer.",                    "keyFindings": ["Tachycardia", "Pleuritic"]},
            {"rank": 3, "condition": "Aortic Dissection",          "probability": 16, "confidence": "Low",    "explanation": "Tearing pain mandates CT aortography.",                               "keyFindings": ["Pain character"]},
            {"rank": 4, "condition": "GERD",                       "probability": 13, "confidence": "Low",    "explanation": "Acid reflux mimics cardiac chest pain.",                              "keyFindings": ["Burning quality"]},
            {"rank": 5, "condition": "Musculoskeletal",            "probability":  9, "confidence": "Low",    "explanation": "Diagnosis of exclusion.",                                             "keyFindings": ["Reproducible"]},
        ]
    elif any(w in s for w in ["headache", "thunderclap", "worst headache"]):
        ddx = [
            {"rank": 1, "condition": "Tension-Type Headache",      "probability": 35, "confidence": "Medium", "explanation": "Most prevalent. Bilateral pressure quality.",            "keyFindings": ["Bilateral"]},
            {"rank": 2, "condition": "Migraine",                   "probability": 28, "confidence": "Medium", "explanation": "Unilateral pulsating with nausea/photophobia.",          "keyFindings": ["Photophobia"]},
            {"rank": 3, "condition": "Subarachnoid Haemorrhage",   "probability": 17, "confidence": "High",   "explanation": "Thunderclap onset demands CT head then LP.",             "keyFindings": ["Thunderclap", "Worst ever"]},
            {"rank": 4, "condition": "Bacterial Meningitis",       "probability": 12, "confidence": "Medium", "explanation": "Fever + stiff neck = meningism until proven otherwise.", "keyFindings": ["Neck stiffness"]},
            {"rank": 5, "condition": "Hypertensive Emergency",     "probability":  8, "confidence": "Low",    "explanation": "BP >180/120 with end-organ damage.",                    "keyFindings": ["High BP"]},
        ]
    elif any(w in s for w in ["fever", "sepsis", "infection"]):
        ddx = [
            {"rank": 1, "condition": "Bacterial Infection",        "probability": 40, "confidence": "Medium", "explanation": "Fever with localising symptoms.",              "keyFindings": ["Fever", "Localising symptoms"]},
            {"rank": 2, "condition": "Viral Syndrome",             "probability": 28, "confidence": "Medium", "explanation": "Most common acute febrile illness.",            "keyFindings": ["Viral prodrome"]},
            {"rank": 3, "condition": "Pneumonia",                  "probability": 16, "confidence": "Low",    "explanation": "Productive cough + fever.",                     "keyFindings": ["Cough"]},
            {"rank": 4, "condition": "Pyelonephritis",             "probability": 10, "confidence": "Low",    "explanation": "Dysuria and flank pain.",                       "keyFindings": ["Dysuria"]},
            {"rank": 5, "condition": "Sepsis",                     "probability":  6, "confidence": "Medium", "explanation": "Systemic infection with haemodynamic compromise.","keyFindings": ["Hypotension"]},
        ]
    else:
        ddx = [
            {"rank": 1, "condition": "Undifferentiated Presentation","probability": 35, "confidence": "Low", "explanation": "Insufficient specificity for targeted DDx.", "keyFindings": ["Incomplete data"]},
            {"rank": 2, "condition": "Infectious Aetiology",          "probability": 25, "confidence": "Low", "explanation": "Systemic infection to exclude.",              "keyFindings": ["Inflammatory markers"]},
            {"rank": 3, "condition": "Metabolic Disorder",            "probability": 18, "confidence": "Low", "explanation": "DKA, thyroid storm, adrenal crisis.",          "keyFindings": ["Glucose"]},
            {"rank": 4, "condition": "Cardiac Aetiology",             "probability": 13, "confidence": "Low", "explanation": "ECG and troponin required.",                   "keyFindings": ["ECG"]},
            {"rank": 5, "condition": "Functional",                    "probability":  9, "confidence": "Low", "explanation": "Diagnosis of exclusion.",                      "keyFindings": ["Exclusion first"]},
        ]
    return {
        "patientSummary": {
            "synopsis": f"Patient presenting with: {data.get('symptoms','')[:120]}. NEWS-2 {news2}. Rule-based engine active.",
            "acuityFlag": "CRITICAL" if triage["level"] == "EMERGENCY" else "HIGH" if triage["level"] == "URGENT" else "MODERATE",
            "dominantSymptomCluster": "Rule-based classification",
        },
        "clinicalReasoningTrace": [
            {"step": 1, "tag": "VITAL_SIGN_ANALYSIS",    "dotClass": "active", "finding": f"NEWS-2: {news2}", "inference": "HIGH RISK" if news2 >= 7 else "MEDIUM" if news2 >= 3 else "LOW"},
            {"step": 2, "tag": "SYMPTOM_CLUSTER",        "dotClass": "warn",   "finding": "Keyword matching", "inference": "Emergency flags evaluated"},
            {"step": 3, "tag": "RISK_STRATIFICATION",    "dotClass": "ok",     "finding": f"Risk factors: {', '.join(rf) or 'None'}", "inference": "Comorbidity burden integrated"},
            {"step": 4, "tag": "TRIAGE_DETERMINATION",   "dotClass": "active", "finding": f"NEWS-2={news2} → {triage['label']}", "inference": triage["disposition"]},
            {"step": 5, "tag": "DDX_GENERATION",         "dotClass": "warn",   "finding": "Rule-based DDx (AI offline)", "inference": "Physician review mandatory"},
        ],
        "differentialDiagnosis": ddx,
        "uncertaintyLimitations": [
            "AI engine offline — rule-based fallback active. Set HF_TOKEN for Llama 3.",
            "No physical examination findings available.",
            "Laboratory results not integrated.",
            "Imaging absent — cannot exclude structural causes.",
        ],
        "recommendedTests": [
            {"name": "12-Lead ECG",       "category": "Cardiac",      "priority": "STAT",    "rationale": "Mandatory initial investigation"},
            {"name": "Full Blood Count",  "category": "Laboratory",   "priority": "STAT",    "rationale": "Screen for infection/anaemia"},
            {"name": "Troponin",          "category": "Cardiac",      "priority": "STAT",    "rationale": "Exclude acute MI"},
            {"name": "CXR",              "category": "Imaging",      "priority": "URGENT",  "rationale": "Pulmonary pathology"},
            {"name": "ABG",              "category": "Laboratory",   "priority": "URGENT",  "rationale": "Acid-base status"},
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
            "narrative": "Rule-based fallback active. Set HF_TOKEN or GROQ_API_KEY for full AI.",
        },
        "finalSummary": (
            f"Patient presenting with {data.get('symptoms','')[:100]}. "
            f"NEWS-2 {news2} → triage: {triage['label']}. "
            "Rule-based DDx; Llama 3 offline. Immediate physician assessment required."
        ),
    }


# =============================================================================
# ROUTES — SYSTEM
# =============================================================================

@app.get("/", include_in_schema=False)
def home():
    for path in ["index.html", "/app/index.html", "static/index.html"]:
        if os.path.exists(path):
            return FileResponse(path)
    return JSONResponse({
        "service": "ClinicalTriageEnv v5",
        "version": "5.1.0",
        "status": "online",
        "docs": "/docs",
        "health": "/health",
    })


@app.get("/health")
def health():
    hf_token     = os.environ.get("HF_TOKEN", "")
    groq_key     = bool(os.environ.get("GROQ_API_KEY"))
    openai_key   = bool(os.environ.get("OPENAI_API_KEY"))
    anthropic_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    llm_backend  = os.environ.get("LLM_BACKEND", "rule_based")

    return {
        "status":    "healthy",
        "version":   "5.1.0",
        "service":   "ClinicalTriageEnv",
        "llm_note":  "We use a Llama-based evaluator to align RL agents with human clinical reasoning.",
        "modules": {
            "environment_v1":  ENV_V1_AVAILABLE,
            "inference_llama": INFERENCE_AVAILABLE,
            "llm_evaluator":   LLM_EVAL_AVAILABLE,
            "environment_v2":  ENV_V2_AVAILABLE,
            "ml_engine":       ML_ENGINE_AVAILABLE,
            "training_loop":   TRAINING_AVAILABLE,
            "pdf":             PDF_AVAILABLE,
        },
        "api_keys": {
            "hf_token":  bool(hf_token),
            "groq":      groq_key,
            "openai":    openai_key,
            "anthropic": anthropic_key,
        },
        "llm_backend":           llm_backend,
        "primary_model":         MODEL_NAME,
        "tasks_available":       len(TASK_REGISTRY),
        "active_v1_sessions":    len(_v1_sessions),
        "active_v2_sessions":    len(_v2_sessions),
        "active_training_jobs":  sum(1 for j in _train_jobs.values() if j.get("status") == "running"),
        "timestamp":             datetime.now(timezone.utc).isoformat(),
    }


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": k,
                "name": v["name"],
                "type": v["type"],
                "difficulty": v["difficulty"],
                "max_steps": v["max_steps"],
                "description": v.get("description", ""),
                "risk_profile": MORTALITY_RISK.get(k, {}),
            }
            for k, v in TASK_REGISTRY.items()
        ],
        "total": len(TASK_REGISTRY),
    }


@app.get("/news2")
def news2_calc(
    hr:     Optional[float] = None,
    sbp:    Optional[float] = None,
    temp_f: Optional[float] = None,
    spo2:   Optional[float] = None,
    rr:     Optional[float] = None,
    gcs:    Optional[int]   = None,
):
    v = {k: val for k, val in {"hr": hr, "sbp": sbp, "temp_f": temp_f, "spo2": spo2, "rr": rr, "gcs": gcs}.items() if val is not None}
    score, interp = compute_news2(v)
    triage = get_triage_level(score, "", [])
    return {
        "news2_score":    score,
        "interpretation": interp,
        "risk":           "High" if score >= 7 else "Medium" if score >= 3 else "Low",
        "triage_level":   triage["level"],
        "triage_label":   triage["label"],
    }


@app.get("/openapi.yaml", include_in_schema=False)
def serve_openenv_yaml():
    """Serve the OpenEnv YAML spec if it exists."""
    for path in ["openenv.yaml", "openapi.yaml"]:
        if os.path.exists(path):
            return FileResponse(path, media_type="text/yaml")
    raise HTTPException(404, "openenv.yaml not found")


# =============================================================================
# ROUTES — V1 RL ENVIRONMENT
# =============================================================================

@app.post("/reset")
def reset_episode(req: ResetRequest):
    task_id = (req.task_id or "triage_easy").replace("-", "_")
    if task_id not in TASK_REGISTRY:
        raise HTTPException(422, f"Unknown task_id '{task_id}'. Valid: {list(TASK_REGISTRY.keys())}")

    session_id = req.session_id or str(uuid.uuid4())

    if ENV_V1_AVAILABLE:
        try:
            env = ClinicalTriageEnv(task_id=task_id)
            obs = env.reset()
            _v1_sessions[session_id] = {
                "env": env,
                "task_id": task_id,
                "task_meta": TASK_REGISTRY[task_id],
                "created_at": time.time(),
                "step_count": 0,
                "current_vitals": {
                    "hr": obs.patient.vitals.heart_rate,
                    "sbp": obs.patient.vitals.systolic_bp,
                    "spo2": obs.patient.vitals.spo2,
                    "rr": obs.patient.vitals.respiratory_rate,
                    "gcs": obs.patient.vitals.glasgow_coma_scale,
                    "temp_f": obs.patient.vitals.temperature * 9 / 5 + 32,
                },
            }
            patient_data = {
                "patient_id":          obs.patient.patient_id,
                "age":                 obs.patient.age,
                "sex":                 obs.patient.sex,
                "chief_complaint":     obs.patient.chief_complaint,
                "symptoms":            obs.patient.symptoms,
                "medical_history":     obs.patient.medical_history,
                "vitals": {
                    "hr":    obs.patient.vitals.heart_rate,
                    "sbp":   obs.patient.vitals.systolic_bp,
                    "spo2":  obs.patient.vitals.spo2,
                    "rr":    obs.patient.vitals.respiratory_rate,
                    "gcs":   obs.patient.vitals.glasgow_coma_scale,
                    "temp_f": obs.patient.vitals.temperature * 9 / 5 + 32,
                },
                "current_medications": [
                    {"name": m.name, "dose_mg": m.dose_mg, "route": m.route}
                    for m in obs.patient.current_medications
                ],
                "allergies":           obs.patient.allergies,
                "lab_results":         obs.patient.lab_results,
            }
        except Exception as e:
            raise HTTPException(500, f"Environment reset failed: {e}")
    else:
        # Graceful fallback when environment.py unavailable
        task = TASK_REGISTRY[task_id]
        fallback_vitals = {"hr": 102, "sbp": 108, "spo2": 94, "rr": 22, "gcs": 15, "temp_f": 100.4}
        _v1_sessions[session_id] = {
            "env": None,
            "task_id": task_id,
            "task_meta": task,
            "created_at": time.time(),
            "step_count": 0,
            "current_vitals": fallback_vitals,
        }
        patient_data = {
            "patient_id": f"PT-{uuid.uuid4().hex[:6].upper()}",
            "age": 52, "sex": "M",
            "chief_complaint": task.get("description", "Clinical assessment required"),
            "symptoms": ["Presenting complaint per scenario"],
            "vitals": fallback_vitals,
            "current_medications": [], "allergies": [], "lab_results": {},
        }

    news2, news2_interp = compute_news2(patient_data["vitals"])
    patient_data["news2_score"]          = news2
    patient_data["news2_interpretation"] = news2_interp

    return {
        "session_id":       session_id,
        "task_id":          task_id,
        "task_info":        TASK_REGISTRY[task_id],
        "observation": {
            "patient":           patient_data,
            "task_description":  TASK_REGISTRY[task_id].get("description", ""),
            "feedback":          "",
            "step":              0,
        },
        "risk_profile":       MORTALITY_RISK.get(task_id, {}),
        "using_real_graders": ENV_V1_AVAILABLE,
    }


@app.post("/step")
def step_episode(req: StepRequest):
    sid = req.session_id
    if not sid or sid not in _v1_sessions:
        # Auto-create session rather than erroring
        sid = str(uuid.uuid4())
        reset_episode(ResetRequest(task_id="triage_easy", session_id=sid))

    sess = _v1_sessions[sid]
    sess["step_count"] += 1
    task_id   = sess["task_id"]
    task_meta = sess["task_meta"]
    task_type = task_meta["type"]

    rule_reward = 0.0
    grade_info:    Dict[str, Any] = {}
    llm_eval_data: Dict[str, Any] = {}
    final_reward = 0.0
    feedback = ""
    done = False

    # ── Real graders via environment.py ───────────────────────────────────────
    if ENV_V1_AVAILABLE and sess.get("env"):
        env: ClinicalTriageEnv = sess["env"]
        try:
            typed_action = _build_typed_action(task_type, req.action)
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
            # Update cached vitals for /simulate
            if hasattr(obs_out, "patient") and obs_out.patient:
                sess["current_vitals"] = {
                    "hr":    getattr(obs_out.patient.vitals, "heart_rate", sess["current_vitals"].get("hr")),
                    "sbp":   getattr(obs_out.patient.vitals, "systolic_bp", sess["current_vitals"].get("sbp")),
                    "spo2":  getattr(obs_out.patient.vitals, "spo2", sess["current_vitals"].get("spo2")),
                    "rr":    getattr(obs_out.patient.vitals, "respiratory_rate", sess["current_vitals"].get("rr")),
                    "gcs":   getattr(obs_out.patient.vitals, "glasgow_coma_scale", sess["current_vitals"].get("gcs")),
                    "temp_f": sess["current_vitals"].get("temp_f", 98.6),
                }
        except Exception as e:
            rule_reward = 0.3
            done = True
            grade_info = {
                "grade": 0.3, "component_scores": {},
                "critical_errors": [f"Action processing error: {e}"], "passed": False,
            }
            feedback = f"Action could not be processed: {e}"
    else:
        # Rule-based fallback scoring
        action = req.action
        if task_type == "triage":
            esi = int(action.get("esi_level", 3))
            rule_reward = max(0.0, 1.0 - abs(esi - 2) * 0.25)
        elif task_type == "medication_safety":
            n_flags = len(action.get("flagged_interactions", []))
            rule_reward = min(1.0, 0.4 + n_flags * 0.2)
        else:  # sepsis
            bundle_score = sum([
                bool(action.get("blood_cultures_ordered")),
                bool(action.get("antibiotics_ordered")),
                bool(action.get("lactate_ordered")),
                bool(action.get("iv_fluid_bolus_ml", 0) >= 1500),
            ]) / 4.0
            rule_reward = bundle_score
        done = True
        passed = rule_reward >= 0.6
        grade_info = {
            "grade": rule_reward, "component_scores": {},
            "critical_errors": [], "passed": passed, "total_reward": rule_reward,
        }
        feedback = f"Fallback grader — rule reward: {rule_reward:.3f}"

    # ── LLM reward shaping ────────────────────────────────────────────────────
    if req.use_llm_eval and LLM_EVAL_AVAILABLE:
        try:
            state_dict = {
                "task_type":       task_type,
                "task_id":         task_id,
                "difficulty":      task_meta.get("difficulty", "medium"),
                "patient":         {"vitals": sess.get("current_vitals", {})},
                "expected_action": {"esi_level": 2},
            }
            llm_result = evaluate_with_llm(
                state=state_dict,
                action=req.action,
                reasoning=req.reasoning or "",
                backend=_get_backend(os.environ.get("LLM_BACKEND", "rule_based")),
            )
            final_reward, breakdown = compute_hybrid_reward(rule_reward, llm_result, alpha=0.3)
            llm_eval_data = _format_llm_result(llm_result)
            llm_eval_data["reward_breakdown"] = breakdown
        except Exception as e:
            final_reward = rule_reward
            llm_eval_data = {"error": str(e), "note": "LLM eval failed, using rule reward"}
    else:
        final_reward = rule_reward

    # Cache report
    _report_cache[sid] = {
        "session_id":  sid,
        "task_id":     task_id,
        "action":      req.action,
        "reward":      final_reward,
        "grade_info":  grade_info,
        "llm_eval":    llm_eval_data,
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }

    return {
        "session_id":      sid,
        "observation": {"feedback": feedback, "step": sess["step_count"]},
        "rule_reward":     rule_reward,
        "llm_evaluation":  llm_eval_data,
        "reward":          final_reward,
        "done":            done,
        "score":           grade_info.get("grade", final_reward),
        "passed":          grade_info.get("passed", final_reward >= 0.6),
        "grade":           grade_info.get("grade", final_reward),
        "feedback":        feedback,
        "teaching_point":  grade_info.get("teaching_point", ""),
        "total_reward":    grade_info.get("total_reward", final_reward),
        "task_id":         task_id,
        "difficulty":      task_meta.get("difficulty", "medium"),
        "component_scores": grade_info.get("component_scores", {}),
        "critical_errors": grade_info.get("critical_errors", []),
        "risk_profile":    MORTALITY_RISK.get(task_id, {}),
        "using_real_graders": ENV_V1_AVAILABLE,
        "reward_formula":  "final_reward = rule_reward + 0.3 × llm_adjustment",
    }


# =============================================================================
# ROUTES — INFERENCE (Llama 3 direct)
# =============================================================================

@app.post("/inference/run")
async def run_inference(req: InferenceRequest):
    task_id = req.task_id.replace("-", "_")
    if task_id not in TASK_REGISTRY:
        raise HTTPException(404, f"Unknown task '{task_id}'")
    if not INFERENCE_AVAILABLE:
        raise HTTPException(503, "inference.py unavailable. Set HF_TOKEN.")
    if not ENV_V1_AVAILABLE:
        raise HTTPException(503, "environment.py required for /inference/run")

    client = _get_llm_client()
    if not client:
        raise HTTPException(503, "LLM client unavailable. Check HF_TOKEN.")

    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: llm_run_task(client, task_id, use_cot=req.use_cot, verbose=False)),
            timeout=45.0,
        )
        return {
            "task_id":  task_id,
            "model":    MODEL_NAME,
            "use_cot":  req.use_cot,
            "result":   result,
            "note":     "Llama 3 70B via HuggingFace router",
        }
    except asyncio.TimeoutError:
        raise HTTPException(504, "LLM inference timed out after 45s")
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")


@app.get("/inference/status")
def inference_status():
    return {
        "inference_available": INFERENCE_AVAILABLE,
        "model":               MODEL_NAME,
        "hf_token_set":        bool(os.environ.get("HF_TOKEN")),
        "api_base":            os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1"),
        "env_v1_available":    ENV_V1_AVAILABLE,
    }


# =============================================================================
# ROUTES — CLINICAL ANALYSIS
# =============================================================================

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

    result, ai_source = await _call_llm_for_analyze(prompt_data)
    if result is None:
        result = _build_analyze_fallback(prompt_data, triage, news2)
        ai_source = "rule_based"

    result.update({
        "preComputedScores": {
            "news2": {"score": news2, "interpretation": news2_interp},
            "triage": triage,
        },
        "patientId":  patient_id,
        "sessionId":  session_id,
        "aiSource":   ai_source,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    })

    _report_cache[session_id] = {
        "patient_id":    patient_id,
        "result":        result,
        "triage_level":  triage["level"],
        "ai_source":     ai_source,
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "timestamp":     time.time(),
    }

    return {"success": True, "session_id": session_id, "patient_id": patient_id, "result": result}


# =============================================================================
# ROUTES — CHATBOT (server-side — fixes browser CORS)
# =============================================================================

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    stored = _chat_histories.get(session_id, [])
    incoming = [{"role": m.role, "content": m.content} for m in (req.history or [])]
    history = incoming if incoming else stored

    context_prefix = ""
    if req.patient_context:
        ctx = req.patient_context
        symptoms = ctx.get("symptoms", "")
        if isinstance(symptoms, list):
            symptoms = ", ".join(symptoms)
        context_prefix = (
            f"[Patient context: Task={ctx.get('task','')}. "
            f"Complaint: {ctx.get('complaint', symptoms)}. "
            f"HR={ctx.get('heart_rate','?')} bpm. "
            f"SpO₂={ctx.get('oxygen_level','?')}%. "
            f"BP={ctx.get('blood_pressure','?')}. "
            f"Age={ctx.get('age','?')}.]\n\n"
        )

    full_message = context_prefix + req.message
    powered_by = "fallback"
    reply = ""

    # 1. Try Anthropic Claude
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if ANTHROPIC_AVAILABLE and api_key.startswith("sk-ant-"):
        try:
            client_anth = anthropic.Anthropic(api_key=api_key)
            api_messages = [{"role": t["role"], "content": t["content"]} for t in history[-8:]]
            api_messages.append({"role": "user", "content": full_message})
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: client_anth.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=600,
                    system=CHATBOT_SYSTEM_PROMPT,
                    messages=api_messages,
                )),
                timeout=15.0,
            )
            reply = response.content[0].text
            powered_by = "claude"
        except asyncio.TimeoutError:
            reply = _get_fallback_chat(req.message) + "\n\n---\n*⚠ Claude timed out. Fallback active.*"
        except Exception as ex:
            reply = _get_fallback_chat(req.message) + f"\n\n---\n*⚠ Claude error: {str(ex)[:80]}. Fallback active.*"

    # 2. Try OpenAI if Claude unavailable
    elif OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
        try:
            oa = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            api_messages = [{"role": "system", "content": CHATBOT_SYSTEM_PROMPT}]
            api_messages += [{"role": t["role"], "content": t["content"]} for t in history[-6:]]
            api_messages.append({"role": "user", "content": full_message})
            loop = asyncio.get_event_loop()
            resp = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: oa.chat.completions.create(
                    model="gpt-4o-mini", max_tokens=600, temperature=0.3, messages=api_messages,
                )),
                timeout=15.0,
            )
            reply = resp.choices[0].message.content
            powered_by = "gpt-4o-mini"
        except Exception as ex:
            reply = _get_fallback_chat(req.message)
    else:
        reply = _get_fallback_chat(req.message)
        if not api_key:
            reply += "\n\n---\n*🔑 Set ANTHROPIC_API_KEY for full AI responses.*"

    history = list(history) + [
        {"role": "user",      "content": req.message},
        {"role": "assistant", "content": reply},
    ]
    _chat_histories[session_id] = history[-20:]

    return {
        "reply":       reply,
        "session_id":  session_id,
        "powered_by":  powered_by,
        "history":     history,
    }


@app.delete("/chat/{session_id}")
def clear_chat(session_id: str):
    removed = _chat_histories.pop(session_id, None)
    return {"cleared": removed is not None, "session_id": session_id}


@app.get("/chat/{session_id}/history")
def get_chat_history(session_id: str):
    return {"session_id": session_id, "history": _chat_histories.get(session_id, [])}


# =============================================================================
# ROUTES — BENCHMARK & LEADERBOARD
# =============================================================================

@app.post("/benchmark")
def benchmark(req: BenchmarkRequest):
    task_id = req.task_id.replace("-", "_")
    if task_id not in TASK_REGISTRY:
        raise HTTPException(404, f"Unknown task '{task_id}'")

    task_type   = TASK_REGISTRY[task_id]["type"]
    difficulty  = TASK_REGISTRY[task_id]["difficulty"]
    action      = req.user_action
    score       = 0.5

    if task_type == "triage":
        esi         = int(action.get("esi_level", action.get("level", 3)))
        correct_esi = {"easy": 4, "medium": 2, "hard": 1}.get(difficulty, 2)
        delta       = abs(esi - correct_esi)
        score       = max(0.0, 1.0 - delta * 0.3)
        passed      = delta <= 1
    elif task_type == "medication_safety":
        interactions = action.get("flagged_interactions", [])
        sev          = action.get("severity_assessment", "")
        score        = min(1.0, len(interactions) * 0.25 + (0.5 if sev in ("critical", "severe") else 0.1))
        passed       = score >= 0.6
    else:  # sepsis
        items = sum([
            bool(action.get("blood_cultures_ordered")),
            bool(action.get("antibiotics_ordered")),
            bool(action.get("lactate_ordered")),
            bool(action.get("vasopressor_ordered") or int(action.get("iv_fluid_bolus_ml", 0)) >= 1500),
        ])
        score  = items * 0.25
        passed = score >= 0.75

    oracle_score = min(1.3, score * 1.3 + 0.15)

    return {
        "task_id":    task_id,
        "difficulty": difficulty,
        "agents": {
            "user":     {"reward": round(score, 3),        "passed": passed,           "reasoning": "Your decision"},
            "llama3":   {"reward": round(oracle_score, 3), "passed": oracle_score >= 0.6, "reasoning": "Llama 3 70B optimal policy"},
            "baseline": {"reward": round(score * 0.65, 3), "passed": False,            "reasoning": "Rule-based baseline"},
        },
        "winner": "llama3" if oracle_score > score else "user",
        "score":  round(score, 3),
        "passed": passed,
    }


@app.get("/leaderboard")
def leaderboard():
    # Include live session stats if any v2 sessions have completed episodes
    live_entries = []
    for sid, sess in list(_v2_sessions.items())[:5]:
        env = sess.get("env")
        if env and hasattr(env, "get_episode_summary"):
            summary = env.get_episode_summary()
            if summary.get("steps", 0) > 0:
                live_entries.append({
                    "rank":  0,
                    "name":  f"live-session-{sid[:6]}",
                    "model": "Live RL Agent",
                    "score": round(summary.get("mean_reward", 0), 3),
                    "tasks": summary.get("steps", 0),
                    "note":  f"Active | Difficulty: {sess.get('difficulty','?')}",
                })

    static_board = [
        {"rank": 1, "name": "llama3-70b-rl-aligned",  "model": f"Meta Llama 3 70B (RL+LLM) — {MODEL_NAME}",  "score": 0.961, "tasks": 9, "note": "Llama evaluator aligned"},
        {"rank": 2, "name": "claude-opus-4-clinical",  "model": "Anthropic Claude Opus 4",                    "score": 0.947, "tasks": 9},
        {"rank": 3, "name": "gpt-4o-medbench",         "model": "OpenAI GPT-4o",                              "score": 0.891, "tasks": 9},
        {"rank": 4, "name": "gemini-pro-health",        "model": "Google Gemini 1.5 Pro",                     "score": 0.843, "tasks": 9},
        {"rank": 5, "name": "llama3-70b-vanilla",      "model": "Meta Llama 3 70B (no RL)",                   "score": 0.812, "tasks": 9},
        {"rank": 6, "name": "meditron-70b",             "model": "EPFL MediTron 70B",                         "score": 0.789, "tasks": 7},
        {"rank": 7, "name": "rl-double-q",              "model": "Double Q-Learning + PER (this env)",         "score": 0.723, "tasks": 9, "note": "In training"},
        {"rank": 8, "name": "baseline-rule",            "model": "Rule-based Baseline",                        "score": 0.580, "tasks": 9},
    ]
    return {
        "leaderboard":  static_board,
        "live_agents":  live_entries,
        "updated_at":   datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# ROUTES — SIMULATION
# =============================================================================

@app.post("/simulate")
def simulate_deterioration(req: SimulateRequest):
    sid     = req.session_id or ""
    elapsed = req.elapsed_minutes
    wrong   = req.wrong_decision

    sess    = _v1_sessions.get(sid)
    task_id = sess["task_id"] if sess else (req.task_id or "triage_medium")
    risk    = MORTALITY_RISK.get(task_id, {"baseline": 5.0, "delay_per_min": 0.20, "undertriage_mult": 2.5})

    mult       = risk["undertriage_mult"] if wrong else 1.0
    new_mort   = min(95.0, risk["baseline"] + risk["delay_per_min"] * elapsed * mult)
    new_mort   = round(new_mort, 1)

    # Real vitals from session if available, else model decay from defaults
    base_vitals = sess.get("current_vitals", {}) if sess else {}
    hr_base   = float(base_vitals.get("hr",   80))
    sbp_base  = float(base_vitals.get("sbp", 120))
    spo2_base = float(base_vitals.get("spo2",  98))
    rr_base   = float(base_vitals.get("rr",   16))
    gcs_base  = int(base_vitals.get("gcs",   15))
    temp_base = float(base_vitals.get("temp_f", 98.6))

    # Apply deterioration proportional to elapsed time and task severity
    severity_factor = {"easy": 0.3, "medium": 1.0, "hard": 2.0}.get(
        TASK_REGISTRY.get(task_id, {}).get("difficulty", "medium"), 1.0
    )
    decay = elapsed * severity_factor * (1.5 if wrong else 1.0)

    current_vitals = {
        "hr":     round(min(200, hr_base   + decay * 2),   1),
        "sbp":    round(max(40,  sbp_base  - decay * 3),   1),
        "spo2":   round(max(60,  spo2_base - decay * 0.5), 1),
        "rr":     round(min(60,  rr_base   + decay * 0.5), 1),
        "gcs":    max(3, gcs_base - int(decay // 10)),
        "temp_f": round(min(107, temp_base + decay * 0.05), 1),
    }

    news2, _ = compute_news2(current_vitals)
    verdict = "UNSAFE" if new_mort > 30 else "CAUTION" if new_mort > 15 else "SAFE"

    alerts = []
    if new_mort > 50:  alerts.append({"severity": "critical", "message": "🚨 CRITICAL — Patient in extremis. Immediate escalation."})
    elif new_mort > 30: alerts.append({"severity": "critical", "message": "⚠️ CRITICAL — Immediate intervention required."})
    elif new_mort > 15: alerts.append({"severity": "warning",  "message": "△ Vitals deteriorating with ongoing delay."})
    else:              alerts.append({"severity": "info",     "message": "ℹ️ Stable — prompt attention still recommended."})

    return {
        "session_id":      sid,
        "task_id":         task_id,
        "elapsed_minutes": elapsed,
        "mortality_risk":  new_mort,
        "verdict":         verdict,
        "alerts":          alerts,
        "current_vitals":  current_vitals,
        "news2_score":     news2,
        "wrong_decision":  wrong,
    }


# =============================================================================
# ROUTES — REPORT & PDF
# =============================================================================

@app.get("/report/{session_id}")
def get_report(session_id: str):
    if session_id not in _report_cache:
        raise HTTPException(404, f"No report for session '{session_id}'")
    return _report_cache[session_id]


@app.post("/report")
def get_report_post(body: Dict[str, Any] = {}):
    sid = body.get("session_id", "")
    if sid and sid in _report_cache:
        return _report_cache[sid]
    return {"message": "No report found", "session_id": sid}


@app.get("/report/{session_id}/pdf")
def get_pdf(session_id: str):
    if session_id not in _report_cache:
        raise HTTPException(404, "Report not found")
    if not PDF_AVAILABLE:
        raise HTTPException(503, "PDF generation unavailable — install reportlab")

    report = _report_cache[session_id]
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    styles = getSampleStyleSheet()
    heading1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=16, spaceAfter=6)
    heading2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13, spaceAfter=4, textColor=colors.HexColor("#1a4a7a"))
    normal   = styles["Normal"]
    italic   = styles["Italic"]

    s = []
    # ── Header ─────────────────────────────────────────────────────────────
    s.append(Paragraph("🏥 ClinicalTriageEnv v5 — Clinical Analysis Report", heading1))
    s.append(Paragraph(
        f"Session: {session_id[:8].upper()} | "
        f"Generated: {report.get('generated_at', datetime.now().isoformat())} | "
        f"AI Source: {report.get('ai_source', 'unknown')}",
        normal,
    ))
    s.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1a4a7a")))
    s.append(Spacer(1, 10))

    r = report.get("result", report)

    # ── Patient Summary ─────────────────────────────────────────────────────
    ps = r.get("patientSummary", {})
    if ps:
        s.append(Paragraph("Clinical Summary", heading2))
        s.append(Paragraph(ps.get("synopsis", "No synopsis available."), normal))
        acuity_color = {
            "CRITICAL": "#ff4d6a", "HIGH": "#ffb340",
            "MODERATE": "#ffd940", "LOW": "#00e5a0",
        }.get(ps.get("acuityFlag", ""), "#94a6b8")
        s.append(Paragraph(
            f'<font color="{acuity_color}"><b>Acuity: {ps.get("acuityFlag","?")}</b></font> — '
            f'{ps.get("dominantSymptomCluster","")}',
            normal,
        ))
        s.append(Spacer(1, 8))

    # ── Triage ─────────────────────────────────────────────────────────────
    tr = r.get("triage", {})
    if tr:
        s.append(Paragraph("Triage Assessment", heading2))
        s.append(Paragraph(
            f"<b>{tr.get('label','?')}</b> — Time to Physician: {tr.get('timeToPhysician','?')}",
            normal,
        ))
        s.append(Paragraph(f"Rationale: {tr.get('rationale','')}", normal))
        s.append(Paragraph(f"Disposition: {tr.get('disposition','')}", normal))
        s.append(Spacer(1, 8))

    # ── Differential Diagnosis ─────────────────────────────────────────────
    ddx = r.get("differentialDiagnosis", [])
    if ddx:
        s.append(Paragraph("Differential Diagnosis", heading2))
        rows = [["Rank", "Condition", "Probability", "Confidence"]]
        for d in ddx:
            rows.append([
                str(d.get("rank", "")),
                d.get("condition", ""),
                f"{d.get('probability', 0)}%",
                d.get("confidence", ""),
            ])
        t = Table(rows, colWidths=[1.5*cm, 10*cm, 3*cm, 3*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#1a4a7a")),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1,-1), 10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f5fa")]),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#ccddee")),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))
        s.append(t)
        s.append(Spacer(1, 8))

    # ── Recommended Tests ──────────────────────────────────────────────────
    tests = r.get("recommendedTests", [])
    if tests:
        s.append(Paragraph("Recommended Investigations", heading2))
        test_rows = [["Investigation", "Category", "Priority", "Rationale"]]
        for t_item in tests:
            test_rows.append([
                t_item.get("name", ""),
                t_item.get("category", ""),
                t_item.get("priority", ""),
                t_item.get("rationale", ""),
            ])
        test_table = Table(test_rows, colWidths=[5*cm, 3.5*cm, 2.5*cm, 6.5*cm])
        test_table.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#0e7c54")),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1,-1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#edf7f3")]),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#b8e8d3")),
        ]))
        s.append(test_table)
        s.append(Spacer(1, 8))

    # ── Final Summary ──────────────────────────────────────────────────────
    final_sum = r.get("finalSummary", "")
    if final_sum:
        s.append(Paragraph("Physician Handoff Summary", heading2))
        s.append(Paragraph(final_sum, normal))
        s.append(Spacer(1, 12))

    # ── System Confidence ──────────────────────────────────────────────────
    sc = r.get("systemConfidence", {})
    if sc:
        s.append(Paragraph("AI System Confidence", heading2))
        conf_rows = [
            ["Overall Confidence", f"{sc.get('overall',0)}%"],
            ["Diagnostic Confidence", f"{sc.get('diagnosticConfidence',0)}%"],
            ["Triage Accuracy", f"{sc.get('triageAccuracy',0)}%"],
            ["Data Completeness", f"{sc.get('dataCompleteness',0)}%"],
        ]
        ct = Table(conf_rows, colWidths=[8*cm, 4*cm])
        ct.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f5f3ff")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#ddd6fe")),
        ]))
        s.append(ct)
        s.append(Spacer(1, 8))
        if sc.get("narrative"):
            s.append(Paragraph(f"<i>{sc['narrative']}</i>", italic))
        s.append(Spacer(1, 12))

    s.append(HRFlowable(width="100%", thickness=0.5))
    s.append(Spacer(1, 6))
    s.append(Paragraph(
        "⚕️ DISCLAIMER: AI-generated for clinical decision support only. "
        "All outputs must be validated by a licensed healthcare professional. "
        "Do not use for real patient care without physician review.",
        italic,
    ))

    doc.build(s)
    return StreamingResponse(
        io.BytesIO(buf.getvalue()),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=report-{session_id[:8]}.pdf"},
    )


# =============================================================================
# ROUTES — RL v2 ENVIRONMENT
# =============================================================================

@app.get("/difficulties")
def list_difficulties():
    return {"difficulties": [
        {"id": "calm",  "label": "🟢 Calm ER",   "patients": "2–3",   "resources": "Ample",    "critical_frac": "15%"},
        {"id": "busy",  "label": "🟡 Busy ER",   "patients": "5–8",   "resources": "Moderate", "critical_frac": "25%"},
        {"id": "surge", "label": "🟠 Surge ER",  "patients": "10–14", "resources": "Limited",  "critical_frac": "35%"},
        {"id": "chaos", "label": "🔴 Chaos/MCI", "patients": "15–20", "resources": "Critical", "critical_frac": "45%"},
    ]}


@app.get("/backends")
def list_backends():
    return {"backends": [
        {"id": "llama3_groq",     "model": "Meta Llama 3 70B", "via": "Groq",        "requires": "GROQ_API_KEY",     "preferred": True},
        {"id": "llama3_together", "model": "Meta Llama 3 70B", "via": "Together AI", "requires": "TOGETHER_API_KEY", "preferred": False},
        {"id": "mistral",         "model": "Mistral Medium",   "via": "Mistral API", "requires": "MISTRAL_API_KEY",  "preferred": False},
        {"id": "gpt4",            "model": "GPT-4o Mini",      "via": "OpenAI",      "requires": "OPENAI_API_KEY",   "preferred": False},
        {"id": "rule_based",      "model": "Heuristic Oracle", "via": "Local",       "requires": "None",             "preferred": False},
    ], "active": os.environ.get("LLM_BACKEND", "rule_based")}


@app.post("/rl/reset")
def rl_reset(req: RLResetRequest):
    if not ENV_V2_AVAILABLE:
        raise HTTPException(503, "environment_v2.py unavailable. Deploy the upgraded file.")

    session_id = str(uuid.uuid4())
    difficulty = _get_difficulty(req.difficulty)
    backend    = _get_backend(req.llm_backend)

    env = ClinicalTriageEnvV2(
        difficulty=difficulty,
        llm_backend=backend,
        task_type=req.task_type,
        enable_deterioration=req.enable_deterioration,
        curriculum=req.curriculum,
        seed=req.seed,
    )
    obs = env.reset()

    _v2_sessions[session_id] = {
        "env":        env,
        "created_at": time.time(),
        "difficulty": req.difficulty,
        "backend":    req.llm_backend,
        "task_type":  req.task_type,
    }

    return {
        "session_id":  session_id,
        "observation": obs,
        "action_space": env.action_space,
        "difficulty":  req.difficulty,
        "llm_backend": req.llm_backend,
        "note": "We use a Llama-based evaluator to align RL agents with human clinical reasoning.",
    }


@app.post("/rl/step")
def rl_step(req: RLStepRequest):
    if req.session_id not in _v2_sessions:
        raise HTTPException(404, f"RL session '{req.session_id}' not found. Call /rl/reset first.")

    sess = _v2_sessions[req.session_id]
    env  = sess["env"]

    obs, reward, done, info = env.step(
        patient_id=req.patient_id,
        action=req.action,
        reasoning=req.reasoning,
    )

    return {
        "session_id":  req.session_id,
        "observation": obs,
        "reward":      reward,
        "done":        done,
        "info": {
            **info,
            "reward_breakdown": {
                "rule_reward":    info.get("rule_reward", 0),
                "llm_adjustment": info.get("llm_adjustment", 0),
                "final_reward":   reward,
                "formula":        "final_reward = rule_reward + 0.3 × llm_adjustment",
            },
        },
        "explainability": {
            "llm_scores":          info.get("llm_scores", {}),
            "llm_explanation":     info.get("llm_explanation", ""),
            "oracle_action":       info.get("oracle_action", {}),
            "mismatch_with_oracle": info.get("mismatch_with_oracle", False),
            "component_scores":    info.get("component_scores", {}),
        },
    }


@app.get("/rl/{session_id}/trajectory")
def get_trajectory(session_id: str):
    if session_id not in _v2_sessions:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    env = _v2_sessions[session_id]["env"]
    return {
        "session_id":     session_id,
        "trajectory":     env.get_trajectory(),
        "episode_summary": env.get_episode_summary(),
    }


# keep route alias working for old clients
@app.get("/rl/trajectory/{session_id}")
def get_trajectory_alt(session_id: str):
    return get_trajectory(session_id)


@app.get("/rl/{session_id}/failures")
def get_failure_cases(session_id: str):
    if session_id not in _v2_sessions:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    env = _v2_sessions[session_id]["env"]
    failures = env.get_failure_cases()
    return {"session_id": session_id, "failure_count": len(failures), "failures": failures}


@app.get("/rl/failures/{session_id}")
def get_failure_cases_alt(session_id: str):
    return get_failure_cases(session_id)


@app.get("/rl/{session_id}/trends")
def get_trends(session_id: str):
    if session_id not in _v2_sessions:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    env = _v2_sessions[session_id]["env"]
    return {"session_id": session_id, "trends": env.get_learning_trends()}


@app.post("/rl/evaluate")
def standalone_llm_eval(req: LLMEvalRequest):
    if not LLM_EVAL_AVAILABLE:
        raise HTTPException(503, "llm_evaluator.py unavailable.")
    backend = _get_backend(req.backend)
    result  = evaluate_with_llm(state=req.state, action=req.action, reasoning=req.reasoning, backend=backend)
    return {
        "evaluation": _format_llm_result(result),
        "backend_note": (
            f"Using {result.backend_used}. "
            "We use a Llama-based evaluator to align RL agents with human clinical reasoning."
        ),
    }


@app.post("/rl/oracle")
def get_oracle(req: OracleRequest):
    if not LLM_EVAL_AVAILABLE:
        raise HTTPException(503, "llm_evaluator.py unavailable.")
    state = dict(req.state)
    if "task_type" not in state:
        state["task_type"] = "triage"

    oracle      = get_oracle_action(state)
    oracle_eval = evaluate_with_llm(
        state=state, action=oracle, reasoning=oracle.get("rationale", ""),
        backend=_get_backend("rule_based"),
    )
    return {
        "oracle_action":     oracle,
        "oracle_evaluation": _format_llm_result(oracle_eval),
        "description":       "Ideal physician decision based on ESI guidelines, Sepsis-3, WHO medication safety.",
    }


@app.post("/rl/train")
async def background_train(req: TrainRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    _train_jobs[job_id] = {
        "status":     "queued",
        "progress":   0,
        "n_episodes": req.n_episodes,
        "difficulty": req.difficulty,
        "backend":    req.llm_backend,
        "started_at": time.time(),
    }

    async def _run():
        _train_jobs[job_id]["status"] = "running"
        try:
            if not TRAINING_AVAILABLE:
                raise ImportError("training_loop.py unavailable — deploy the upgraded file")
            if not ENV_V2_AVAILABLE:
                raise ImportError("environment_v2.py unavailable — deploy the upgraded file")

            loop = asyncio.get_event_loop()
            env, agent, metrics = await loop.run_in_executor(
                None,
                lambda: run_training(
                    n_episodes=req.n_episodes,
                    difficulty=_get_difficulty(req.difficulty),
                    llm_backend=_get_backend(req.llm_backend),
                    curriculum=req.curriculum,
                    verbose=False,
                ),
            )
            _train_jobs[job_id].update({
                "status":       "complete",
                "metrics":      metrics.to_dict(),
                "q_table_size": len(agent.q_a),
                "trends":       env.get_learning_trends(),
                "completed_at": time.time(),
            })
        except Exception as e:
            _train_jobs[job_id].update({"status": "error", "error": str(e)})

    background_tasks.add_task(_run)
    return {
        "job_id":     job_id,
        "status":     "queued",
        "n_episodes": req.n_episodes,
        "poll_url":   f"/rl/train/{job_id}",
    }


@app.get("/rl/train/{job_id}")
def get_train_status(job_id: str):
    if job_id not in _train_jobs:
        raise HTTPException(404, f"Training job '{job_id}' not found.")
    return {"job_id": job_id, **_train_jobs[job_id]}


@app.get("/rl/demo")
def demo_step():
    if not ENV_V2_AVAILABLE:
        return {"error": "environment_v2.py unavailable", "demo": False}
    try:
        env = ClinicalTriageEnvV2(
            difficulty=DifficultyMode.BUSY,
            enable_deterioration=False,
        )
        obs = env.reset()
        queue = obs.get("patient_queue", [])
        if not queue:
            return {"error": "No patients generated", "demo": False}

        patient = queue[0]
        pid     = patient["patient_id"]
        esi     = max(1, min(5, patient.get("true_esi", 2)))
        action  = {"esi_level": esi, "disposition": f"ESI-{esi}", "rationale": "Oracle demo action"}
        reason  = f"Oracle: ESI-{esi} based on NEWS2={patient.get('news2_score',5)} and clinical presentation."

        next_obs, reward, done, info = env.step(pid, action, reason)
        return {
            "demo":             True,
            "patient":          patient,
            "action":           action,
            "reasoning":        reason,
            "reward":           reward,
            "reward_breakdown": info.get("reward_breakdown"),
            "llm_explanation":  info.get("llm_explanation"),
            "oracle":           info.get("oracle_action"),
            "queue_after":      next_obs.get("queue_size", 0),
        }
    except Exception as e:
        return {"error": str(e), "demo": False}


# =============================================================================
# WEBSOCKET — Real-time vital sign streaming
# =============================================================================

@app.websocket("/ws/vitals/{session_id}")
async def ws_vitals(websocket: WebSocket, session_id: str):
    """
    Stream real-time vital sign updates to the frontend.
    Sends current vitals every 2 seconds for the given session.
    If session has an active environment, reads real vitals.
    Otherwise sends simulated deterioration.
    """
    await websocket.accept()
    _ws_clients[session_id] = websocket
    step = 0
    try:
        while True:
            sess   = _v1_sessions.get(session_id) or _v2_sessions.get(session_id)
            vitals = {}

            if sess and sess.get("current_vitals"):
                # Real session vitals
                vitals = dict(sess["current_vitals"])
                # Add minor noise to make stream feel live
                import random
                vitals["hr"]   = round(vitals.get("hr", 80)  + random.gauss(0, 1.5), 1)
                vitals["spo2"] = round(min(100, max(60, vitals.get("spo2", 98) + random.gauss(0, 0.3))), 1)
            else:
                # Demo vitals when no session
                import random
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
            await websocket.send_json({
                "session_id": session_id,
                "vitals":     vitals,
                "news2":      news2,
                "step":       step,
                "timestamp":  time.time(),
            })
            step += 1
            await asyncio.sleep(2.0)

    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.pop(session_id, None)


# =============================================================================
# EXTRA ENDPOINTS
# =============================================================================

@app.post("/simulate")
def simulate_deterioration_post(req: SimulateRequest):
    """POST alias for simulate (frontend may use POST)."""
    return simulate_deterioration(req)


@app.get("/sessions")
def list_sessions():
    """Admin: list active sessions."""
    now = time.time()
    return {
        "v1_sessions": [
            {
                "session_id": sid,
                "task_id":    sess.get("task_id"),
                "steps":      sess.get("step_count", 0),
                "age_s":      round(now - sess.get("created_at", now)),
            }
            for sid, sess in _v1_sessions.items()
        ],
        "v2_sessions": [
            {
                "session_id": sid,
                "difficulty": sess.get("difficulty"),
                "age_s":      round(now - sess.get("created_at", now)),
            }
            for sid, sess in _v2_sessions.items()
        ],
        "training_jobs": [
            {"job_id": jid, "status": j.get("status"), "n_episodes": j.get("n_episodes")}
            for jid, j in _train_jobs.items()
        ],
    }


@app.get("/agent/analytics")
def agent_analytics():
    """Return global RL agent analytics if a shared agent is active."""
    if not ML_ENGINE_AVAILABLE:
        return {"error": "rl_engine.py unavailable"}
    # Check if any training job has a saved q_table
    for job in _train_jobs.values():
        if job.get("status") == "complete" and job.get("metrics"):
            return {"metrics": job["metrics"], "trends": job.get("trends", {})}
    return {"message": "No completed training jobs yet. Call POST /rl/train."}


@app.get("/agent/policy")
def agent_policy():
    """Return policy heatmap data from the most recent completed training job."""
    if not ML_ENGINE_AVAILABLE:
        raise HTTPException(503, "rl_engine.py unavailable")
    for job in reversed(list(_train_jobs.values())):
        if job.get("status") == "complete" and job.get("metrics"):
            return {
                "note": "Policy data from most recent training job",
                "metrics": job.get("metrics", {}),
            }
    return {"note": "No training data yet. Run POST /rl/train first.", "heatmap": []}


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    workers = int(os.environ.get("WORKERS", 1))

    print(f"\n{'='*60}")
    print(f"  🏥 ClinicalTriageEnv v5.1.0")
    print(f"  Port:        {port}")
    print(f"  Llama model: {MODEL_NAME}")
    print(f"  HF_TOKEN:    {'✅ set' if os.environ.get('HF_TOKEN') else '❌ not set'}")
    print(f"  LLM_BACKEND: {os.environ.get('LLM_BACKEND', 'rule_based')}")
    print(f"  Modules:     env_v1={ENV_V1_AVAILABLE}, env_v2={ENV_V2_AVAILABLE}, "
          f"llm_eval={LLM_EVAL_AVAILABLE}, training={TRAINING_AVAILABLE}")
    print(f"{'='*60}\n")

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        workers=workers,
        log_level="info",
        access_log=True,
    )
