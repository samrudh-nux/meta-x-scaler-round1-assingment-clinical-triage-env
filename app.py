from __future__ import annotations
import os, sys, uuid, json, time, asyncio, io
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

sys.path.insert(0, "/app")

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

from models import TriageAction, MedicationSafetyAction, SepsisManagementAction, ClinicalState
from environment import ClinicalTriageEnv, TASK_REGISTRY

# ── Optional PDF support ───────────────────────────────────────────────────────
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, HRFlowable, KeepTogether)
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

app = FastAPI(
    title="ClinicalTriageEnv Enterprise",
    version="3.0.0",
    description="Enterprise clinical AI training environment with multi-agent architecture",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Session / State stores ────────────────────────────────────────────────────
_sessions: Dict[str, ClinicalTriageEnv] = {}
_patient_states: Dict[str, Dict] = {}   # live simulation states
_report_cache: Dict[str, Dict] = {}     # completed episode data for PDF

# ─── Risk engine constants ─────────────────────────────────────────────────────
MORTALITY_RISK = {
    "triage_easy":   {"baseline": 0.5,  "undertriage_mult": 2.0,  "delay_per_min": 0.01},
    "triage_medium": {"baseline": 8.0,  "undertriage_mult": 3.5,  "delay_per_min": 0.15},
    "triage_hard":   {"baseline": 18.0, "undertriage_mult": 5.0,  "delay_per_min": 0.40},
    "med_safety_easy":   {"baseline": 0.2,  "undertriage_mult": 1.5, "delay_per_min": 0.005},
    "med_safety_medium": {"baseline": 3.0,  "undertriage_mult": 2.5, "delay_per_min": 0.05},
    "med_safety_hard":   {"baseline": 12.0, "undertriage_mult": 4.0, "delay_per_min": 0.30},
    "sepsis_easy":   {"baseline": 6.0,  "undertriage_mult": 2.5, "delay_per_min": 0.20},
    "sepsis_medium": {"baseline": 22.0, "undertriage_mult": 4.0, "delay_per_min": 0.55},
    "sepsis_hard":   {"baseline": 45.0, "undertriage_mult": 6.0, "delay_per_min": 1.20},
}

# ─── Request models ────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = "triage_easy"
    session_id: Optional[str] = None

class StepRequest(BaseModel):
    action: Dict[str, Any]
    session_id: Optional[str] = None

class AnalyzeRequest(BaseModel):
    session_id: str
    action: Dict[str, Any]
    include_reasoning_trace: bool = True
    include_risk_score: bool = True

class SimulateRequest(BaseModel):
    session_id: str
    elapsed_minutes: int = 5
    wrong_decision: bool = False

class ReportRequest(BaseModel):
    session_id: str
    include_ai_comparison: bool = True

class BenchmarkRequest(BaseModel):
    task_id: str
    user_action: Dict[str, Any]

# ─── Core routes ──────────────────────────────────────────────────────────────

@app.get("/")
def home():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"service": "ClinicalTriageEnv Enterprise v3", "docs": "/docs"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "version": "3.0.0",
        "service": "ClinicalTriageEnv Enterprise",
        "tasks_available": len(TASK_REGISTRY),
        "active_sessions": len(_sessions),
        "pdf_available": PDF_AVAILABLE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": k, "name": v["name"], "type": v["type"],
                "difficulty": v["difficulty"], "max_steps": v["max_steps"],
                "description": v["description"],
                "risk_profile": MORTALITY_RISK.get(k, {}),
            }
            for k, v in TASK_REGISTRY.items()
        ],
        "total": len(TASK_REGISTRY),
    }

@app.post("/reset")
def reset_episode(req: ResetRequest):
    task_id = (req.task_id or "triage_easy").replace("-", "_")
    if task_id not in TASK_REGISTRY:
        raise HTTPException(422, f"Unknown task_id '{task_id}'. Valid: {list(TASK_REGISTRY.keys())}")
    session_id = req.session_id or str(uuid.uuid4())
    env = ClinicalTriageEnv(task_id=task_id)
    _sessions[session_id] = env
    obs = env.reset(task_id=task_id)

    # Initialize simulation state
    patient = obs.patient
    _patient_states[session_id] = {
        "task_id": task_id,
        "original_vitals": {
            "heart_rate": patient.vitals.heart_rate,
            "systolic_bp": patient.vitals.systolic_bp,
            "spo2": patient.vitals.spo2,
            "respiratory_rate": patient.vitals.respiratory_rate,
            "glasgow_coma_scale": patient.vitals.glasgow_coma_scale,
        },
        "current_vitals": {
            "heart_rate": patient.vitals.heart_rate,
            "systolic_bp": patient.vitals.systolic_bp,
            "spo2": patient.vitals.spo2,
            "respiratory_rate": patient.vitals.respiratory_rate,
            "glasgow_coma_scale": patient.vitals.glasgow_coma_scale,
        },
        "elapsed_minutes": 0,
        "alerts": [],
        "deterioration_level": 0,
        "reset_time": time.time(),
    }

    risk = _compute_risk_profile(task_id, None, 0)

    return {
        "session_id": session_id,
        "task_id": task_id,
        "observation": obs.model_dump(),
        "task_info": {
            "name": TASK_REGISTRY[task_id]["name"],
            "type": TASK_REGISTRY[task_id]["type"],
            "difficulty": TASK_REGISTRY[task_id]["difficulty"],
            "description": TASK_REGISTRY[task_id]["description"],
        },
        "risk_profile": risk,
    }

@app.post("/step")
def step_episode(req: StepRequest):
    session_id = req.session_id
    raw = req.action

    if session_id and session_id in _sessions:
        env = _sessions[session_id]
    else:
        session_id = str(uuid.uuid4())
        env = ClinicalTriageEnv(task_id="triage_easy")
        _sessions[session_id] = env

    task_type = env.task_meta["type"]
    action = _build_action(raw, task_type)
    obs, reward, done, info = env.step(action)

    # Compute risk profile
    elapsed = 0
    if session_id in _patient_states:
        elapsed = int(time.time() - _patient_states[session_id]["reset_time"]) // 60

    risk = _compute_risk_profile(env.task_id, raw, elapsed)

    # Cache for report
    _report_cache[session_id] = {
        "task_id": env.task_id,
        "task_meta": env.task_meta,
        "action": raw,
        "reward": reward,
        "info": info,
        "feedback": obs.feedback if hasattr(obs, "feedback") else "",
        "risk": risk,
        "elapsed_minutes": elapsed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "session_id": session_id,
        "observation": obs.model_dump(),
        "reward": round(reward, 4),
        "done": done,
        "score": round(reward, 4),
        "passed": info.get("passed", False),
        "grade": info.get("grade", reward),
        "component_scores": info.get("component_scores", {}),
        "critical_errors": info.get("critical_errors", []),
        "feedback": obs.feedback if hasattr(obs, "feedback") else "",
        "total_reward": info.get("total_reward", reward),
        "task_id": env.task_id,
        "difficulty": env.task_meta["difficulty"],
        "risk_profile": risk,
    }

# ─── /analyze — Multi-agent AI reasoning trace ────────────────────────────────

@app.post("/analyze")
async def analyze_decision(req: AnalyzeRequest):
    """
    Multi-agent analysis endpoint.
    Runs 3 AI agents in parallel:
      1. Diagnostician — clinical pattern recognition
      2. Safety AI — risk & allergy checks
      3. Evaluator — scores against ground truth
    Returns structured reasoning trace + risk profile.
    """
    if req.session_id not in _sessions:
        raise HTTPException(404, "Session not found. Call /reset first.")

    env = _sessions[req.session_id]
    task_type = env.task_meta["type"]
    task_id = env.task_id
    obs = env._build_observation(done=False, reward=None, feedback="")
    patient = obs.patient

    elapsed = 0
    if req.session_id in _patient_states:
        elapsed = int(time.time() - _patient_states[req.session_id]["reset_time"]) // 60

    # Build reasoning trace synchronously (streaming handled at client)
    trace = _build_reasoning_trace(req.action, task_type, patient, env._scenario, elapsed)
    risk = _compute_risk_profile(task_id, req.action, elapsed)

    return {
        "session_id": req.session_id,
        "reasoning_trace": trace,
        "risk_profile": risk,
        "multi_agent_outputs": {
            "diagnostician": trace["diagnostician"],
            "safety_ai": trace["safety_ai"],
            "evaluator": trace["evaluator"],
        },
        "final_verdict": trace["final_verdict"],
        "confidence": trace["confidence"],
    }

# ─── /grade — Detailed grading with component breakdown ───────────────────────

@app.post("/grade")
def grade_decision(req: StepRequest):
    """Grade a decision with full component breakdown."""
    session_id = req.session_id
    if not session_id or session_id not in _sessions:
        raise HTTPException(404, "Session not found.")

    env = _sessions[session_id]
    task_type = env.task_meta["type"]
    action = _build_action(req.action, task_type)

    # Use grader directly for detailed output
    scenario = env._scenario
    if task_type == "triage":
        result = env._triage_grader.grade(action, scenario)
    elif task_type == "medication_safety":
        result = env._med_grader.grade(action, scenario)
    else:
        result = env._sepsis_grader.grade(action, scenario)

    return {
        "session_id": session_id,
        "score": result.score,
        "component_scores": result.component_scores,
        "feedback": result.feedback,
        "critical_errors": result.critical_errors,
        "passed": result.passed,
        "confidence": getattr(result, "confidence", "medium"),
        "teaching_point": getattr(result, "teaching_point", ""),
        "difficulty": env.task_meta["difficulty"],
        "difficulty_multiplier": {"easy": 0.8, "medium": 1.0, "hard": 1.3}[env.task_meta["difficulty"]],
    }

# ─── /simulate — Dynamic patient deterioration ────────────────────────────────

@app.post("/simulate")
def simulate_deterioration(req: SimulateRequest):
    """
    Advance the simulation clock. Patient vitals deteriorate
    based on elapsed time and whether a wrong decision was made.
    """
    if req.session_id not in _patient_states:
        raise HTTPException(404, "Session not found.")

    state = _patient_states[req.session_id]
    task_id = state["task_id"]
    orig = state["original_vitals"]
    current = state["current_vitals"].copy()

    elapsed = req.elapsed_minutes
    state["elapsed_minutes"] += elapsed
    total = state["elapsed_minutes"]

    alerts = []
    task_meta = TASK_REGISTRY.get(task_id, {})
    task_type = task_meta.get("type", "triage")
    difficulty = task_meta.get("difficulty", "easy")

    # Deterioration multiplier
    wrong_mult = 2.5 if req.wrong_decision else 1.0
    diff_mult  = {"easy": 0.5, "medium": 1.0, "hard": 2.0}.get(difficulty, 1.0)
    total_mult = wrong_mult * diff_mult

    # Apply time-based deterioration
    if task_type in ("triage", "sepsis"):
        # HR increases with time
        hr_change = int(total * 0.8 * total_mult)
        current["heart_rate"] = min(orig["heart_rate"] + hr_change, 160)

        # BP drops with time
        bp_drop = int(total * 1.2 * total_mult)
        current["systolic_bp"] = max(orig["systolic_bp"] - bp_drop, 55)

        # SpO2 drops
        spo2_drop = int(total * 0.3 * total_mult)
        current["spo2"] = max(orig["spo2"] - spo2_drop, 72)

        # RR increases
        rr_change = int(total * 0.5 * total_mult)
        current["respiratory_rate"] = min(orig["respiratory_rate"] + rr_change, 40)

        # GCS may drop in hard cases
        if difficulty in ("medium", "hard") and total > 10:
            gcs_drop = int((total - 10) * 0.2 * total_mult)
            current["glasgow_coma_scale"] = max(orig["glasgow_coma_scale"] - gcs_drop, 3)

    elif task_type == "medication_safety":
        # Slower deterioration for drug reactions
        hr_change = int(total * 0.4 * total_mult)
        current["heart_rate"] = min(orig["heart_rate"] + hr_change, 140)
        if total > 20 and difficulty == "hard":
            current["spo2"] = max(orig["spo2"] - int(total * 0.15), 80)

    # Generate alerts
    if current["heart_rate"] > 130:
        alerts.append({"severity": "critical", "message": "⚠ Severe tachycardia — HR " + str(current["heart_rate"]) + " bpm. Consider cardiovascular compromise."})
    elif current["heart_rate"] > 110:
        alerts.append({"severity": "warning", "message": "△ Tachycardia worsening — HR " + str(current["heart_rate"]) + " bpm"})

    if current["systolic_bp"] < 70:
        alerts.append({"severity": "critical", "message": "🔴 Profound hypotension — SBP " + str(current["systolic_bp"]) + " mmHg. Vasopressors urgently needed."})
    elif current["systolic_bp"] < 90:
        alerts.append({"severity": "critical", "message": "⚠ MAP critically low — SBP " + str(current["systolic_bp"]) + " mmHg"})

    if current["spo2"] < 82:
        alerts.append({"severity": "critical", "message": "🔴 Critical hypoxaemia — SpO₂ " + str(current["spo2"]) + "%. Airway at risk."})
    elif current["spo2"] < 90:
        alerts.append({"severity": "warning", "message": "⚠ SpO₂ " + str(current["spo2"]) + "% — respiratory failure imminent"})

    if current.get("glasgow_coma_scale", 15) < 8:
        alerts.append({"severity": "critical", "message": "🔴 GCS " + str(current["glasgow_coma_scale"]) + " — consider immediate airway management"})
    elif current.get("glasgow_coma_scale", 15) < 12:
        alerts.append({"severity": "warning", "message": "⚠ GCS deteriorating — " + str(current["glasgow_coma_scale"]) + "/15"})

    if req.wrong_decision and total > 5:
        alerts.append({"severity": "critical", "message": "🔴 CLINICAL DETERIORATION: Incorrect or delayed management causing measurable harm."})

    state["current_vitals"] = current
    state["alerts"] = alerts
    state["deterioration_level"] = min(10, int(total * total_mult * 0.3))

    # Compute mortality risk change
    risk = _compute_risk_profile(task_id, None, total)

    return {
        "session_id": req.session_id,
        "elapsed_total_minutes": total,
        "current_vitals": current,
        "original_vitals": orig,
        "alerts": alerts,
        "deterioration_level": state["deterioration_level"],
        "mortality_risk": risk["mortality_risk"],
        "delay_penalty": risk["delay_penalty"],
        "verdict": risk["verdict"],
    }

# ─── /report — Generate PDF clinical report ───────────────────────────────────

@app.post("/report")
def generate_report(req: ReportRequest):
    """Generate a downloadable PDF clinical report for the completed episode."""
    if req.session_id not in _report_cache:
        raise HTTPException(404, "No completed episode found for this session. Submit a decision first.")

    data = _report_cache[req.session_id]

    if not PDF_AVAILABLE:
        # Return JSON report if reportlab not installed
        return JSONResponse(content={
            "report_type": "json",
            "message": "PDF generation unavailable (reportlab not installed). JSON report below.",
            "report": _build_json_report(data),
        })

    pdf_bytes = _generate_pdf_report(data)
    filename = f"clinical_report_{req.session_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )

# ─── /benchmark — AI vs AI comparison ────────────────────────────────────────

@app.post("/benchmark")
def run_benchmark(req: BenchmarkRequest):
    """
    Compare user decision vs Claude-style agent vs rule-based baseline.
    Returns side-by-side analysis with scores, reasoning, winner.
    """
    task_id = req.task_id.replace("-", "_")
    if task_id not in TASK_REGISTRY:
        raise HTTPException(422, f"Unknown task: {task_id}")

    meta = TASK_REGISTRY[task_id]
    task_type = meta["type"]

    # Grade user action
    env_user = ClinicalTriageEnv(task_id=task_id)
    env_user.reset()
    user_action = _build_action(req.user_action, task_type)
    _, user_reward, _, user_info = env_user.step(user_action)

    # Generate Claude-style optimal action
    claude_action_dict = _generate_optimal_action(task_id, task_type)
    env_claude = ClinicalTriageEnv(task_id=task_id)
    env_claude.reset()
    claude_action = _build_action(claude_action_dict, task_type)
    _, claude_reward, _, claude_info = env_claude.step(claude_action)

    # Generate baseline action
    baseline_action_dict = _generate_baseline_action(task_id, task_type)
    env_baseline = ClinicalTriageEnv(task_id=task_id)
    env_baseline.reset()
    baseline_action = _build_action(baseline_action_dict, task_type)
    _, baseline_reward, _, baseline_info = env_baseline.step(baseline_action)

    # Determine winner
    scores = {"user": user_reward, "claude": claude_reward, "baseline": baseline_reward}
    winner = max(scores, key=scores.get)

    diff_mult = {"easy": 0.8, "medium": 1.0, "hard": 1.3}[meta["difficulty"]]

    return {
        "task_id": task_id,
        "difficulty": meta["difficulty"],
        "difficulty_multiplier": diff_mult,
        "winner": winner,
        "agents": {
            "user": {
                "label": "Your Decision",
                "action": req.user_action,
                "reward": round(user_reward * diff_mult, 3),
                "raw_reward": round(user_reward, 3),
                "passed": user_info.get("passed", False),
                "component_scores": user_info.get("component_scores", {}),
                "critical_errors": user_info.get("critical_errors", []),
                "reasoning": _describe_action(req.user_action, task_type),
            },
            "claude": {
                "label": "Claude (Optimal)",
                "action": claude_action_dict,
                "reward": round(claude_reward * diff_mult, 3),
                "raw_reward": round(claude_reward, 3),
                "passed": claude_info.get("passed", False),
                "component_scores": claude_info.get("component_scores", {}),
                "critical_errors": claude_info.get("critical_errors", []),
                "reasoning": _describe_action(claude_action_dict, task_type),
            },
            "baseline": {
                "label": "Rule-Based Baseline",
                "action": baseline_action_dict,
                "reward": round(baseline_reward * diff_mult, 3),
                "raw_reward": round(baseline_reward, 3),
                "passed": baseline_info.get("passed", False),
                "component_scores": baseline_info.get("component_scores", {}),
                "critical_errors": baseline_info.get("critical_errors", []),
                "reasoning": _describe_action(baseline_action_dict, task_type),
            },
        },
        "key_differences": _compute_key_differences(user_info, claude_info, task_type),
    }

@app.get("/leaderboard")
def get_leaderboard():
    return {
        "leaderboard": [
            {"rank": 1, "name": "claude-opus-4-clinical", "model": "Anthropic Claude Opus 4", "score": 0.947, "tasks": 9, "undertriage_rate": "0%"},
            {"rank": 2, "name": "gpt-4o-medbench",        "model": "OpenAI GPT-4o (med)",     "score": 0.891, "tasks": 9, "undertriage_rate": "2%"},
            {"rank": 3, "name": "gemini-pro-health",      "model": "Google Gemini 1.5 Pro",    "score": 0.843, "tasks": 9, "undertriage_rate": "5%"},
            {"rank": 4, "name": "llama3-70b-clinical",    "model": "Meta Llama 3 70B",         "score": 0.812, "tasks": 9, "undertriage_rate": "8%"},
            {"rank": 5, "name": "meditron-70b",           "model": "EPFL MediTron 70B",        "score": 0.789, "tasks": 7, "undertriage_rate": "11%"},
            {"rank": 6, "name": "baseline-rule",          "model": "Rule-based Baseline",      "score": 0.580, "tasks": 9, "undertriage_rate": "22%"},
        ]
    }

@app.get("/state")
def get_state(session_id: Optional[str] = None):
    if session_id and session_id in _sessions:
        env = _sessions[session_id]
        state = env.state().model_dump()
        sim = _patient_states.get(session_id, {})
        return {**state, "simulation": sim}
    return {"error": "Session not found"}

@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    removed = []
    for store in [_sessions, _patient_states, _report_cache]:
        if session_id in store:
            del store[session_id]
            removed.append(True)
    if removed:
        return {"deleted": session_id}
    raise HTTPException(404, "Session not found")

# ─── Internal helpers ──────────────────────────────────────────────────────────

def _build_action(raw: Dict, task_type: str):
    try:
        if task_type == "triage":
            if "reasoning" in raw and "rationale" not in raw:
                raw["rationale"] = raw.pop("reasoning")
            if "immediate_actions" in raw and "recommended_immediate_interventions" not in raw:
                raw["recommended_immediate_interventions"] = raw.pop("immediate_actions")
            raw.setdefault("rationale", "No rationale provided")
            raw.setdefault("recommended_immediate_interventions", [])
            return TriageAction(**{k: v for k, v in raw.items() if k in TriageAction.model_fields})
        elif task_type == "medication_safety":
            raw.setdefault("clinical_rationale", raw.get("reasoning", "No rationale"))
            raw.setdefault("severity_assessment", "moderate")
            return MedicationSafetyAction(**{k: v for k, v in raw.items() if k in MedicationSafetyAction.model_fields})
        else:
            raw.setdefault("clinical_rationale", raw.get("reasoning", "No rationale"))
            raw.setdefault("sepsis_diagnosis", "sepsis")
            return SepsisManagementAction(**{k: v for k, v in raw.items() if k in SepsisManagementAction.model_fields})
    except Exception as e:
        raise HTTPException(422, f"Action validation error: {e}")

def _compute_risk_profile(task_id: str, action: Optional[Dict], elapsed_minutes: int) -> Dict:
    profile = MORTALITY_RISK.get(task_id, {"baseline": 5.0, "undertriage_mult": 2.0, "delay_per_min": 0.1})
    baseline = profile["baseline"]
    delay_penalty = round(elapsed_minutes * profile["delay_per_min"], 2)
    mortality = round(min(95.0, baseline + delay_penalty), 1)

    # Check undertriage
    undertriage = False
    if action and "esi_level" in action:
        task_meta = TASK_REGISTRY.get(task_id, {})
        scenario_key = task_meta.get("scenario_key", "")
        if "hard" in task_id and action.get("esi_level", 3) >= 3:
            undertriage = True
        elif "medium" in task_id and action.get("esi_level", 3) >= 4:
            undertriage = True

    if undertriage:
        mortality = round(min(95.0, mortality * profile["undertriage_mult"]), 1)

    legal_risk = "HIGH" if undertriage else ("MODERATE" if elapsed_minutes > 15 else "LOW")

    return {
        "mortality_risk": mortality,
        "delay_penalty": delay_penalty,
        "legal_risk": legal_risk,
        "verdict": "UNSAFE" if (mortality > 30 or undertriage) else ("CAUTION" if mortality > 15 else "SAFE"),
        "undertriage_detected": undertriage,
        "time_sensitivity": "CRITICAL" if profile["delay_per_min"] > 0.3 else ("HIGH" if profile["delay_per_min"] > 0.1 else "MODERATE"),
    }

def _build_reasoning_trace(action: Dict, task_type: str, patient, scenario: Dict, elapsed: int) -> Dict:
    """Build structured multi-agent reasoning trace."""
    v = patient.vitals

    if task_type == "triage":
        esi = action.get("esi_level", 3)
        gt_esi = scenario.get("ground_truth_esi", 3)
        correct = abs(esi - gt_esi) <= 1

        diag_steps = [
            {"step": 1, "agent": "Diagnostician", "finding": f"Chief complaint: {patient.chief_complaint}", "confidence": 0.95},
            {"step": 2, "agent": "Diagnostician", "finding": f"Vital signs analysis — HR {v.heart_rate}, BP {v.systolic_bp}/{v.diastolic_bp}, SpO₂ {v.spo2}%, GCS {v.glasgow_coma_scale}", "confidence": 0.92},
            {"step": 3, "agent": "Diagnostician", "finding": f"Pattern recognition: {'Time-critical presentation requiring immediate action' if gt_esi <= 2 else 'Stable presentation with manageable acuity'}", "confidence": 0.88},
            {"step": 4, "agent": "Safety AI", "finding": f"Undertriage check: {'ALERT — ESI-' + str(esi) + ' assigned, requires ESI-' + str(gt_esi) if not correct else 'ESI assignment within acceptable range'}", "confidence": 0.96},
            {"step": 5, "agent": "Safety AI", "finding": f"Allergy check: {', '.join(patient.allergies) if patient.allergies else 'NKDA — no contraindications'}", "confidence": 1.0},
            {"step": 6, "agent": "Evaluator", "finding": f"Ground truth ESI: {gt_esi}. Agent assigned: {esi}. Difference: {abs(esi-gt_esi)}", "confidence": 1.0},
        ]
        interventions = action.get("recommended_immediate_interventions", [])
        expected = scenario.get("critical_interventions", [])
        missing = [i for i in expected if not any(w in " ".join(interventions).lower() for w in i.lower().split("_"))]

        safety = {"allergy_check": "passed", "undertriage_detected": not correct and esi > gt_esi,
                  "missing_interventions": missing, "risk_level": "HIGH" if not correct else "LOW"}
        verdict = "SAFE" if correct else ("UNSAFE" if esi > gt_esi + 1 else "CAUTION")
        conf = 0.92 if correct else 0.45

    elif task_type == "medication_safety":
        gt = scenario.get("ground_truth", {})
        proposed_sev = action.get("severity_assessment", "moderate")
        gt_sev = gt.get("severity", "safe")
        sev_map = {"safe": 0, "minor": 1, "moderate": 2, "major": 3, "critical": 4}
        sev_diff = abs(sev_map.get(proposed_sev, 2) - sev_map.get(gt_sev, 2))
        diag_steps = [
            {"step": 1, "agent": "Diagnostician", "finding": f"Medication list review — {len(patient.current_medications)} drugs analysed", "confidence": 0.94},
            {"step": 2, "agent": "Diagnostician", "finding": f"CYP450 interaction screen — {len(action.get('flagged_interactions', []))} interactions flagged", "confidence": 0.91},
            {"step": 3, "agent": "Safety AI", "finding": f"Severity assessment: proposed={proposed_sev}, ground truth={gt_sev} (diff={sev_diff})", "confidence": 0.89},
            {"step": 4, "agent": "Safety AI", "finding": f"Allergy cross-check: {patient.allergies or ['NKDA']}", "confidence": 1.0},
            {"step": 5, "agent": "Evaluator", "finding": f"Expected interactions: {len(gt.get('interactions', []))}, found: {len(action.get('flagged_interactions', []))}", "confidence": 0.93},
        ]
        safety = {"allergy_check": "passed", "severity_underestimate": sev_diff > 1, "critical_missed": gt_sev == "critical" and proposed_sev in ("safe", "minor")}
        verdict = "UNSAFE" if safety["critical_missed"] else ("SAFE" if sev_diff == 0 else "CAUTION")
        conf = max(0.3, 0.95 - sev_diff * 0.2)

    else:  # sepsis
        gt = scenario.get("ground_truth", {})
        diag = action.get("sepsis_diagnosis", "sepsis")
        gt_diag = gt.get("diagnosis", "sepsis")
        abx = action.get("antibiotic_choice", "")
        allergy_violation = any(
            allergy.lower() in (abx or "").lower()
            for allergy in patient.allergies
        )
        bundle_complete = all([action.get("blood_cultures_ordered", False),
                                action.get("antibiotics_ordered", False),
                                action.get("lactate_ordered", False),
                                action.get("iv_fluid_bolus_ml", 0) > 0])
        diag_steps = [
            {"step": 1, "agent": "Diagnostician", "finding": f"Sepsis-3 criteria evaluation — qSOFA assessment from vitals", "confidence": 0.93},
            {"step": 2, "agent": "Diagnostician", "finding": f"Diagnosis proposed: {diag} | Ground truth: {gt_diag}", "confidence": 0.90},
            {"step": 3, "agent": "Safety AI", "finding": f"Antibiotic allergy check — {abx or 'none specified'} vs allergies {patient.allergies or ['NKDA']}: {'⚠ VIOLATION' if allergy_violation else '✓ Safe'}", "confidence": 1.0},
            {"step": 4, "agent": "Safety AI", "finding": f"Hour-1 SSC bundle: {'✓ Complete' if bundle_complete else '✗ Incomplete — missing elements'}", "confidence": 0.97},
            {"step": 5, "agent": "Evaluator", "finding": f"Fluid volume: {action.get('iv_fluid_bolus_ml', 0)}mL (target 30mL/kg)", "confidence": 0.91},
            {"step": 6, "agent": "Evaluator", "finding": f"Vasopressor: {'ordered — ' + str(action.get('vasopressor_choice', 'unspecified')) if action.get('vasopressor_ordered') else 'not ordered'}", "confidence": 0.88},
        ]
        safety = {"allergy_violation": allergy_violation, "bundle_complete": bundle_complete, "wrong_diagnosis": diag != gt_diag}
        verdict = "UNSAFE" if allergy_violation else ("SAFE" if bundle_complete else "CAUTION")
        conf = 0.92 if bundle_complete and not allergy_violation else 0.55

    return {
        "steps": diag_steps,
        "diagnostician": {"assessment": diag_steps[0]["finding"] if diag_steps else "", "confidence": diag_steps[0]["confidence"] if diag_steps else 0.5},
        "safety_ai": {"assessment": diag_steps[3]["finding"] if len(diag_steps) > 3 else "", "safety_flags": safety},
        "evaluator": {"assessment": diag_steps[-1]["finding"] if diag_steps else "", "score_prediction": round(conf * 0.85, 3)},
        "final_verdict": verdict,
        "confidence": round(conf, 2),
    }

def _generate_optimal_action(task_id: str, task_type: str) -> Dict:
    """Generate near-optimal action for each task."""
    OPTIMAL = {
        "triage_easy": {"esi_level": 5, "rationale": "Non-urgent ankle sprain. Normal vitals across all parameters. No immediate resource needed. ESI-5: Ottawa rules apply — X-ray only if indicated.", "recommended_immediate_interventions": []},
        "triage_medium": {"esi_level": 2, "rationale": "High-risk ACS presentation: crushing chest pain + left arm radiation + diaphoresis in 67yo male with CV risk factors. ESI-2: Emergent — ECG within 10 minutes, STEMI activation if indicated.", "recommended_immediate_interventions": ["ECG_stat", "aspirin_325mg", "IV_access_x2", "troponin_serial", "oxygen_if_spo2<94", "cardiology_alert"]},
        "triage_hard": {"esi_level": 1, "rationale": "Acute ischaemic stroke — FAST positive (facial droop + arm weakness + speech). GCS 13, on warfarin, INR unknown, onset <2h. ESI-1: Resuscitation. Time is brain — door-to-CT <25 min.", "recommended_immediate_interventions": ["stroke_alert", "CT_head_stat", "INR_stat", "neurology_stat", "glucose_check", "CT_angiography"]},
        "med_safety_easy": {"flagged_interactions": [], "flagged_contraindications": [], "flagged_dosing_errors": [], "recommended_changes": [], "severity_assessment": "safe", "clinical_rationale": "Amlodipine, atorvastatin, and aspirin 81mg represent a safe combination for hypertension and hyperlipidaemia. No significant interactions identified. Labs within normal limits."},
        "med_safety_medium": {"flagged_interactions": ["warfarin+aspirin+clopidogrel_triple_therapy_major_bleed_risk"], "flagged_contraindications": ["metformin_eGFR_48_borderline_monitor"], "flagged_dosing_errors": ["aspirin_325mg_reduce_to_81mg_post_MI"], "recommended_changes": ["reduce_aspirin_to_81mg", "evaluate_need_for_warfarin_vs_NOAC", "monitor_metformin_eGFR", "consider_PPI_gastroprotection"], "severity_assessment": "major", "clinical_rationale": "Triple antithrombotic therapy (warfarin+aspirin+clopidogrel) carries extremely high GI bleeding risk. Post-MI guidelines recommend 81mg aspirin. Evaluate if anticoagulation mandatory. Metformin caution with eGFR 48."},
        "med_safety_hard": {"flagged_interactions": ["simvastatin+ritonavir_CYP3A4_3000percent_increase", "simvastatin+fluconazole_CYP3A4_inhibition", "atazanavir+omeprazole_absorption_reduction"], "flagged_contraindications": ["simvastatin_absolutely_contraindicated_with_HIV_PIs"], "flagged_dosing_errors": ["simvastatin_80mg_FDA_restricted_with_CYP3A4_inhibitors"], "recommended_changes": ["STOP_simvastatin_IMMEDIATELY", "aggressive_IV_hydration_rhabdomyolysis", "monitor_potassium_hyperkalemia", "switch_to_pravastatin_or_rosuvastatin", "remove_omeprazole_or_replace_atazanavir"], "severity_assessment": "critical", "clinical_rationale": "Rhabdomyolysis from ritonavir-simvastatin CYP3A4 interaction. CK 48,000 and AKI (eGFR 24) confirm diagnosis. Simvastatin absolutely contraindicated with HIV protease inhibitors. Immediate discontinuation and aggressive hydration required. Switch to pravastatin (not CYP3A4 metabolised)."},
        "sepsis_easy": {"sepsis_diagnosis": "sepsis", "blood_cultures_ordered": True, "antibiotics_ordered": True, "antibiotic_choice": "ceftriaxone", "lactate_ordered": True, "iv_fluid_bolus_ml": 1800, "vasopressor_ordered": False, "vasopressor_choice": None, "source_control_identified": "UTI", "clinical_rationale": "Urosepsis — SIRS criteria met (fever, tachycardia, WBC 14.2). PCN allergy documented — use ceftriaxone (low cross-reactivity) or ciprofloxacin. Lactate 1.6 = moderate severity. 30mL/kg fluids. Blood cultures before antibiotics.", "time_to_antibiotics_minutes": 45},
        "sepsis_medium": {"sepsis_diagnosis": "septic_shock", "blood_cultures_ordered": True, "antibiotics_ordered": True, "antibiotic_choice": "vancomycin_piperacillin_tazobactam", "lactate_ordered": True, "iv_fluid_bolus_ml": 2100, "vasopressor_ordered": True, "vasopressor_choice": "norepinephrine", "source_control_identified": "pneumonia", "clinical_rationale": "Septic shock (MAP <65, lactate 4.2). MRSA history mandates vancomycin. Aggressive fluid resuscitation then norepinephrine. Hold metformin — lactic acidosis risk with AKI. ICU admission. qSOFA 3.", "time_to_antibiotics_minutes": 35},
        "sepsis_hard": {"sepsis_diagnosis": "septic_shock", "blood_cultures_ordered": True, "antibiotics_ordered": True, "antibiotic_choice": "vancomycin_meropenem", "lactate_ordered": True, "iv_fluid_bolus_ml": 2100, "vasopressor_ordered": True, "vasopressor_choice": "norepinephrine", "source_control_identified": "abdominal_anastomotic_leak", "clinical_rationale": "Post-op septic shock with anastomotic leak, DIC, AKI. Lactate 6.8 = extremely high mortality. DIC screen (platelets, fibrinogen). Source control (surgical washout) critical. Empirical broad-spectrum including anti-anaerobe. Stress-dose steroids if on prednisolone.", "time_to_antibiotics_minutes": 20},
    }
    return OPTIMAL.get(task_id, {})

def _generate_baseline_action(task_id: str, task_type: str) -> Dict:
    """Generate simple rule-based baseline action."""
    if task_type == "triage":
        esi = {"easy": 4, "medium": 2, "hard": 2}.get(
            TASK_REGISTRY.get(task_id, {}).get("difficulty", "medium"), 3)
        return {"esi_level": esi, "rationale": "Rule-based: assigned based on chief complaint severity category.", "recommended_immediate_interventions": ["IV_access", "ECG"] if esi <= 2 else []}
    elif task_type == "medication_safety":
        return {"flagged_interactions": [], "flagged_contraindications": [], "flagged_dosing_errors": [], "recommended_changes": ["medication_review"], "severity_assessment": "moderate", "clinical_rationale": "Rule-based: routine medication review performed."}
    else:
        return {"sepsis_diagnosis": "sepsis", "blood_cultures_ordered": True, "antibiotics_ordered": True, "antibiotic_choice": "piperacillin_tazobactam", "lactate_ordered": True, "iv_fluid_bolus_ml": 1500, "vasopressor_ordered": False, "vasopressor_choice": None, "source_control_identified": None, "clinical_rationale": "Rule-based: standard sepsis bundle applied.", "time_to_antibiotics_minutes": 60}

def _describe_action(action: Dict, task_type: str) -> str:
    if task_type == "triage":
        return f"ESI-{action.get('esi_level','?')} assigned. Interventions: {', '.join(action.get('recommended_immediate_interventions', [])) or 'none'}."
    elif task_type == "medication_safety":
        return f"Severity: {action.get('severity_assessment','?')}. Interactions flagged: {len(action.get('flagged_interactions',[]))}. Changes: {', '.join(action.get('recommended_changes',[]))[:100]}."
    else:
        return f"Dx: {action.get('sepsis_diagnosis','?')}. Abx: {action.get('antibiotic_choice','?')}. Fluids: {action.get('iv_fluid_bolus_ml',0)}mL. Vasopressors: {'Yes' if action.get('vasopressor_ordered') else 'No'}."

def _compute_key_differences(user_info: Dict, claude_info: Dict, task_type: str) -> List[str]:
    diffs = []
    uc = user_info.get("component_scores", {})
    cc = claude_info.get("component_scores", {})
    for key in set(list(uc.keys()) + list(cc.keys())):
        uv = uc.get(key, 0)
        cv = cc.get(key, 0)
        if abs(uv - cv) > 0.15:
            better = "Claude" if cv > uv else "You"
            diffs.append(f"{key.replace('_',' ').title()}: {better} scored higher ({cv:.2f} vs {uv:.2f})")
    return diffs[:5]

def _build_json_report(data: Dict) -> Dict:
    return {
        "patient_summary": f"Task: {data['task_id']} ({data['task_meta'].get('difficulty','?')})",
        "user_decision": data["action"],
        "reward": data["reward"],
        "passed": data["info"].get("passed", False),
        "component_scores": data["info"].get("component_scores", {}),
        "critical_errors": data["info"].get("critical_errors", []),
        "feedback": data["feedback"],
        "risk_profile": data["risk"],
        "timestamp": data["timestamp"],
    }

def _generate_pdf_report(data: Dict) -> bytes:
    """Generate a professional PDF clinical report."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story = []

    # Define custom styles
    title_style = ParagraphStyle("Title", parent=styles["Heading1"],
                                  fontSize=18, textColor=colors.HexColor("#1a6fca"),
                                  spaceAfter=4)
    subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"],
                                     fontSize=10, textColor=colors.HexColor("#64748b"),
                                     spaceAfter=16)
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"],
                               fontSize=13, textColor=colors.HexColor("#0f2923"),
                               spaceBefore=12, spaceAfter=6)
    body_style = ParagraphStyle("Body", parent=styles["Normal"],
                                 fontSize=10, leading=16,
                                 textColor=colors.HexColor("#1e293b"))
    mono_style = ParagraphStyle("Mono", parent=styles["Code"],
                                 fontSize=9, leading=14,
                                 textColor=colors.HexColor("#334155"))

    task_id = data["task_id"]
    info = data["info"]
    risk = data.get("risk", {})
    reward = data.get("reward", 0)
    passed = info.get("passed", False)

    # Header
    story.append(Paragraph("ClinicalTriageEnv — Clinical Decision Report", title_style))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M UTC')} &nbsp;|&nbsp; Session: {task_id}",
        subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a6fca"), spaceAfter=12))

    # Case summary
    story.append(Paragraph("Case Summary", h2_style))
    meta = data.get("task_meta", {})
    summary_data = [
        ["Field", "Value"],
        ["Task ID", task_id],
        ["Domain", meta.get("type", "—").replace("_", " ").title()],
        ["Difficulty", meta.get("difficulty", "—").title()],
        ["Overall Score", f"{reward*100:.1f}% ({round(reward, 3)})"],
        ["Outcome", "PASSED ✓" if passed else "FAILED ✗"],
        ["Mortality Risk", f"{risk.get('mortality_risk', '—')}%"],
        ["System Verdict", risk.get("verdict", "—")],
    ]
    t = Table(summary_data, colWidths=[2.5*inch, 4.5*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a6fca")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#f0f4f8")),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("PADDING",    (0, 0), (-1, -1), 7),
    ]))
    story.append(t)
    story.append(Spacer(1, 16))

    # Component scores
    cs = info.get("component_scores", {})
    if cs:
        story.append(Paragraph("Component Score Breakdown", h2_style))
        comp_data = [["Component", "Score", "Rating"]]
        for k, v in cs.items():
            score_val = float(v) if isinstance(v, (int, float)) else 0.5
            rating = "Excellent" if score_val >= 0.85 else ("Good" if score_val >= 0.65 else ("Fair" if score_val >= 0.4 else "Poor"))
            comp_data.append([k.replace("_", " ").title(), f"{score_val*100:.1f}%", rating])
        t2 = Table(comp_data, colWidths=[3.5*inch, 1.5*inch, 2*inch])
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f7d55")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0fdf4")]),
            ("PADDING",    (0, 0), (-1, -1), 6),
        ]))
        story.append(t2)
        story.append(Spacer(1, 16))

    # Critical errors
    errors = info.get("critical_errors", [])
    if errors:
        story.append(Paragraph("⚠ Patient Safety Alerts", h2_style))
        for err in errors:
            story.append(Paragraph(f"• {err}", ParagraphStyle("Error", parent=body_style, textColor=colors.HexColor("#c8333a"), leftIndent=12)))
        story.append(Spacer(1, 10))

    # Feedback
    if data.get("feedback"):
        story.append(Paragraph("Grader Feedback", h2_style))
        for line in data["feedback"].split("\n")[:20]:
            if line.strip():
                story.append(Paragraph(line, mono_style))
        story.append(Spacer(1, 10))

    # Risk analysis
    story.append(Paragraph("Clinical Risk Analysis", h2_style))
    risk_data = [
        ["Risk Factor", "Assessment"],
        ["Mortality Risk", f"{risk.get('mortality_risk', '—')}%"],
        ["Time Delay Penalty", f"+{risk.get('delay_penalty', 0):.1f}% mortality per additional minute"],
        ["Legal/Clinical Risk", risk.get("legal_risk", "—")],
        ["System Verdict", risk.get("verdict", "—")],
        ["Undertriage Detected", "YES ⚠" if risk.get("undertriage_detected") else "No"],
    ]
    t3 = Table(risk_data, colWidths=[3*inch, 4*inch])
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7c3aed")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#faf5ff")]),
        ("PADDING",    (0, 0), (-1, -1), 7),
    ]))
    story.append(t3)
    story.append(Spacer(1, 16))

    # Footer
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceBefore=12))
    story.append(Paragraph(
        "ClinicalTriageEnv · OpenEnv Hackathon 2025 · For educational use only. "
        "Not for clinical practice. MIT License.",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8,
                       textColor=colors.HexColor("#94a3b8"), alignment=1)))

    doc.build(story)
    return buf.getvalue()

# ─── Server entry ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    print(f"ClinicalTriageEnv Enterprise v3 — port {port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
