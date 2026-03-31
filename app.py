import os
    import sys
import uuid
                       from typing import Any, Dict, List, Optional

sys.path.insert(0, "/app")

from fastapi import FastAPI, HTTPException
             from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
               from pydantic import BaseModel

from models import (
    TriageAction, MedicationSafetyAction, SepsisManagementAction, ClinicalState
)
from environment import ClinicalTriageEnv, TASK_REGISTRY

app = FastAPI(
    title="ClinicalTriageEnv",
    version="1.0.0",
    description="Real-world clinical AI training environment — OpenEnv compatible",
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

# ─── Session store ────────────────────────────────────────────────────────────
_sessions: Dict[str, ClinicalTriageEnv] = {}
_DEFAULT_TASK = "triage_easy"


                    def _get_or_create_session(session_id: str, task_id: str) -> ClinicalTriageEnv:
    if session_id not in _sessions:
        _sessions[session_id] = ClinicalTriageEnv(task_id=task_id)
    return _sessions[session_id]


# ─── Request / Response models ────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = _DEFAULT_TASK
    session_id: Optional[str] = None


class TriageStepRequest(BaseModel):
    action: Dict[str, Any]
    session_id: Optional[str] = None


               class GenericStepRequest(BaseModel):
    action: Dict[str, Any]
    session_id: Optional[str] = None


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    """Serve the ICU Dashboard frontend."""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "ClinicalTriageEnv API running", "docs": "/docs"}


@app.get("/health")
                     def health():
    """Health check — always returns 200 while server is up."""
    return {
        "status": "healthy",
        "service": "ClinicalTriageEnv",
        "version": "1.0.0",
        "tasks_available": len(TASK_REGISTRY),
        "active_sessions": len(_sessions),
    }


@app.get("/api/test")
def api_test():
    """Legacy test endpoint — kept for backwards compatibility."""
    return {"status": "API working", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    """Return all available tasks with metadata."""
    return {
        "tasks": [
            {
                "id": k,
                "name": v["name"],
                "type": v["type"],
                "difficulty": v["difficulty"],
                "max_steps": v["max_steps"],
                "description": v["description"],
            }
            for k, v in TASK_REGISTRY.items()
        ],
        "total": len(TASK_REGISTRY),
    }


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    """Return metadata for a single task."""
    if task_id not in TASK_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    t = TASK_REGISTRY[task_id]
    return {
        "id": task_id,
        "name": t["name"],
        "type": t["type"],
        "difficulty": t["difficulty"],
        "max_steps": t["max_steps"],
        "description": t["description"],
    }


@app.post("/reset")
def reset_episode(req: ResetRequest):
    """
    Start a new episode for the given task.
    Body:
        task_id: str  — one of the 9 task IDs (default: triage_easy)
        session_id: str  — optional client session identifier
    Returns:
        Full observation (patient data, vitals, task description, etc.)
    """
    task_id = req.task_id or _DEFAULT_TASK

    # Accept both underscore and hyphen variants
    task_id = task_id.replace("-", "_")

    if task_id not in TASK_REGISTRY:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown task_id '{task_id}'. Valid: {list(TASK_REGISTRY.keys())}",
        )

    session_id = req.session_id or str(uuid.uuid4())

    # Always create a fresh env on reset
    env = ClinicalTriageEnv(task_id=task_id)
    _sessions[session_id] = env

    obs = env.reset(task_id=task_id)

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
    }


@app.post("/step")
def step_episode(req: GenericStepRequest):
    """
    Submit a clinical action for the current episode.
    Body:
        action: dict  — see task-specific action schemas below
        session_id: str  — session returned by /reset
    Action schemas by task type:
    TRIAGE tasks:
        {
          "esi_level": 1-5,
          "rationale": "...",
          "recommended_immediate_interventions": ["ECG", "IV_access", ...]
        }
    MEDICATION SAFETY tasks:
        {
          "flagged_interactions": ["simvastatin+clarithromycin"],
          "flagged_contraindications": [...],
          "flagged_dosing_errors": [...],
          "recommended_changes": [...],
          "severity_assessment": "critical|major|moderate|minor|safe",
          "clinical_rationale": "..."
        }
    SEPSIS tasks:
        {
          "sepsis_diagnosis": "sepsis|septic_shock|SIRS_only|no_sepsis",
          "blood_cultures_ordered": true,
          "antibiotics_ordered": true,
          "antibiotic_choice": "piperacillin_tazobactam",
          "lactate_ordered": true,
          "iv_fluid_bolus_ml": 1500,
          "vasopressor_ordered": false,
          "vasopressor_choice": null,
          "source_control_identified": "UTI",
          "clinical_rationale": "...",
          "time_to_antibiotics_minutes": 45
        }
    Returns:
        observation, reward (0-1.5), done, info (grade details, component scores)
    """
    session_id = req.session_id
    raw_action = req.action

    # Find session
    if session_id and session_id in _sessions:
        env = _sessions[session_id]
    else:
        # Fallback: create default env
        session_id = str(uuid.uuid4())
        env = ClinicalTriageEnv(task_id=_DEFAULT_TASK)
        _sessions[session_id] = env

    task_type = env.task_meta["type"]

    # Build typed action
    try:
        if task_type == "triage":
            # Accept both 'reasoning' (old frontend) and 'rationale' (schema)
            if "reasoning" in raw_action and "rationale" not in raw_action:
                raw_action["rationale"] = raw_action.pop("reasoning")
            if "immediate_actions" in raw_action and "recommended_immediate_interventions" not in raw_action:
                raw_action["recommended_immediate_interventions"] = raw_action.pop("immediate_actions")
            # Ensure required fields
            if "rationale" not in raw_action:
                raw_action["rationale"] = "No rationale provided"
            if "esi_level" not in raw_action:
                raise HTTPException(status_code=422, detail="esi_level required for triage tasks")
            action = TriageAction(**raw_action)

        elif task_type == "medication_safety":
            if "clinical_rationale" not in raw_action:
                raw_action["clinical_rationale"] = raw_action.get("reasoning", "No rationale provided")
            if "severity_assessment" not in raw_action:
                raw_action["severity_assessment"] = "moderate"
            action = MedicationSafetyAction(**raw_action)

        elif task_type == "sepsis":
            if "clinical_rationale" not in raw_action:
                raw_action["clinical_rationale"] = raw_action.get("reasoning", "No rationale provided")
            if "sepsis_diagnosis" not in raw_action:
                raw_action["sepsis_diagnosis"] = "sepsis"
            action = SepsisManagementAction(**raw_action)

        else:
            raise HTTPException(status_code=400, detail=f"Unknown task type: {task_type}")

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Action validation error: {str(e)}")

    # Execute
    obs, reward, done, info = env.step(action)

    return {
        "session_id": session_id,
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
        # Convenience fields for the frontend
        "score": round(reward, 4),
        "passed": info.get("passed", False),
        "grade": info.get("grade", reward),
        "component_scores": info.get("component_scores", {}),
        "critical_errors": info.get("critical_errors", []),
        "feedback": obs.feedback if hasattr(obs, "feedback") else "",
        "total_reward": info.get("total_reward", reward),
        "task_id": env.task_id,
        "difficulty": env.task_meta["difficulty"],
    }


@app.get("/state")
def get_state(session_id: Optional[str] = None):
    """Return current episode state for a session."""
    if session_id and session_id in _sessions:
        env = _sessions[session_id]
        return env.state().model_dump()
    elif _sessions:
        # Return most recent session
        last = list(_sessions.values())[-1]
        return last.state().model_dump()
    return {"error": "No active session", "hint": "Call /reset first"}


@app.get("/leaderboard")
def leaderboard():
    """Return mock leaderboard data (extend with real DB as needed)."""
    return {
        "leaderboard": [
            {"rank": 1, "name": "claude-opus-4-clinical", "model": "Anthropic Claude Opus 4", "score": 0.947, "tasks": 9},
            {"rank": 2, "name": "gpt-4o-medbench", "model": "OpenAI GPT-4o (med tuned)", "score": 0.891, "tasks": 9},
            {"rank": 3, "name": "gemini-pro-health", "model": "Google Gemini 1.5 Pro", "score": 0.843, "tasks": 9},
            {"rank": 4, "name": "llama3-70b-clinical", "model": "Meta Llama 3 70B", "score": 0.812, "tasks": 9},
            {"rank": 5, "name": "meditron-70b", "model": "EPFL MediTron 70B", "score": 0.789, "tasks": 7},
            {"rank": 6, "name": "baseline-rule", "model": "Rule-based Baseline", "score": 0.580, "tasks": 9},
        ]
    }


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Clean up a session."""
    if session_id in _sessions:
        del _sessions[session_id]
        return {"deleted": session_id}
    raise HTTPException(status_code=404, detail="Session not found")


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    print(f"Starting ClinicalTriageEnv on port {port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
