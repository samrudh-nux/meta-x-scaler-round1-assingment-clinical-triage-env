------
#Creator: Samrudh 
**-https://huggingface.co/spaces/samrudh-nux/my-healthcare-ev4u 
---
title: ClinicalTriageEnv
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---
# 🏥 ClinicalTriageEnv

**OpenEnv-compatible healthcare AI training environment for clinical decision-making agents.**

[![OpenEnv Spec 0.1](https://img.shields.io/badge/OpenEnv-0.1-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green)](https://python.org)
[![HF Space](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co/spaces)

---

## 🎯 Why This Environment?

Clinical decision-making is one of the highest-stakes reasoning tasks humans perform. Medical errors cause **250,000+ deaths annually** in the US alone. Yet there is **no standardised RL training environment** for evaluating AI agents on real clinical judgment tasks.

ClinicalTriageEnv fills this gap by simulating three tasks that physicians and nurses perform daily:

| Task | Real-World Analog | Stakes |
|------|------------------|--------|
| **ED Triage** | Emergency nurse assigns ESI level | Wrong level = patient deterioration |
| **Medication Safety** | Pharmacist reviews prescriptions | Missed interaction = ADR/death |
| **Sepsis Management** | Clinician executes Hour-1 bundle | Delay = 7% mortality increase/hour |

---

## 🏗️ Environment Architecture

```
┌─────────────────────────────────────────────────────┐
│              ClinicalTriageEnv                      │
│                                                     │
│  ┌────────────┐  ┌────────────┐  ┌──────────────┐  │
│  │  ED Triage │  │  Med Safety│  │   Sepsis Mgmt│  │
│  │  (3 tasks) │  │  (3 tasks) │  │   (3 tasks)  │  │
│  └────────────┘  └────────────┘  └──────────────┘  │
│         ↕                ↕               ↕          │
│  ┌──────────────────────────────────────────────┐   │
│  │         Programmatic Graders (0.0-1.0)       │   │
│  │  • Partial credit  • Safety penalties        │   │
│  │  • Deterministic   • Medical accuracy        │   │
│  └──────────────────────────────────────────────┘   │
│         ↕                                           │
│  ┌──────────────────────────────────────────────┐   │
│  │     FastAPI Server (HTTP + WebSocket)        │   │
│  │  POST /reset  POST /step  GET /state         │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## 📋 Tasks

### Task 1: Emergency Department Triage (ESI Level Assignment)

The **Emergency Severity Index (ESI)** is a 5-level triage algorithm used in 90%+ of US emergency departments.

| Level | Name | Action |
|-------|------|--------|
| ESI-1 | Resuscitation | Immediate life-saving intervention |
| ESI-2 | Emergent | High-risk / severe pain |
| ESI-3 | Urgent | Stable, needs 2+ resources |
| ESI-4 | Less Urgent | Needs 1 resource |
| ESI-5 | Non-urgent | No resources needed |

**Scenarios:**
- 🟢 **Easy**: Ankle sprain (ESI-5) — validates baseline understanding
- 🟡 **Medium**: Crushing chest pain + diaphoresis (ESI-2 ACS) — STEMI recognition
- 🔴 **Hard**: Acute stroke on warfarin (ESI-1) — anticoagulation + time-critical

**Grading:** ESI accuracy (65%) + rationale quality (20%) + critical interventions (15%)
Undertriage penalty: −40% for ESI-1/2 patients assigned ESI-3+

---

### Task 2: Medication Safety Review

Agents review patient medication lists for:
- **Drug-drug interactions** (e.g., warfarin + aspirin + clopidogrel = triple therapy)
- **Contraindications** (e.g., metformin in GFR<30, simvastatin with HIV PIs)
- **Dosing errors** (e.g., aspirin 325mg long-term post-MI should be 81mg)
- **Overall severity assessment**

**Scenarios:**
- 🟢 **Easy**: Clean hypertension/lipid regimen — tests ability to say "safe"
- 🟡 **Medium**: Post-MI triple antithrombotic therapy with CKD + diabetes
- 🔴 **Hard**: HIV patient on ritonavir + simvastatin presenting with rhabdomyolysis (CK=48,000)

**Grading:** Interaction detection (25%) + contraindication detection (20%) + dosing errors (15%) + severity (15%) + rationale (15%) + false-positive penalty (10%)

---

### Task 3: Sepsis Recognition & Hour-1 Bundle Management

Based on **Sepsis-3 criteria** and the **Surviving Sepsis Campaign Hour-1 Bundle**:
1. Blood cultures before antibiotics
2. Broad-spectrum antibiotics within 1 hour
3. Lactate measurement
4. 30 mL/kg crystalloid for hypotension/lactate≥4
5. Vasopressors if MAP<65 despite fluids

**Scenarios:**
- 🟢 **Easy**: Urosepsis with penicillin allergy — standard bundle with allergy awareness
- 🟡 **Medium**: Septic shock in elderly nursing home patient with MRSA history
- 🔴 **Hard**: Post-op anastomotic leak with multi-organ failure, DIC, vancomycin allergy

**Grading:** Diagnosis (20%) + bundle compliance (20%) + antibiotic choice (20%) + fluids (15%) + vasopressors (15%) + rationale (10%)

---

## 🎮 Action & Observation Spaces

### TriageAction
```python
class TriageAction(Action):
    esi_level: int                              # 1-5
    rationale: str                              # clinical reasoning
    recommended_immediate_interventions: List[str]  # e.g. ["ECG", "troponin"]
```

### MedicationSafetyAction
```python
class MedicationSafetyAction(Action):
    flagged_interactions: List[str]
    flagged_contraindications: List[str]
    flagged_dosing_errors: List[str]
    recommended_changes: List[str]
    severity_assessment: str                    # safe/minor/moderate/major/critical
    clinical_rationale: str
```

### SepsisManagementAction
```python
class SepsisManagementAction(Action):
    sepsis_diagnosis: str                       # sepsis/septic_shock/SIRS_only/no_sepsis
    blood_cultures_ordered: bool
    antibiotics_ordered: bool
    antibiotic_choice: Optional[str]
    lactate_ordered: bool
    iv_fluid_bolus_ml: int                      # 30mL/kg target
    vasopressor_ordered: bool
    vasopressor_choice: Optional[str]
    source_control_identified: Optional[str]
    clinical_rationale: str
    time_to_antibiotics_minutes: Optional[int]
```

### Observations
All observations include:
- `patient: PatientRecord` — full patient data (vitals, history, labs, medications)
- `task_description: str` — clear task instructions
- `current_step: int`, `max_steps: int`
- `feedback: str` — grader feedback (populated after step)
- `score_so_far: float`
- `done: bool`, `reward: Optional[float]`

---

## 🏆 Reward Function

```
reward = (grade_score - safety_penalty + efficiency_bonus) × difficulty_multiplier
```

| Component | Value |
|-----------|-------|
| Base grade | 0.0 – 1.0 |
| Safety penalty | −0.3 per critical error |
| Efficiency bonus | +0.05 × (max_steps − current_step) |
| Difficulty multiplier | Easy: 0.8 × Medium: 1.0 × Hard: 1.3 |

**Reward range:** −1.0 to +1.5

Partial progress signals are provided at each grading component, not just binary end-of-episode.

---

## 📊 Baseline Scores

Run with `meta-llama/Llama-3.3-70B-Instruct` via `inference.py`:

| Task | Score | Difficulty |
|------|-------|------------|
| triage_easy | 0.78 | 🟢 Easy |
| triage_medium | 0.62 | 🟡 Medium |
| triage_hard | 0.45 | 🔴 Hard |
| med_safety_easy | 0.90 | 🟢 Easy |
| med_safety_medium | 0.58 | 🟡 Medium |
| med_safety_hard | 0.31 | 🔴 Hard |
| sepsis_easy | 0.72 | 🟢 Easy |
| sepsis_medium | 0.55 | 🟡 Medium |
| sepsis_hard | 0.28 | 🔴 Hard |
| **Overall Mean** | **0.58** | — |

Hard tasks genuinely challenge frontier models because they require nuanced pharmacological knowledge and multi-organ reasoning.

---

## 🚀 Setup & Usage

### Local Setup

```bash
# Clone / download the project
cd clinical_triage_env
# Install dependencies
pip install -r requirements.txt
# Run the server
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
# In another terminal, test it:
curl http://localhost:7860/health
curl http://localhost:7860/tasks
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id": "triage_easy"}'
```

### Docker

```bash
# Build
docker build -t clinical-triage-env .
# Run
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct \
  clinical-triage-env
# Health check
curl http://localhost:7860/health
```

### Run Inference (Baseline)

```bash
# Set environment variables
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="your_hf_token"
# Run all 9 tasks
python inference.py
# Run specific tasks only
python inference.py --tasks triage_easy triage_hard sepsis_hard
# Save results to file
python inference.py --output my_results.json
```

### Python API

```python
from environment import ClinicalTriageEnv
from models import TriageAction
# Create environment
env = ClinicalTriageEnv(task_id="triage_medium")
# Reset
obs = env.reset()
print(f"Patient: {obs.patient.chief_complaint}")
print(f"Task: {obs.task_description}")
# Step with an action
action = TriageAction(
    esi_level=2,
    rationale="Crushing chest pain with diaphoresis in 67yo with cardiac risk factors suggests ACS. Time-critical.",
    recommended_immediate_interventions=["ECG", "troponin", "aspirin_325mg", "IV_access"]
)
obs, reward, done, info = env.step(action)
print(f"Score: {info['grade']:.3f}")
print(f"Reward: {reward:.4f}")
print(f"Passed: {info['passed']}")
print(f"Feedback:\n{obs.feedback}")
# State
state = env.state()
print(f"Episode: {state.episode_id}, Steps: {state.step_count}")
```

### WebSocket API

```python
import asyncio, json, websockets
async def run():
    async with websockets.connect("ws://localhost:7860/ws") as ws:
        # Reset
        await ws.send(json.dumps({"type": "reset", "task_id": "sepsis_medium"}))
        obs = json.loads(await ws.recv())
        
        # Step
        action = {
            "type": "step",
            "action": {
                "sepsis_diagnosis": "septic_shock",
                "blood_cultures_ordered": True,
                "antibiotics_ordered": True,
                "antibiotic_choice": "vancomycin_plus_piperacillin_tazobactam",
                "lactate_ordered": True,
                "iv_fluid_bolus_ml": 2100,
                "vasopressor_ordered": True,
                "vasopressor_choice": "norepinephrine",
                "clinical_rationale": "Septic shock criteria met: MAP<65, lactate>4, MRSA history"
            }
        }
        await ws.send(json.dumps(action))
        result = json.loads(await ws.recv())
        print(f"Score: {result['info']['grade']}")
asyncio.run(run())
```

---

## 🗂️ Project Structure

```
clinical_triage_env/
├── __init__.py              # Public API exports
├── models.py                # Typed Pydantic Action/Observation/State models
├── scenarios.py             # Realistic clinical scenarios with ground truth
├── graders.py               # Programmatic graders (0.0–1.0) for all 3 tasks
├── environment.py           # Main env: reset/step/state + reward shaping
├── inference.py             # Baseline inference script (OpenAI client)
├── openenv.yaml             # OpenEnv spec manifest
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Package configuration
├── Dockerfile               # Container definition
└── server/
    ├── __init__.py
    └── app.py               # FastAPI server (HTTP + WebSocket)
```

---

## 🔬 Medical Accuracy Notes

All scenarios are medically reviewed for accuracy:

- **ESI levels** follow the official ESI Implementation Handbook (5th edition)
- **Drug interactions** sourced from FDA drug interaction databases and clinical pharmacology
- **Sepsis criteria** follow Sepsis-3 (Singer et al., JAMA 2016) and Surviving Sepsis Campaign 2021
- **Antibiotic choices** follow IDSA guidelines
- **Graders penalize clinical safety errors** (undertriage, missed critical interactions, no vasopressors in shock)

> ⚠️ **Disclaimer**: This environment is for AI training/research purposes only. Not for actual clinical use.

---

## 🤝 Contributing

Contributions welcome! Ideas for expanding the environment:
- New clinical scenarios (MI, PE, DKA, stroke)
- Additional task types (radiology triage, ICU scoring)
- Multi-turn dialogue scenarios
- Multi-patient queue management

---

## 📄 License

MIT License. See LICENSE file.
