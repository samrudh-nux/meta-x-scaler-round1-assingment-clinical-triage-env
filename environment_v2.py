from __future__ import annotations

import random
import uuid
import time
import math
import json
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

class DifficultyMode(str, Enum):
    CALM  = "calm"    # 2-3 patients, ample resources
    BUSY  = "busy"    # 5-10 patients, moderate pressure
    SURGE = "surge"   # 10-15 patients, limited resources
    CHAOS = "chaos"   # 15-20 patients, MCI mode

class PatientAcuity(str, Enum):
    CRITICAL = "CRITICAL"   # ESI 1-2, deteriorates fast
    URGENT   = "URGENT"     # ESI 2-3, moderate decay
    STABLE   = "STABLE"     # ESI 4-5, minimal change

DIFFICULTY_CONFIG = {
    DifficultyMode.CALM:  {"n_patients": (2, 3),  "arrival_rate": 0.3, "critical_frac": 0.15, "resources": 10},
    DifficultyMode.BUSY:  {"n_patients": (5, 8),  "arrival_rate": 0.6, "critical_frac": 0.25, "resources": 6},
    DifficultyMode.SURGE: {"n_patients": (10,14), "arrival_rate": 1.2, "critical_frac": 0.35, "resources": 3},
    DifficultyMode.CHAOS: {"n_patients": (15,20), "arrival_rate": 2.0, "critical_frac": 0.45, "resources": 1},
}

DETERIORATION_RATES = {
    PatientAcuity.CRITICAL: {"hr": +8,  "sbp": -6,  "spo2": -2, "gcs": -1},
    PatientAcuity.URGENT:   {"hr": +3,  "sbp": -2,  "spo2": -1, "gcs":  0},
    PatientAcuity.STABLE:   {"hr": +1,  "sbp": -1,  "spo2":  0, "gcs":  0},
}

# Rich patient templates per acuity (chief complaint, vitals baseline, true ESI)
PATIENT_TEMPLATES = {
    PatientAcuity.CRITICAL: [
        {
            "cc": "Crushing chest pain, diaphoresis, radiation to left arm",
            "vitals": {"hr": 118, "sbp": 88, "spo2": 91, "rr": 24, "gcs": 14, "temp_f": 98.4},
            "true_esi": 1, "dx": "STEMI",
            "tags": ["Cardiovascular", "Time-Critical", "Thrombolytics"],
            "risk_factors": ["Hypertension", "Diabetes", "Smoking"],
            "allergies": [],
            "interventions": ["12-lead ECG STAT", "Aspirin 325mg", "Cath lab activation", "IV access ×2"],
        },
        {
            "cc": "Sudden onset worst headache of life, neck stiffness, photophobia",
            "vitals": {"hr": 102, "sbp": 178, "spo2": 96, "rr": 18, "gcs": 13, "temp_f": 101.2},
            "true_esi": 1, "dx": "Subarachnoid Haemorrhage",
            "tags": ["Neurological", "SAH", "Time-Critical"],
            "risk_factors": ["Hypertension", "Family History of Aneurysm"],
            "allergies": ["Penicillin"],
            "interventions": ["CT Head STAT", "Neurosurgery consult", "Airway assessment", "BP control"],
        },
        {
            "cc": "Unresponsive, witnessed seizure, GCS 7",
            "vitals": {"hr": 132, "sbp": 82, "spo2": 87, "rr": 28, "gcs": 7, "temp_f": 103.1},
            "true_esi": 1, "dx": "Status Epilepticus",
            "tags": ["Neurological", "Airway Risk", "Seizure"],
            "risk_factors": ["Epilepsy", "Non-compliant with medications"],
            "allergies": [],
            "interventions": ["Airway management", "Lorazepam IV", "Glucose check", "O2 15L NRB"],
        },
        {
            "cc": "Anaphylaxis — bee sting, throat swelling, urticaria, stridor",
            "vitals": {"hr": 138, "sbp": 72, "spo2": 88, "rr": 32, "gcs": 14, "temp_f": 99.1},
            "true_esi": 1, "dx": "Anaphylactic Shock",
            "tags": ["Allergy", "Airway Emergency", "Shock"],
            "risk_factors": ["Known bee venom allergy"],
            "allergies": ["Bee venom"],
            "interventions": ["Epinephrine 0.5mg IM", "Airway prep", "IV access", "Fluid bolus"],
        },
    ],
    PatientAcuity.URGENT: [
        {
            "cc": "Shortness of breath at rest, wheeze, known asthma",
            "vitals": {"hr": 108, "sbp": 128, "spo2": 93, "rr": 22, "gcs": 15, "temp_f": 98.8},
            "true_esi": 2, "dx": "Acute Severe Asthma",
            "tags": ["Respiratory", "Bronchospasm"],
            "risk_factors": ["Asthma", "Recent URTI"],
            "allergies": ["Aspirin"],
            "interventions": ["Nebulised salbutamol", "O2 titration", "PEFR measurement", "IV access"],
        },
        {
            "cc": "High fever, productive cough, pleuritic chest pain, rigors",
            "vitals": {"hr": 112, "sbp": 104, "spo2": 92, "rr": 26, "gcs": 15, "temp_f": 102.8},
            "true_esi": 2, "dx": "Community-Acquired Pneumonia with Sepsis",
            "tags": ["Respiratory", "Sepsis", "Pneumonia"],
            "risk_factors": ["COPD", "Immunocompromised", "Smoking"],
            "allergies": [],
            "interventions": ["Blood cultures", "CXR STAT", "Broad-spectrum antibiotics", "IV fluids"],
        },
        {
            "cc": "Sudden painless visual loss right eye, no light perception",
            "vitals": {"hr": 78, "sbp": 162, "spo2": 98, "rr": 16, "gcs": 15, "temp_f": 98.6},
            "true_esi": 2, "dx": "Central Retinal Artery Occlusion",
            "tags": ["Ophthalmology", "Stroke Equivalent", "Time-Critical"],
            "risk_factors": ["Atrial Fibrillation", "Hypertension"],
            "allergies": ["Sulfa"],
            "interventions": ["Ophthalmology emergency consult", "IOP check", "Stroke workup", "ECG"],
        },
        {
            "cc": "Severe abdominal pain, rebound tenderness, rigidity, fever",
            "vitals": {"hr": 116, "sbp": 98, "spo2": 97, "rr": 20, "gcs": 15, "temp_f": 101.9},
            "true_esi": 2, "dx": "Perforated Peptic Ulcer",
            "tags": ["Surgical", "Peritonitis", "Sepsis"],
            "risk_factors": ["NSAID use", "H. pylori"],
            "allergies": [],
            "interventions": ["Surgery consult STAT", "CT abdomen", "Blood cultures", "NPO"],
        },
    ],
    PatientAcuity.STABLE: [
        {
            "cc": "Right wrist pain after fall, mild swelling, no neurovascular deficit",
            "vitals": {"hr": 78, "sbp": 124, "spo2": 99, "rr": 16, "gcs": 15, "temp_f": 98.6},
            "true_esi": 4, "dx": "Distal Radius Fracture",
            "tags": ["Orthopaedic", "Trauma", "Minor"],
            "risk_factors": ["Osteoporosis"],
            "allergies": ["Codeine"],
            "interventions": ["X-ray wrist AP/Lateral", "Analgesia", "Splint"],
        },
        {
            "cc": "Sore throat, fever 3 days, no stridor, no drooling",
            "vitals": {"hr": 88, "sbp": 118, "spo2": 99, "rr": 16, "gcs": 15, "temp_f": 100.8},
            "true_esi": 5, "dx": "Viral Pharyngitis",
            "tags": ["ENT", "Viral", "Minor"],
            "risk_factors": [],
            "allergies": [],
            "interventions": ["Throat swab", "Analgesia", "Discharge with advice"],
        },
        {
            "cc": "Urinary frequency, burning, dysuria, no systemic features",
            "vitals": {"hr": 72, "sbp": 118, "spo2": 99, "rr": 14, "gcs": 15, "temp_f": 98.9},
            "true_esi": 4, "dx": "Uncomplicated UTI",
            "tags": ["Urology", "Infection", "Minor"],
            "risk_factors": ["Recurrent UTIs"],
            "allergies": [],
            "interventions": ["MSU culture", "Nitrofurantoin", "Fluid advice"],
        },
        {
            "cc": "Tension headache, gradual onset, no focal neuro, no neck stiffness",
            "vitals": {"hr": 76, "sbp": 122, "spo2": 99, "rr": 15, "gcs": 15, "temp_f": 98.5},
            "true_esi": 5, "dx": "Tension-Type Headache",
            "tags": ["Neurology", "Minor", "Primary Headache"],
            "risk_factors": ["Stress", "Poor posture"],
            "allergies": [],
            "interventions": ["Analgesia", "Discharge with safety netting"],
        },
    ],
}

NAMES_POOL = [
    ("James", "M"), ("Emma", "F"), ("Oliver", "M"), ("Sophia", "F"),
    ("Liam", "M"), ("Ava", "F"), ("Noah", "M"), ("Isabella", "F"),
    ("William", "M"), ("Mia", "F"), ("Ethan", "M"), ("Charlotte", "F"),
    ("Muhammad", "M"), ("Aisha", "F"), ("Ravi", "M"), ("Priya", "F"),
    ("Carlos", "M"), ("Sofia", "F"), ("Ahmed", "M"), ("Fatima", "F"),
]

@dataclass
class PatientState:
    patient_id: str
    name: str
    age: int
    sex: str
    chief_complaint: str
    acuity: PatientAcuity
    true_esi: int
    dx: str
    vitals: Dict[str, Any]
    vitals_initial: Dict[str, Any]
    tags: List[str]
    risk_factors: List[str]
    allergies: List[str]
    recommended_interventions: List[str]
    arrival_time: float
    triage_time: Optional[float] = None
    steps_waited: int = 0
    triaged: bool = False
    deteriorated: bool = False
    outcome: str = "pending"  # pending | discharged | admitted | deceased

    def news2_score(self) -> int:
        """Compute NEWS-2 score from current vitals."""
        v = self.vitals
        score = 0
        rr   = float(v.get("rr", 16))
        spo2 = float(v.get("spo2", 98))
        sbp  = float(v.get("sbp", 120))
        hr   = float(v.get("hr", 72))
        tf   = float(v.get("temp_f", 98.6))
        gcs  = int(v.get("gcs", 15))
        tc   = (tf - 32) * 5 / 9

        if rr <= 8 or rr >= 25: score += 3
        elif rr >= 21: score += 2
        elif rr <= 11: score += 1

        if spo2 <= 91: score += 3
        elif spo2 <= 93: score += 2
        elif spo2 <= 95: score += 1

        if sbp <= 90 or sbp >= 220: score += 3
        elif sbp <= 100: score += 2
        elif sbp <= 110: score += 1

        if hr <= 40 or hr >= 131: score += 3
        elif hr >= 111 or hr <= 50: score += 2
        elif hr >= 91: score += 1

        if tc <= 35.0: score += 3
        elif tc >= 39.1: score += 2
        elif tc <= 36.0 or tc >= 38.1: score += 1

        if gcs <= 8: score += 3
        elif gcs <= 11: score += 2
        elif gcs <= 14: score += 1

        return score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "name": self.name,
            "age": self.age,
            "sex": self.sex,
            "chief_complaint": self.chief_complaint,
            "acuity": self.acuity.value,
            "true_esi": self.true_esi,
            "diagnosis": self.dx,
            "vitals": self.vitals,
            "tags": self.tags,
            "risk_factors": self.risk_factors,
            "allergies": self.allergies,
            "recommended_interventions": self.recommended_interventions,
            "steps_waited": self.steps_waited,
            "triaged": self.triaged,
            "deteriorated": self.deteriorated,
            "outcome": self.outcome,
            "news2_score": self.news2_score(),
            "wait_seconds": round(time.time() - self.arrival_time, 1),
        }


@dataclass
class StepRecord:
    step_number: int
    patient_id: str
    patient_name: str
    action: Dict[str, Any]
    reasoning: str
    rule_reward: float
    llm_adjustment: float
    final_reward: float
    llm_scores: Dict[str, Any]
    llm_explanation: str
    oracle_action: Dict[str, Any]
    mismatch_with_oracle: bool
    vitals_at_decision: Dict[str, Any]
    true_esi: int
    agent_esi: int
    is_undertriage: bool
    is_overtriage: bool
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



def _oracle_triage(patient: PatientState) -> Dict[str, Any]:
    """Ideal physician action based on ESI guidelines + vital sign thresholds."""
    v = patient.vitals
    spo2 = float(v.get("spo2", 98))
    sbp  = float(v.get("sbp", 120))
    hr   = float(v.get("hr", 72))
    gcs  = int(v.get("gcs", 15))
    rr   = float(v.get("rr", 16))
    news2 = patient.news2_score()

    # Immediate life threats → ESI 1
    if gcs <= 8 or spo2 < 88 or sbp < 80:
        esi = 1
        rationale = f"ESI-1: Immediate life threat — GCS={gcs}, SpO₂={spo2}%, SBP={sbp}. Resuscitation bay."
    elif news2 >= 7 or spo2 < 92 or sbp < 95 or hr > 130:
        esi = 2
        rationale = f"ESI-2: High-risk, NEWS-2={news2}. Emergent assessment within 10 minutes."
    elif news2 >= 4 or hr > 100 or rr > 20:
        esi = 3
        rationale = f"ESI-2→3: Urgent presentation, NEWS-2={news2}. Requires ≥2 resources."
    elif news2 >= 1:
        esi = 3
        rationale = "ESI-3: Stable vitals but complex presentation."
    else:
        esi = 4 if patient.acuity == PatientAcuity.STABLE else 3
        rationale = f"ESI-{esi}: Low acuity. Routine triage pathway."

    # Override with true clinical picture
    esi = min(esi, patient.true_esi + 1)  # Never more than 1 level off oracle
    esi = max(esi, max(1, patient.true_esi - 1))

    return {
        "esi_level": esi,
        "rationale": rationale,
        "interventions": patient.recommended_interventions[:3],
        "disposition": ["Resuscitation Bay", "High Acuity", "Standard Bay", "Fast Track", "Discharge"][esi - 1],
        "source": "oracle_physician",
    }



def _grade_action(patient: PatientState, action: Dict[str, Any], reasoning: str) -> Tuple[float, Dict]:
    """
    Compute rule-based reward for a triage action.
    Returns (reward: float, component_scores: dict)
    """
    agent_esi = int(action.get("esi_level", action.get("triage_level", 3)))
    true_esi  = patient.true_esi
    delta     = abs(agent_esi - true_esi)
    news2     = patient.news2_score()

    # ── ESI accuracy ──────────────────────────────────────────────────────────
    if delta == 0:
        esi_score = 1.0
    elif delta == 1:
        esi_score = 0.6
    else:
        esi_score = max(0.0, 1.0 - delta * 0.35)

    # ── Under-triage penalty (critical miss) ──────────────────────────────────
    undertriage = agent_esi > true_esi  # assigned LOWER priority than needed
    undertriage_penalty = 0.0
    if undertriage and true_esi <= 2:
        # Missed critical patient — severe penalty
        undertriage_penalty = -0.5 * delta
    elif undertriage:
        undertriage_penalty = -0.2 * delta

    # ── Wait time efficiency ──────────────────────────────────────────────────
    wait = patient.steps_waited
    if wait == 0:
        wait_score = 1.0  # Immediate
    elif wait == 1:
        wait_score = 0.8 if true_esi <= 2 else 0.95
    else:
        # Each extra step costs more for higher acuity
        decay = 0.15 * (3 - true_esi)  # critical decays faster
        wait_score = max(0.0, 1.0 - wait * decay)

    # ── Reasoning quality ──────────────────────────────────────────────────────
    words = len(reasoning.split()) if reasoning else 0
    reasoning_score = min(1.0, words / 25)  # Target ~25 words

    # ── Vital sign acknowledgment ──────────────────────────────────────────────
    vitals_mentioned = 0
    r_lower = reasoning.lower()
    for kw in ["spo2", "oxygen", "blood pressure", "sbp", "heart rate", "hr", "gcs", "news", "rr", "respiratory"]:
        if kw in r_lower:
            vitals_mentioned += 1
    vitals_score = min(1.0, vitals_mentioned / 3)

    # ── Intervention appropriateness ──────────────────────────────────────────
    agent_interventions = action.get("interventions", action.get("recommended_immediate_interventions", []))
    correct_interventions = patient.recommended_interventions
    if agent_interventions and correct_interventions:
        overlap = sum(
            1 for ai in agent_interventions
            if any(ci.lower()[:10] in ai.lower() or ai.lower()[:10] in ci.lower()
                   for ci in correct_interventions)
        )
        intervention_score = min(1.0, overlap / max(1, len(correct_interventions[:3])))
    else:
        intervention_score = 0.3

    # ── Final weighted reward ──────────────────────────────────────────────────
    base_reward = (
        esi_score        * 0.45 +
        wait_score       * 0.25 +
        reasoning_score  * 0.15 +
        vitals_score     * 0.10 +
        intervention_score * 0.05
    ) + undertriage_penalty

    rule_reward = round(max(-1.0, min(1.5, base_reward)), 4)

    return rule_reward, {
        "esi_accuracy": round(esi_score, 3),
        "wait_penalty": round(wait_score, 3),
        "reasoning_quality": round(reasoning_score, 3),
        "vitals_acknowledgment": round(vitals_score, 3),
        "intervention_match": round(intervention_score, 3),
        "undertriage_penalty": round(undertriage_penalty, 3),
        "agent_esi": agent_esi,
        "true_esi": true_esi,
        "delta": delta,
        "is_undertriage": undertriage,
        "is_overtriage": agent_esi < true_esi,
    }


# 
class ClinicalTriageEnvV2:
    """
    Multi-patient queue RL environment for ClinicalTriageEnv v5.
    Compatible with OpenEnv spec. Supports:
      - Poisson patient arrivals with acuity weighting
      - Per-step physiological deterioration
      - LLM-aligned hybrid reward (optional)
      - Curriculum difficulty scheduling
      - Full episode logging & analytics
    """

    def __init__(
        self,
        difficulty: DifficultyMode = DifficultyMode.BUSY,
        llm_backend=None,   # LLMBackend enum or None
        task_type: str = "triage",
        enable_deterioration: bool = True,
        curriculum: bool = False,
        seed: Optional[int] = None,
    ):
        self.difficulty = difficulty
        self.llm_backend = llm_backend
        self.task_type = task_type
        self.enable_deterioration = enable_deterioration
        self.curriculum = curriculum

        if seed is not None:
            random.seed(seed)

        self._cfg = DIFFICULTY_CONFIG[difficulty]
        self._step_count = 0
        self._episode_count = 0
        self._trajectory: List[StepRecord] = []
        self._failure_cases: List[Dict] = []
        self._cumulative_reward = 0.0
        self._patients: Dict[str, PatientState] = {}
        self._triaged_ids: List[str] = []
        self._episode_start = time.time()

        # Curriculum state
        self._curriculum_scores: List[float] = []

        # Lazy-import LLM evaluator to avoid circular deps
        self._llm_evaluate = None
        self._compute_hybrid = None
        if llm_backend is not None:
            try:
                from llm_evaluator import evaluate_with_llm, compute_hybrid_reward
                self._llm_evaluate = evaluate_with_llm
                self._compute_hybrid = compute_hybrid_reward
            except ImportError:
                pass

        # Action space descriptor
        self.action_space = {
            "type": "dict",
            "fields": {
                "esi_level": {"type": "int", "range": [1, 5], "description": "ESI triage level"},
                "rationale": {"type": "str", "description": "Clinical reasoning"},
                "interventions": {"type": "list[str]", "description": "Immediate interventions"},
                "disposition": {"type": "str", "description": "Patient disposition"},
            }
        }


    def reset(self) -> Dict[str, Any]:
        """Start a new episode. Returns initial observation."""
        self._step_count = 0
        self._trajectory = []
        self._failure_cases = []
        self._cumulative_reward = 0.0
        self._patients = {}
        self._triaged_ids = []
        self._episode_start = time.time()

        # Apply curriculum ramp if enabled
        if self.curriculum and len(self._curriculum_scores) >= 5:
            recent_avg = sum(self._curriculum_scores[-5:]) / 5
            if recent_avg > 0.75 and self.difficulty != DifficultyMode.CHAOS:
                _next = {
                    DifficultyMode.CALM:  DifficultyMode.BUSY,
                    DifficultyMode.BUSY:  DifficultyMode.SURGE,
                    DifficultyMode.SURGE: DifficultyMode.CHAOS,
                }.get(self.difficulty, self.difficulty)
                self.difficulty = _next
                self._cfg = DIFFICULTY_CONFIG[_next]

        # Spawn initial patient cohort
        n_patients = random.randint(*self._cfg["n_patients"])
        for _ in range(n_patients):
            p = self._spawn_patient()
            self._patients[p.patient_id] = p

        self._episode_count += 1
        return self._make_obs()

    def step(
        self,
        patient_id: str,
        action: Dict[str, Any],
        reasoning: str = "",
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Triage one patient from the queue.
        Args:
            patient_id: Which patient to triage.
            action:     Triage action dict (esi_level, rationale, interventions, disposition).
            reasoning:  Free-text clinical reasoning from agent.
        Returns:
            (observation, reward, done, info)
        """
        self._step_count += 1

        # Deteriorate all waiting patients
        if self.enable_deterioration:
            self._deteriorate_all()

        # Validate patient
        if patient_id not in self._patients:
            return self._make_obs(), -0.5, False, {
                "error": f"Patient '{patient_id}' not in queue",
                "rule_reward": -0.5,
                "llm_adjustment": 0.0,
            }

        patient = self._patients[patient_id]

        # ── Grade rule-based ──────────────────────────────────────────────────
        rule_reward, component_scores = _grade_action(patient, action, reasoning)

        # ── Oracle for comparison ─────────────────────────────────────────────
        oracle = _oracle_triage(patient)
        agent_esi = int(action.get("esi_level", 3))
        mismatch  = abs(agent_esi - oracle["esi_level"]) > 1

        # ── LLM reward shaping ────────────────────────────────────────────────
        llm_scores: Dict[str, Any] = {}
        llm_explanation = ""
        llm_adjustment = 0.0
        final_reward = rule_reward
        backend_used = "rule_based"

        if self._llm_evaluate is not None and self._compute_hybrid is not None:
            try:
                state_for_llm = {
                    "task_type": self.task_type,
                    "difficulty": self.difficulty.value,
                    "patient": patient.to_dict(),
                    "expected_action": {"esi_level": patient.true_esi},
                }
                llm_result = self._llm_evaluate(
                    state=state_for_llm,
                    action=action,
                    reasoning=reasoning,
                    backend=self.llm_backend,
                )
                final_reward, reward_breakdown = self._compute_hybrid(rule_reward, llm_result, alpha=0.3)
                llm_adjustment = llm_result.reward_adjustment
                llm_scores = {
                    "clinical":   llm_result.clinical_score,
                    "safety":     llm_result.safety_score,
                    "efficiency": llm_result.efficiency_score,
                    "ethics":     llm_result.ethics_score,
                    "reasoning":  llm_result.reasoning_score,
                    "total":      llm_result.total_score,
                    "confidence": llm_result.confidence,
                }
                llm_explanation = llm_result.explanation
                backend_used = llm_result.backend_used
            except Exception as ex:
                llm_explanation = f"LLM eval unavailable: {ex}"

        # ── Update patient state ──────────────────────────────────────────────
        patient.triaged = True
        patient.triage_time = time.time()
        patient.outcome = "admitted" if agent_esi <= 2 else "discharged"
        self._triaged_ids.append(patient_id)

        # ── Log trajectory ────────────────────────────────────────────────────
        record = StepRecord(
            step_number=self._step_count,
            patient_id=patient_id,
            patient_name=patient.name,
            action=action,
            reasoning=reasoning,
            rule_reward=rule_reward,
            llm_adjustment=llm_adjustment,
            final_reward=final_reward,
            llm_scores=llm_scores,
            llm_explanation=llm_explanation,
            oracle_action=oracle,
            mismatch_with_oracle=mismatch,
            vitals_at_decision=deepcopy(patient.vitals),
            true_esi=patient.true_esi,
            agent_esi=agent_esi,
            is_undertriage=component_scores["is_undertriage"],
            is_overtriage=component_scores["is_overtriage"],
        )
        self._trajectory.append(record)

        # ── Log failures ──────────────────────────────────────────────────────
        if component_scores["is_undertriage"] and patient.true_esi <= 2:
            self._failure_cases.append({
                "type": "CRITICAL_UNDERTRIAGE",
                "severity": "HIGH",
                "patient_id": patient_id,
                "patient_name": patient.name,
                "true_esi": patient.true_esi,
                "agent_esi": agent_esi,
                "dx": patient.dx,
                "explanation": f"Critical patient assigned ESI-{agent_esi} (correct: ESI-{patient.true_esi}). Potential mortality.",
                "step": self._step_count,
            })

        self._cumulative_reward += final_reward

        # ── Remove from active queue ──────────────────────────────────────────
        del self._patients[patient_id]

        # ── Maybe add new arrivals (surge dynamics) ───────────────────────────
        if random.random() < self._cfg["arrival_rate"] * 0.4:
            new_p = self._spawn_patient()
            self._patients[new_p.patient_id] = new_p

        # ── Check episode end ─────────────────────────────────────────────────
        done = len(self._patients) == 0

        if done:
            self._curriculum_scores.append(
                self._cumulative_reward / max(1, len(self._trajectory))
            )

        obs = self._make_obs()

        info = {
            "rule_reward": rule_reward,
            "llm_adjustment": llm_adjustment,
            "final_reward": final_reward,
            "llm_scores": llm_scores,
            "llm_explanation": llm_explanation,
            "backend_used": backend_used,
            "oracle_action": oracle,
            "mismatch_with_oracle": mismatch,
            "component_scores": component_scores,
            "reward_breakdown": {
                "rule_reward": rule_reward,
                "llm_adjustment": llm_adjustment,
                "llm_contribution": round(llm_adjustment * 0.3, 4),
                "final_reward": final_reward,
                "formula": "final_reward = rule_reward + 0.3 × llm_adjustment",
            },
            "patients_remaining": len(self._patients),
            "total_triaged": len(self._triaged_ids),
            "episode_reward_so_far": round(self._cumulative_reward, 4),
            "failure_count": len(self._failure_cases),
        }

        return obs, final_reward, done, info

    def get_trajectory(self) -> List[Dict]:
        return [r.to_dict() for r in self._trajectory]

    def get_episode_summary(self) -> Dict[str, Any]:
        n = len(self._trajectory)
        if n == 0:
            return {"steps": 0, "mean_reward": 0.0}

        mean_r = self._cumulative_reward / n
        undertriage_rate = sum(1 for r in self._trajectory if r.is_undertriage) / n
        esi_accuracy = sum(1 for r in self._trajectory if r.agent_esi == r.true_esi) / n
        llm_mean = (
            sum(r.llm_scores.get("total", 0) for r in self._trajectory) / n
            if any(r.llm_scores for r in self._trajectory) else None
        )

        return {
            "difficulty": self.difficulty.value,
            "steps": n,
            "total_reward": round(self._cumulative_reward, 4),
            "mean_reward": round(mean_r, 4),
            "esi_exact_accuracy": round(esi_accuracy, 3),
            "undertriage_rate": round(undertriage_rate, 3),
            "critical_failures": len(self._failure_cases),
            "llm_mean_score": round(llm_mean, 2) if llm_mean else None,
            "episode_duration_s": round(time.time() - self._episode_start, 1),
            "performance_grade": self._grade_episode(mean_r),
        }

    def get_failure_cases(self) -> List[Dict]:
        return self._failure_cases

    def get_learning_trends(self) -> Dict[str, Any]:
        scores = self._curriculum_scores
        n = len(scores)
        if n < 2:
            return {"trend": "insufficient_data", "scores": scores, "episodes": n}

        recent = scores[-min(10, n):]
        direction = "improving" if recent[-1] > recent[0] else "declining" if recent[-1] < recent[0] else "stable"
        return {
            "trend": direction,
            "scores": scores[-50:],
            "recent_mean": round(sum(recent) / len(recent), 3),
            "overall_mean": round(sum(scores) / n, 3),
            "episodes": n,
            "current_difficulty": self.difficulty.value,
        }


    def _spawn_patient(self) -> PatientState:
        """Generate a realistic patient based on difficulty acuity distribution."""
        critical_frac = self._cfg["critical_frac"]
        r = random.random()
        if r < critical_frac:
            acuity = PatientAcuity.CRITICAL
        elif r < critical_frac + 0.45:
            acuity = PatientAcuity.URGENT
        else:
            acuity = PatientAcuity.STABLE

        template = random.choice(PATIENT_TEMPLATES[acuity])
        name_pair = random.choice(NAMES_POOL)
        age = random.randint(18, 88)

        # Add realistic noise to vitals
        vitals = {}
        for k, v in template["vitals"].items():
            noise = random.gauss(0, max(1, v * 0.04))
            vitals[k] = round(v + noise, 1)
        vitals = {k: max(1, v) for k, v in vitals.items()}

        patient_id = f"PT-{uuid.uuid4().hex[:6].upper()}"

        return PatientState(
            patient_id=patient_id,
            name=f"{name_pair[0]} {random.choice(['Smith','Jones','Patel','Williams','Brown','Garcia'])}",
            age=age,
            sex=name_pair[1],
            chief_complaint=template["cc"],
            acuity=acuity,
            true_esi=template["true_esi"],
            dx=template["dx"],
            vitals=vitals,
            vitals_initial=deepcopy(vitals),
            tags=template["tags"],
            risk_factors=template["risk_factors"],
            allergies=template["allergies"],
            recommended_interventions=template["interventions"],
            arrival_time=time.time(),
        )

    def _deteriorate_all(self):
        """Apply per-step vital sign deterioration to all untriaged patients."""
        for pid, patient in self._patients.items():
            if patient.triaged:
                continue
            rates = DETERIORATION_RATES[patient.acuity]
            v = patient.vitals

            v["hr"]   = round(min(200, v.get("hr", 80)   + rates["hr"]   + random.gauss(0, 1.5)), 1)
            v["sbp"]  = round(max(40,  v.get("sbp", 120)  + rates["sbp"]  + random.gauss(0, 2.0)), 1)
            v["spo2"] = round(max(60,  v.get("spo2", 98)  + rates["spo2"] + random.gauss(0, 0.5)), 1)
            v["gcs"]  = max(3, v.get("gcs", 15) + rates["gcs"])
            v["rr"]   = round(min(60, v.get("rr", 16) + random.gauss(0.5, 0.5)), 1)

            patient.steps_waited += 1

            # Upgrade acuity if vitals cross critical thresholds
            if (v["spo2"] < 90 or v["sbp"] < 80) and patient.acuity == PatientAcuity.URGENT:
                patient.acuity = PatientAcuity.CRITICAL
                patient.deteriorated = True
                patient.true_esi = min(patient.true_esi, 2)

    def _make_obs(self) -> Dict[str, Any]:
        """Build observation dict from current queue state."""
        queue = sorted(
            [p.to_dict() for p in self._patients.values()],
            key=lambda x: (x["true_esi"], -x["steps_waited"])  # most urgent first
        )
        critical_count = sum(1 for p in self._patients.values() if p.acuity == PatientAcuity.CRITICAL)
        urgent_count   = sum(1 for p in self._patients.values() if p.acuity == PatientAcuity.URGENT)
        stable_count   = sum(1 for p in self._patients.values() if p.acuity == PatientAcuity.STABLE)

        return {
            "patient_queue": queue,
            "queue_size": len(queue),
            "triaged_count": len(self._triaged_ids),
            "step": self._step_count,
            "difficulty": self.difficulty.value,
            "resources_remaining": max(0, self._cfg["resources"] - len(self._triaged_ids)),
            "acuity_breakdown": {
                "critical": critical_count,
                "urgent": urgent_count,
                "stable": stable_count,
            },
            "cumulative_reward": round(self._cumulative_reward, 4),
            "failure_count": len(self._failure_cases),
            "episode": self._episode_count,
        }

    @staticmethod
    def _grade_episode(mean_reward: float) -> str:
        if mean_reward >= 0.9:  return "S — Expert"
        if mean_reward >= 0.75: return "A — Proficient"
        if mean_reward >= 0.60: return "B — Competent"
        if mean_reward >= 0.45: return "C — Developing"
        return "F — Critical Review Required"
