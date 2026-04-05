from __future__ import annotations

import uuid
import random
import math
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from llm_evaluator import (
    evaluate_with_llm, compute_hybrid_reward,
    get_oracle_action, LLMBackend, LLMEvalResult
)


# ──────────────────────────────────────────────────────────────────────────────
# ENUMS & CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

class Severity(str, Enum):
    CRITICAL = "critical"   # ESI 1 — life threat, deteriorates fast
    URGENT   = "urgent"     # ESI 2/3 — serious, moderate deterioration
    STABLE   = "stable"     # ESI 4/5 — can wait
    DISCHARGE = "discharge" # Already treated, leaving queue


class DifficultyMode(str, Enum):
    CALM   = "calm"    # 1-3 patients, ample resources
    BUSY   = "busy"    # 5-10 patients, moderate pressure
    SURGE  = "surge"   # 10-15 patients, limited resources
    CHAOS  = "chaos"   # 15-20 patients, mass casualty conditions


class TriageAction(str, Enum):
    ESI_1    = "esi_1"      # Resuscitation — immediate
    ESI_2    = "esi_2"      # Emergent — < 10 min
    ESI_3    = "esi_3"      # Urgent — < 30 min
    ESI_4    = "esi_4"      # Less Urgent — < 1 hr
    ESI_5    = "esi_5"      # Non-Urgent — < 2 hr
    DISCHARGE = "discharge"


# Resources per difficulty
RESOURCE_PROFILES = {
    DifficultyMode.CALM:  {"beds": 20, "doctors": 5, "nurses": 8,  "ventilators": 5,  "icu_beds": 4},
    DifficultyMode.BUSY:  {"beds": 15, "doctors": 3, "nurses": 6,  "ventilators": 3,  "icu_beds": 2},
    DifficultyMode.SURGE: {"beds": 8,  "doctors": 2, "nurses": 4,  "ventilators": 2,  "icu_beds": 1},
    DifficultyMode.CHAOS: {"beds": 3,  "doctors": 1, "nurses": 2,  "ventilators": 1,  "icu_beds": 0},
}

# Deterioration rates (vitals decay per simulation step)
DETERIORATION_RATES = {
    Severity.CRITICAL: {"hr_delta": +8, "sbp_delta": -6, "spo2_delta": -2, "gcs_delta": -1},
    Severity.URGENT:   {"hr_delta": +3, "sbp_delta": -2, "spo2_delta": -1, "gcs_delta": 0},
    Severity.STABLE:   {"hr_delta": +1, "sbp_delta": -1, "spo2_delta": 0,  "gcs_delta": 0},
    Severity.DISCHARGE: {"hr_delta": 0, "sbp_delta": 0,  "spo2_delta": 0,  "gcs_delta": 0},
}

# Arrival rates per difficulty (patients per step, Poisson lambda)
ARRIVAL_RATES = {
    DifficultyMode.CALM:  0.5,
    DifficultyMode.BUSY:  1.5,
    DifficultyMode.SURGE: 3.0,
    DifficultyMode.CHAOS: 5.0,
}


# ──────────────────────────────────────────────────────────────────────────────
# PATIENT DATA MODEL
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PatientVitals:
    heart_rate: int = 80
    systolic_bp: int = 120
    spo2: int = 98
    respiratory_rate: int = 16
    glasgow_coma_scale: int = 15
    temperature_f: float = 98.6

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def is_critical(self) -> bool:
        return (self.spo2 < 90 or self.systolic_bp < 80 or
                self.glasgow_coma_scale <= 8 or self.heart_rate > 140)

    @property
    def news2_score(self) -> int:
        """Compute NEWS-2 score from vitals."""
        score = 0
        rr = self.respiratory_rate
        if rr <= 8 or rr >= 25: score += 3
        elif rr >= 21: score += 2
        elif rr <= 11: score += 1
        spo2 = self.spo2
        if spo2 <= 91: score += 3
        elif spo2 <= 93: score += 2
        elif spo2 <= 95: score += 1
        sbp = self.systolic_bp
        if sbp <= 90 or sbp >= 220: score += 3
        elif sbp <= 100: score += 2
        elif sbp <= 110: score += 1
        hr = self.heart_rate
        if hr <= 40 or hr >= 131: score += 3
        elif hr >= 111 or hr <= 50: score += 2
        elif hr >= 91: score += 1
        tc = (self.temperature_f - 32) * 5 / 9
        if tc <= 35.0: score += 3
        elif tc >= 39.1: score += 2
        elif tc <= 36.0 or tc >= 38.1: score += 1
        gcs = self.glasgow_coma_scale
        if gcs <= 8: score += 3
        elif gcs <= 11: score += 2
        elif gcs <= 14: score += 1
        return score


@dataclass
class Patient:
    patient_id: str
    name: str
    age: int
    sex: str
    chief_complaint: str
    severity: Severity
    vitals: PatientVitals
    true_esi: int                           # Ground truth ESI level (1-5)
    risk_factors: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    current_medications: List[Dict] = field(default_factory=list)
    arrival_time: float = field(default_factory=time.time)
    wait_time_minutes: float = 0.0
    triage_action: Optional[str] = None     # What the agent assigned
    oracle_action: Optional[str] = None     # What the oracle would assign
    deterioration_events: List[str] = field(default_factory=list)
    is_triaged: bool = False
    outcome: Optional[str] = None           # "improved", "deteriorated", "stable"

    def deteriorate(self) -> Optional[str]:
        """Apply one tick of deterioration. Returns alert string if critical event."""
        if self.severity == Severity.DISCHARGE or self.is_triaged:
            return None

        rates = DETERIORATION_RATES[self.severity]
        noise = lambda: random.randint(-1, 1)

        self.vitals.heart_rate = min(200, self.vitals.heart_rate + rates["hr_delta"] + noise())
        self.vitals.systolic_bp = max(40, self.vitals.systolic_bp + rates["sbp_delta"] + noise())
        self.vitals.spo2 = max(60, min(100, self.vitals.spo2 + rates["spo2_delta"]))
        if rates["gcs_delta"]:
            self.vitals.glasgow_coma_scale = max(3, self.vitals.glasgow_coma_scale + rates["gcs_delta"])

        self.wait_time_minutes += 1.0

        # Check for critical deterioration event
        alert = None
        if self.vitals.spo2 < 90 and self.severity != Severity.CRITICAL:
            self.severity = Severity.CRITICAL
            alert = f"⚠️ CRITICAL UPGRADE: {self.patient_id} — SpO₂ dropped to {self.vitals.spo2}%"
            self.deterioration_events.append(alert)
        elif self.vitals.systolic_bp < 80 and self.severity != Severity.CRITICAL:
            self.severity = Severity.CRITICAL
            alert = f"⚠️ CRITICAL UPGRADE: {self.patient_id} — SBP dropped to {self.vitals.systolic_bp} mmHg"
            self.deterioration_events.append(alert)
        elif self.vitals.glasgow_coma_scale <= 8 and self.severity != Severity.CRITICAL:
            self.severity = Severity.CRITICAL
            alert = f"⚠️ CRITICAL UPGRADE: {self.patient_id} — GCS dropped to {self.vitals.glasgow_coma_scale}"
            self.deterioration_events.append(alert)

        return alert

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "name": self.name,
            "age": self.age,
            "sex": self.sex,
            "chief_complaint": self.chief_complaint,
            "severity": self.severity.value,
            "true_esi": self.true_esi,
            "vitals": self.vitals.to_dict(),
            "news2_score": self.vitals.news2_score,
            "wait_time_minutes": round(self.wait_time_minutes, 1),
            "risk_factors": self.risk_factors,
            "allergies": self.allergies,
            "current_medications": self.current_medications,
            "deterioration_events": self.deterioration_events,
            "is_triaged": self.is_triaged,
        }


# ──────────────────────────────────────────────────────────────────────────────
# PATIENT GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

PATIENT_TEMPLATES = [
    # Critical (ESI 1-2)
    {
        "chief_complaint": "Crushing chest pain radiating to jaw, diaphoresis, nausea",
        "severity": Severity.CRITICAL, "true_esi": 1,
        "vitals": {"heart_rate": 108, "systolic_bp": 88, "spo2": 93, "respiratory_rate": 22, "glasgow_coma_scale": 15},
        "risk_factors": ["Hypertension", "Diabetes", "Smoking"],
        "name_pool": ["James Mitchell", "Robert Chen", "Maria Santos"]
    },
    {
        "chief_complaint": "Sudden thunderclap headache, worst of life, neck stiffness",
        "severity": Severity.CRITICAL, "true_esi": 1,
        "vitals": {"heart_rate": 95, "systolic_bp": 155, "spo2": 96, "respiratory_rate": 18, "glasgow_coma_scale": 13},
        "risk_factors": [],
        "name_pool": ["Sarah Johnson", "Priya Patel", "Elena Rodriguez"]
    },
    {
        "chief_complaint": "Unresponsive, found collapsed, bystander CPR ongoing",
        "severity": Severity.CRITICAL, "true_esi": 1,
        "vitals": {"heart_rate": 140, "systolic_bp": 70, "spo2": 82, "respiratory_rate": 6, "glasgow_coma_scale": 3},
        "risk_factors": ["Cardiovascular Disease"],
        "name_pool": ["David Williams", "Thomas Kim", "Carlos Garcia"]
    },
    # Urgent (ESI 2-3)
    {
        "chief_complaint": "Fever 39.5°C, productive cough, right-sided pleuritic chest pain, 3 days",
        "severity": Severity.URGENT, "true_esi": 2,
        "vitals": {"heart_rate": 102, "systolic_bp": 110, "spo2": 92, "respiratory_rate": 24, "glasgow_coma_scale": 15},
        "risk_factors": ["Chronic Lung Disease"],
        "name_pool": ["Linda Park", "Susan Taylor", "Nancy White"]
    },
    {
        "chief_complaint": "Right facial droop, left arm weakness, slurred speech — onset 45 min ago",
        "severity": Severity.URGENT, "true_esi": 2,
        "vitals": {"heart_rate": 84, "systolic_bp": 172, "spo2": 97, "respiratory_rate": 17, "glasgow_coma_scale": 14},
        "risk_factors": ["Hypertension", "Atrial Fibrillation"],
        "name_pool": ["George Brown", "Harold Davis", "Steven Wilson"]
    },
    {
        "chief_complaint": "Sudden pleuritic chest pain, dyspnea, tachycardia — long haul flight 48h ago",
        "severity": Severity.URGENT, "true_esi": 2,
        "vitals": {"heart_rate": 118, "systolic_bp": 108, "spo2": 91, "respiratory_rate": 26, "glasgow_coma_scale": 15},
        "risk_factors": ["Recent Surgery"],
        "name_pool": ["Amanda Foster", "Michelle Lee", "Rachel Torres"]
    },
    # Stable (ESI 3-4)
    {
        "chief_complaint": "Fever 38°C, dysuria, right flank pain × 2 days",
        "severity": Severity.STABLE, "true_esi": 3,
        "vitals": {"heart_rate": 98, "systolic_bp": 118, "spo2": 98, "respiratory_rate": 18, "glasgow_coma_scale": 15},
        "risk_factors": [],
        "name_pool": ["Jessica Moore", "Ashley Jackson", "Brittany Harris"]
    },
    {
        "chief_complaint": "Right ankle sprain after sports injury, pain 6/10, weight-bearing",
        "severity": Severity.STABLE, "true_esi": 4,
        "vitals": {"heart_rate": 76, "systolic_bp": 126, "spo2": 99, "respiratory_rate": 15, "glasgow_coma_scale": 15},
        "risk_factors": [],
        "name_pool": ["Tyler Anderson", "Brandon Thompson", "Nathan Martinez"]
    },
    {
        "chief_complaint": "Mild sore throat × 3 days, no fever, no difficulty swallowing",
        "severity": Severity.STABLE, "true_esi": 5,
        "vitals": {"heart_rate": 72, "systolic_bp": 118, "spo2": 99, "respiratory_rate": 14, "glasgow_coma_scale": 15},
        "risk_factors": [],
        "name_pool": ["Emily Clark", "Hannah Lewis", "Madison Robinson"]
    },
    {
        "chief_complaint": "Polyuria, polydipsia, 8kg weight loss, fruity breath, abdominal pain",
        "severity": Severity.URGENT, "true_esi": 2,
        "vitals": {"heart_rate": 114, "systolic_bp": 96, "spo2": 98, "respiratory_rate": 28, "glasgow_coma_scale": 14},
        "risk_factors": ["Diabetes"],
        "name_pool": ["Michael Young", "Christopher Hall", "Andrew Allen"]
    },
]


def generate_patient(difficulty: DifficultyMode) -> Patient:
    """Generate a synthetic patient based on difficulty. Higher difficulty → more criticals."""
    if difficulty == DifficultyMode.CHAOS:
        weights = [4, 4, 4, 2, 2, 2, 1, 1, 0, 3]  # skew critical
    elif difficulty == DifficultyMode.SURGE:
        weights = [3, 3, 3, 2, 2, 2, 1, 1, 0, 2]
    elif difficulty == DifficultyMode.BUSY:
        weights = [2, 2, 2, 2, 2, 2, 2, 2, 1, 2]
    else:  # CALM
        weights = [1, 1, 1, 2, 2, 2, 3, 3, 2, 1]

    template = random.choices(PATIENT_TEMPLATES, weights=weights[:len(PATIENT_TEMPLATES)])[0]
    name = random.choice(template["name_pool"])
    vitals_data = {**template["vitals"]}

    # Add noise
    vitals_data["heart_rate"] += random.randint(-5, 5)
    vitals_data["systolic_bp"] += random.randint(-8, 8)
    vitals_data["spo2"] = max(70, min(100, vitals_data["spo2"] + random.randint(-2, 1)))

    vitals = PatientVitals(**vitals_data)

    return Patient(
        patient_id=f"PT-{uuid.uuid4().hex[:6].upper()}",
        name=name,
        age=random.randint(18, 85),
        sex=random.choice(["M", "F"]),
        chief_complaint=template["chief_complaint"],
        severity=template["severity"],
        vitals=vitals,
        true_esi=template["true_esi"],
        risk_factors=template["risk_factors"].copy(),
        allergies=random.choice([[], ["Penicillin"], ["Sulfa"], []]),
    )


# ──────────────────────────────────────────────────────────────────────────────
# TRAJECTORY & EPISODE LOGGING
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrajectoryStep:
    """Single step in an episode trajectory — critical for interpretability."""
    step_id: int
    patient_id: str
    state: Dict[str, Any]
    action: Dict[str, Any]
    reasoning: str
    rule_reward: float
    llm_feedback: LLMEvalResult
    final_reward: float
    reward_breakdown: Dict[str, Any]
    oracle_action: Dict[str, Any]
    mismatch_with_oracle: bool
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["llm_feedback"] = asdict(self.llm_feedback)
        return d


@dataclass
class EpisodeLog:
    """Full episode log — multi-step scoring and learning trends."""
    episode_id: str
    difficulty: str
    task_type: str
    steps: List[TrajectoryStep] = field(default_factory=list)
    total_rule_reward: float = 0.0
    total_llm_adjustment: float = 0.0
    total_final_reward: float = 0.0
    failure_cases: List[Dict] = field(default_factory=list)
    oracle_mismatches: int = 0
    patients_triaged: int = 0
    critical_errors: List[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    @property
    def duration_seconds(self) -> float:
        end = self.end_time or time.time()
        return round(end - self.start_time, 2)

    @property
    def avg_reward_per_step(self) -> float:
        if not self.steps:
            return 0.0
        return round(self.total_final_reward / len(self.steps), 4)

    @property
    def oracle_match_rate(self) -> float:
        if not self.steps:
            return 0.0
        matches = sum(1 for s in self.steps if not s.mismatch_with_oracle)
        return round(matches / len(self.steps), 3)

    def summary(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "difficulty": self.difficulty,
            "task_type": self.task_type,
            "total_steps": len(self.steps),
            "patients_triaged": self.patients_triaged,
            "total_rule_reward": round(self.total_rule_reward, 4),
            "total_llm_adjustment": round(self.total_llm_adjustment, 4),
            "total_final_reward": round(self.total_final_reward, 4),
            "avg_reward_per_step": self.avg_reward_per_step,
            "oracle_match_rate": self.oracle_match_rate,
            "oracle_mismatches": self.oracle_mismatches,
            "failure_cases": len(self.failure_cases),
            "critical_errors": len(self.critical_errors),
            "duration_seconds": self.duration_seconds,
        }


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ENVIRONMENT
# ──────────────────────────────────────────────────────────────────────────────

class ClinicalTriageEnvV2:
    """
    Production-grade RL environment for clinical triage.
    Key features:
      - Multi-patient queue (up to 20 simultaneous patients)
      - Real-time deterioration with stochastic noise
      - Resource constraints (beds, doctors, vents)
      - LLM-aligned hybrid reward shaping
      - Full trajectory logging
      - Oracle comparison ("What Would A Doctor Do?")
      - Curriculum difficulty: CALM → CHAOS
    Compatible with gym-style RL loops (PPO, DQN).
    Usage:
        env = ClinicalTriageEnvV2(difficulty=DifficultyMode.BUSY)
        state = env.reset()
        obs, reward, done, info = env.step(patient_id, action, reasoning)
    """

    MAX_QUEUE_SIZE = 20
    MAX_EPISODE_STEPS = 50
    LLM_ALPHA = 0.3  # LLM contribution weight

    def __init__(
        self,
        difficulty: DifficultyMode = DifficultyMode.CALM,
        llm_backend: LLMBackend = LLMBackend.RULE_BASED,
        task_type: str = "triage",
        enable_deterioration: bool = True,
        seed: Optional[int] = None
    ):
        self.difficulty = difficulty
        self.llm_backend = llm_backend
        self.task_type = task_type
        self.enable_deterioration = enable_deterioration

        if seed is not None:
            random.seed(seed)

        # Resources
        self.resources = dict(RESOURCE_PROFILES[difficulty])
        self.initial_resources = dict(self.resources)

        # Queue & state
        self.patient_queue: deque[Patient] = deque()
        self.triaged_patients: List[Patient] = []
        self.step_count: int = 0
        self.episode_id: str = str(uuid.uuid4())
        self.deterioration_alerts: List[str] = []

        # Episode logging
        self.current_episode = EpisodeLog(
            episode_id=self.episode_id,
            difficulty=difficulty.value,
            task_type=task_type
        )

        # Historical logs (across episodes)
        self.episode_history: List[EpisodeLog] = []
        self.reward_history: List[float] = []

    # ──────────────────────────────────────────────────────────────────────────
    # CORE API
    # ──────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        difficulty: Optional[DifficultyMode] = None
    ) -> Dict[str, Any]:
        """Reset environment for a new episode. Returns initial observation."""
        if self.current_episode.steps:
            self.current_episode.end_time = time.time()
            self.episode_history.append(self.current_episode)
            self.reward_history.append(self.current_episode.total_final_reward)

        if difficulty:
            self.difficulty = difficulty
            self.resources = dict(RESOURCE_PROFILES[difficulty])
            self.initial_resources = dict(self.resources)

        self.patient_queue = deque()
        self.triaged_patients = []
        self.step_count = 0
        self.episode_id = str(uuid.uuid4())
        self.deterioration_alerts = []

        self.current_episode = EpisodeLog(
            episode_id=self.episode_id,
            difficulty=self.difficulty.value,
            task_type=self.task_type
        )

        # Seed initial queue
        n_initial = {
            DifficultyMode.CALM:  2,
            DifficultyMode.BUSY:  5,
            DifficultyMode.SURGE: 10,
            DifficultyMode.CHAOS: 15,
        }[self.difficulty]

        for _ in range(n_initial):
            self.patient_queue.append(generate_patient(self.difficulty))

        return self._build_observation()

    def step(
        self,
        patient_id: str,
        action: Dict[str, Any],
        reasoning: str = ""
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute a triage decision on a specific patient.
        Args:
            patient_id: The ID of the patient being triaged.
            action:     Dict with keys like 'esi_level', 'bundle_items', etc.
            reasoning:  Free-text clinical rationale for the decision.
        Returns:
            (observation, final_reward, done, info)
        """
        self.step_count += 1

        # Find patient
        patient = self._find_patient(patient_id)
        if patient is None:
            return self._build_observation(), -0.5, False, {
                "error": f"Patient {patient_id} not found in queue"
            }

        # Compute state dict for LLM
        state_dict = {
            **patient.to_state_dict(),
            "task_type": self.task_type,
            "task_id": f"{self.task_type}_{self.difficulty.value}",
            "difficulty": self.difficulty.value,
            "resources": self.resources,
            "queue_size": len(self.patient_queue),
            "expected_action": {"esi_level": patient.true_esi},
        }

        # Compute rule-based reward
        rule_reward = self._compute_rule_reward(patient, action)

        # Compute LLM reward
        llm_result = evaluate_with_llm(
            state=state_dict,
            action=action,
            reasoning=reasoning,
            backend=self.llm_backend
        )

        # Hybrid reward
        final_reward, breakdown = compute_hybrid_reward(
            rule_reward=rule_reward,
            llm_result=llm_result,
            alpha=self.LLM_ALPHA
        )

        # Oracle comparison
        oracle = get_oracle_action(state_dict)
        patient.oracle_action = oracle
        patient.triage_action = str(action.get("esi_level", action))

        oracle_esi = oracle.get("esi_level", 3)
        agent_esi = action.get("esi_level", 3)
        mismatch = abs(oracle_esi - agent_esi) >= 2 if (oracle_esi and agent_esi) else False

        # Mark patient as triaged
        patient.is_triaged = True
        patient.severity = Severity.DISCHARGE
        self.patient_queue.remove(patient)
        self.triaged_patients.append(patient)
        self._update_resources(patient, action)

        # Log trajectory step
        traj_step = TrajectoryStep(
            step_id=self.step_count,
            patient_id=patient_id,
            state=state_dict,
            action=action,
            reasoning=reasoning,
            rule_reward=rule_reward,
            llm_feedback=llm_result,
            final_reward=final_reward,
            reward_breakdown=breakdown,
            oracle_action=oracle,
            mismatch_with_oracle=mismatch
        )
        self.current_episode.steps.append(traj_step)
        self.current_episode.total_rule_reward += rule_reward
        self.current_episode.total_llm_adjustment += llm_result.reward_adjustment
        self.current_episode.total_final_reward += final_reward
        self.current_episode.patients_triaged += 1

        if mismatch:
            self.current_episode.oracle_mismatches += 1

        if llm_result.safety_score <= 3:
            err = f"SAFETY ALERT step {self.step_count}: {patient_id} — {llm_result.explanation[:80]}"
            self.current_episode.critical_errors.append(err)

        if final_reward < -0.2:
            self.current_episode.failure_cases.append({
                "step": self.step_count,
                "patient_id": patient_id,
                "reward": final_reward,
                "explanation": llm_result.explanation,
                "oracle": oracle,
                "agent_action": action
            })

        # Stochastic patient arrival
        self._maybe_arrive_patients()

        # Deteriorate waiting patients
        if self.enable_deterioration:
            for p in self.patient_queue:
                alert = p.deteriorate()
                if alert:
                    self.deterioration_alerts.append(alert)

        done = (
            self.step_count >= self.MAX_EPISODE_STEPS or
            len(self.patient_queue) == 0
        )

        if done:
            self.current_episode.end_time = time.time()

        info = {
            "rule_reward": rule_reward,
            "llm_adjustment": llm_result.reward_adjustment,
            "final_reward": final_reward,
            "reward_breakdown": breakdown,
            "llm_scores": {
                "clinical": llm_result.clinical_score,
                "safety": llm_result.safety_score,
                "efficiency": llm_result.efficiency_score,
                "total": llm_result.total_score,
            },
            "llm_explanation": llm_result.explanation,
            "oracle_action": oracle,
            "mismatch_with_oracle": mismatch,
            "patient_triaged": patient_id,
            "queue_size": len(self.patient_queue),
            "resources": self.resources,
            "step_count": self.step_count,
            "episode_reward_so_far": self.current_episode.total_final_reward,
            "deterioration_alerts": list(self.deterioration_alerts[-3:]),
        }

        return self._build_observation(), final_reward, done, info

    def get_trajectory(self) -> List[Dict[str, Any]]:
        """Return full episode trajectory for interpretability analysis."""
        return [step.to_dict() for step in self.current_episode.steps]

    def get_episode_summary(self) -> Dict[str, Any]:
        """Return episode-level scoring summary."""
        return self.current_episode.summary()

    def get_failure_cases(self) -> List[Dict[str, Any]]:
        """Return all failure cases from current episode for failure mode analysis."""
        return self.current_episode.failure_cases

    def get_learning_trends(self) -> Dict[str, Any]:
        """Return reward trends across episodes for training visualization."""
        if not self.episode_history:
            return {"episodes": 0, "rewards": [], "trends": {}}

        rewards = [e.total_final_reward for e in self.episode_history]
        match_rates = [e.oracle_match_rate for e in self.episode_history]

        return {
            "episodes": len(self.episode_history),
            "rewards": rewards,
            "match_rates": match_rates,
            "avg_reward": round(sum(rewards) / len(rewards), 4),
            "best_episode": max(rewards),
            "worst_episode": min(rewards),
            "improvement_rate": round(
                (rewards[-1] - rewards[0]) / max(1, len(rewards) - 1), 4
            ) if len(rewards) > 1 else 0.0,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # OBSERVATION BUILDER
    # ──────────────────────────────────────────────────────────────────────────

    def _build_observation(self) -> Dict[str, Any]:
        queue_list = [p.to_state_dict() for p in self.patient_queue]
        queue_list.sort(key=lambda p: (p["true_esi"], -p["wait_time_minutes"]))

        critical_count = sum(1 for p in self.patient_queue if p["severity"] == "critical")
        urgent_count = sum(1 for p in self.patient_queue if p["severity"] == "urgent")
        stable_count = sum(1 for p in self.patient_queue if p["severity"] == "stable")

        return {
            "episode_id": self.episode_id,
            "difficulty": self.difficulty.value,
            "task_type": self.task_type,
            "step_count": self.step_count,
            "max_steps": self.MAX_EPISODE_STEPS,
            "patient_queue": queue_list,
            "queue_size": len(self.patient_queue),
            "critical_patients": critical_count,
            "urgent_patients": urgent_count,
            "stable_patients": stable_count,
            "resources": self.resources,
            "resource_utilization": self._compute_resource_utilization(),
            "deterioration_alerts": list(self.deterioration_alerts[-5:]),
            "episode_reward": self.current_episode.total_final_reward,
            "episode_steps": self.step_count,
            "triaged_count": len(self.triaged_patients),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # REWARD LOGIC
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_rule_reward(
        self,
        patient: Patient,
        action: Dict[str, Any]
    ) -> float:
        """
        Rule-based reward component.
        Rewards:
          - ESI exact match: +1.0
          - ESI off by 1: +0.4
          - Over-triage (agent ESI < true ESI): -0.2 per level
          - Under-triage (agent ESI > true ESI): -0.5 per level (ASYMMETRIC)
          - Critical patient untriaged while waiting: -0.3 per step
          - Wait time penalty: -0.01 per minute over threshold
        """
        agent_esi = int(action.get("esi_level", 3))
        true_esi = patient.true_esi
        delta = agent_esi - true_esi  # positive = undertriage, negative = overtriage

        if delta == 0:
            reward = 1.0
        elif delta == 1:
            reward = -0.2  # mild undertriage
        elif delta == -1:
            reward = -0.1  # mild overtriage
        elif delta >= 2:
            reward = -0.5 * delta  # serious undertriage (asymmetric penalty)
        else:
            reward = -0.2 * abs(delta)  # overtriage

        # Wait time penalty (under-triage doubly penalized if critical)
        wait_penalty = min(0.3, patient.wait_time_minutes * 0.01)
        if patient.severity == Severity.CRITICAL:
            wait_penalty *= 2.0

        # Resource safety check
        if agent_esi == 1 and self.resources.get("icu_beds", 0) <= 0:
            reward -= 0.1  # resource unavailability penalty

        difficulty_mult = {
            DifficultyMode.CALM: 0.8,
            DifficultyMode.BUSY: 1.0,
            DifficultyMode.SURGE: 1.2,
            DifficultyMode.CHAOS: 1.5,
        }[self.difficulty]

        raw = (reward - wait_penalty) * difficulty_mult
        return round(max(-2.0, min(1.5, raw)), 4)

    def _update_resources(self, patient: Patient, action: Dict[str, Any]) -> None:
        """Update resource pool after triaging a patient."""
        esi = int(action.get("esi_level", 3))
        if esi <= 1:
            self.resources["icu_beds"] = max(0, self.resources["icu_beds"] - 1)
            self.resources["ventilators"] = max(0, self.resources["ventilators"] - 1)
        if esi <= 3:
            self.resources["beds"] = max(0, self.resources["beds"] - 1)
            self.resources["doctors"] = max(0, self.resources["doctors"] - 1)

    def _compute_resource_utilization(self) -> Dict[str, float]:
        util = {}
        for k in ["beds", "doctors", "ventilators", "icu_beds"]:
            init = self.initial_resources.get(k, 1)
            curr = self.resources.get(k, 0)
            util[k] = round((init - curr) / max(1, init), 2)
        return util

    # ──────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _find_patient(self, patient_id: str) -> Optional[Patient]:
        for p in self.patient_queue:
            if p.patient_id == patient_id:
                return p
        return None

    def _maybe_arrive_patients(self) -> None:
        """Stochastic patient arrival via Poisson process."""
        lam = ARRIVAL_RATES[self.difficulty]
        n_arrivals = min(
            random.choices(range(8), weights=[
                math.exp(-lam) * (lam**k) / math.factorial(k)
                for k in range(8)
            ])[0],
            self.MAX_QUEUE_SIZE - len(self.patient_queue)
        )
        for _ in range(n_arrivals):
            self.patient_queue.append(generate_patient(self.difficulty))

    @property
    def action_space(self) -> List[str]:
        """Available triage actions."""
        return [a.value for a in TriageAction]

    @property
    def observation_space_keys(self) -> List[str]:
        """Keys in the observation dict."""
        return [
            "episode_id", "difficulty", "task_type", "step_count",
            "patient_queue", "queue_size", "resources", "episode_reward"
        ]

    @staticmethod
    def list_difficulties() -> List[str]:
        return [d.value for d in DifficultyMode]
