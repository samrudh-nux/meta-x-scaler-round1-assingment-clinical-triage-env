from __future__ import annotations
import uuid
from typing import Any, Dict, Optional, Tuple, Union

from models import (
    ClinicalState,
    TriageAction, TriageObservation,
    MedicationSafetyAction, MedicationSafetyObservation,
    SepsisManagementAction, SepsisManagementObservation,
)
from scenarios import TRIAGE_SCENARIOS, MEDICATION_SCENARIOS, SEPSIS_SCENARIOS
from graders import TriageGrader, MedicationSafetyGrader, SepsisGrader

# Task metadata
TASK_REGISTRY = {
    # ── Task 1: ED Triage ──────────────────────────────────────
    "triage_easy": {
        "name": "Emergency Triage - Easy",
        "type": "triage",
        "scenario_key": "triage_easy_01",
        "difficulty": "easy",
        "max_steps": 3,
        "description": (
            "Assign the correct ESI (Emergency Severity Index) triage level "
            "to the patient and identify any immediate interventions needed. "
            "ESI levels: 1=Resuscitation, 2=Emergent, 3=Urgent, 4=Less Urgent, 5=Non-Urgent."
        )
    },
    "triage_medium": {
        "name": "Emergency Triage - Medium",
        "type": "triage",
        "scenario_key": "triage_medium_01",
        "difficulty": "medium",
        "max_steps": 3,
        "description": (
            "Triage a patient presenting with potential ACS (acute coronary syndrome). "
            "Assign ESI level, provide clinical rationale, and list critical immediate interventions."
        )
    },
    "triage_hard": {
        "name": "Emergency Triage - Hard",
        "type": "triage",
        "scenario_key": "triage_hard_01",
        "difficulty": "hard",
        "max_steps": 3,
        "description": (
            "Triage a complex patient with acute neurological symptoms and significant comorbidities. "
            "Consider anticoagulation status and time-critical interventions."
        )
    },
    # ── Task 2: Medication Safety ───────────────────────────────
    "med_safety_easy": {
        "name": "Medication Safety Review - Easy",
        "type": "medication_safety",
        "scenario_key": "med_easy_01",
        "difficulty": "easy",
        "max_steps": 3,
        "description": (
            "Review a patient's medication list for drug interactions, contraindications, "
            "and dosing errors. Provide a safety assessment and recommended changes."
        )
    },
    "med_safety_medium": {
        "name": "Medication Safety Review - Medium",
        "type": "medication_safety",
        "scenario_key": "med_medium_01",
        "difficulty": "medium",
        "max_steps": 3,
        "description": (
            "Review a post-cardiac catheterization patient on triple antithrombotic therapy "
            "with comorbidities including CKD and diabetes."
        )
    },
    "med_safety_hard": {
        "name": "Medication Safety Review - Hard",
        "type": "medication_safety",
        "scenario_key": "med_hard_01",
        "difficulty": "hard",
        "max_steps": 3,
        "description": (
            "A patient on HIV antiretrovirals presenting with rhabdomyolysis symptoms. "
            "Identify the life-threatening drug interaction causing this presentation."
        )
    },
    # ── Task 3: Sepsis Management ───────────────────────────────
    "sepsis_easy": {
        "name": "Sepsis Management - Easy",
        "type": "sepsis",
        "scenario_key": "sepsis_easy_01",
        "difficulty": "easy",
        "max_steps": 3,
        "description": (
            "Recognise sepsis criteria and execute the Hour-1 Surviving Sepsis Campaign bundle. "
            "Consider the patient's allergy profile when selecting antibiotics."
        )
    },
    "sepsis_medium": {
        "name": "Sepsis Management - Medium",
        "type": "sepsis",
        "scenario_key": "sepsis_medium_01",
        "difficulty": "medium",
        "max_steps": 3,
        "description": (
            "Manage septic shock in an elderly nursing home patient with MRSA history "
            "and multiple comorbidities. Requires vasopressor decision-making."
        )
    },
    "sepsis_hard": {
        "name": "Sepsis Management - Hard",
        "type": "sepsis",
        "scenario_key": "sepsis_hard_01",
        "difficulty": "hard",
        "max_steps": 3,
        "description": (
            "Post-operative septic shock with anastomotic leak, multi-organ failure, DIC, "
            "and vancomycin allergy. Navigate complex antibiotic choices and source control."
        )
    },
}


class ClinicalTriageEnv:
    """
    OpenEnv-compatible healthcare environment.
    
    Simulates three real-world clinical decision-making tasks:
      1. Emergency Department Triage (ESI Level Assignment)
      2. Medication Safety Review (Drug Interaction Detection)
      3. Sepsis Recognition & Management (Hour-1 Bundle)
    
    Each task has easy/medium/hard difficulty levels with programmatic grading.
    """

    def __init__(self, task_id: str = "triage_easy"):
        if task_id not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Valid tasks: {list(TASK_REGISTRY.keys())}"
            )
        self.task_id = task_id
        self.task_meta = TASK_REGISTRY[task_id]

        # Graders
        self._triage_grader = TriageGrader()
        self._med_grader = MedicationSafetyGrader()
        self._sepsis_grader = SepsisGrader()

        # Episode state
        self._state = ClinicalState(episode_id=str(uuid.uuid4()))
        self._last_feedback = ""
        self._scenario = self._load_scenario()

    # ──────────────────────────────────────────────
    # OpenEnv Core API
    # ──────────────────────────────────────────────

    def reset(self, task_id: Optional[str] = None) -> Union[
        TriageObservation, MedicationSafetyObservation, SepsisManagementObservation
    ]:
        """Reset the environment, optionally switching to a new task."""
        if task_id and task_id in TASK_REGISTRY:
            self.task_id = task_id
            self.task_meta = TASK_REGISTRY[task_id]

        self._state = ClinicalState(
            episode_id=str(uuid.uuid4()),
            task_id=self.task_id,
            task_name=self.task_meta["name"],
        )
        self._last_feedback = ""
        self._scenario = self._load_scenario()

        return self._build_observation(done=False, reward=None, feedback="")

    def step(
        self,
        action: Union[TriageAction, MedicationSafetyAction, SepsisManagementAction]
    ) -> Tuple[
        Union[TriageObservation, MedicationSafetyObservation, SepsisManagementObservation],
        float, bool, Dict[str, Any]
    ]:
        """
        Execute an action in the environment.
        Returns: (observation, reward, done, info)
        """
        if self._state.is_done:
            obs = self._build_observation(done=True, reward=0.0, feedback="Episode already done. Call reset().")
            return obs, 0.0, True, {"error": "episode_done"}

        self._state.step_count += 1
        task_type = self.task_meta["type"]
        max_steps = self.task_meta["max_steps"]

        # Grade the action
        grade_result = self._grade_action(action, task_type)

        # Reward shaping: partial rewards throughout episode
        step_reward = self._compute_step_reward(grade_result, self._state.step_count, max_steps)
        self._state.total_reward += step_reward
        self._state.partial_scores[f"step_{self._state.step_count}"] = grade_result.score

        # Episode done on first graded step (single-turn tasks) or max steps
        done = (self._state.step_count >= max_steps or grade_result.score >= 0.0)
        # For these clinical tasks, one substantive action ends the episode
        done = True  # Each task is a single clinical decision
        self._state.is_done = done

        # Build feedback
        feedback_lines = [
            grade_result.feedback,
            f"\nStep Score: {grade_result.score:.3f}",
            f"Episode Total Reward: {self._state.total_reward:.3f}",
        ]
        if grade_result.critical_errors:
            feedback_lines.append("\n⚠️  PATIENT SAFETY ALERTS:")
            feedback_lines.extend([f"  ⚠ {e}" for e in grade_result.critical_errors])
        feedback_lines.append(f"\n{'✅ PASSED' if grade_result.passed else '❌ NOT PASSED'} (threshold: 0.60)")

        self._last_feedback = "\n".join(feedback_lines)
        self._state.metadata["last_grade"] = {
            "score": grade_result.score,
            "component_scores": grade_result.component_scores,
            "critical_errors": grade_result.critical_errors,
            "passed": grade_result.passed,
        }

        obs = self._build_observation(done=done, reward=step_reward, feedback=self._last_feedback)

        info = {
            "grade": grade_result.score,
            "component_scores": grade_result.component_scores,
            "critical_errors": grade_result.critical_errors,
            "passed": grade_result.passed,
            "total_reward": self._state.total_reward,
            "task_id": self.task_id,
            "difficulty": self.task_meta["difficulty"],
        }

        return obs, step_reward, done, info

    def state(self) -> ClinicalState:
        """Return current episode state."""
        return self._state

    # ──────────────────────────────────────────────
    # Helper Methods
    # ──────────────────────────────────────────────

    def _load_scenario(self) -> Dict[str, Any]:
        scenario_key = self.task_meta["scenario_key"]
        task_type = self.task_meta["type"]

        if task_type == "triage":
            return TRIAGE_SCENARIOS[scenario_key]
        elif task_type == "medication_safety":
            return MEDICATION_SCENARIOS[scenario_key]
        elif task_type == "sepsis":
            return SEPSIS_SCENARIOS[scenario_key]
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def _grade_action(self, action, task_type):
        if task_type == "triage":
            return self._triage_grader.grade(action, self._scenario)
        elif task_type == "medication_safety":
            return self._med_grader.grade(action, self._scenario)
        elif task_type == "sepsis":
            return self._sepsis_grader.grade(action, self._scenario)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def _compute_step_reward(self, grade_result, step_num: int, max_steps: int) -> float:
        """
        Shaped reward function providing signal throughout the episode.
        
        Reward components:
          - Base score (0–1): direct grade
          - Speed bonus: faster/fewer steps = small bonus
          - Safety penalty: critical errors penalized heavily
          - Difficulty multiplier: harder tasks give higher max reward
        """
        difficulty_multiplier = {"easy": 0.8, "medium": 1.0, "hard": 1.3}[self.task_meta["difficulty"]]

        base_reward = grade_result.score

        # Critical error penalty (patient safety)
        safety_penalty = 0.3 * len(grade_result.critical_errors)

        # Step efficiency: slight bonus for solving in fewer steps
        efficiency_bonus = 0.05 * max(0, (max_steps - step_num))

        raw_reward = (base_reward - safety_penalty + efficiency_bonus) * difficulty_multiplier
        return round(max(-1.0, min(1.5, raw_reward)), 4)

    def _build_observation(self, done: bool, reward: Optional[float], feedback: str):
        task_type = self.task_meta["type"]
        patient = self._scenario["patient"]
        task_desc = self.task_meta["description"]
        step = self._state.step_count
        max_steps = self.task_meta["max_steps"]
        score = self._state.total_reward

        if task_type == "triage":
            return TriageObservation(
                patient=patient,
                task_description=task_desc,
                current_step=step,
                max_steps=max_steps,
                feedback=feedback,
                score_so_far=score,
                done=done,
                reward=reward,
            )
        elif task_type == "medication_safety":
            # Provide drug reference info (simulates formulary lookup)
            drug_info = self._get_drug_reference_info()
            return MedicationSafetyObservation(
                patient=patient,
                task_description=task_desc,
                current_step=step,
                max_steps=max_steps,
                feedback=feedback,
                score_so_far=score,
                done=done,
                reward=reward,
                available_drug_info=drug_info,
            )
        elif task_type == "sepsis":
            qsofa = self._compute_qsofa(patient)
            return SepsisManagementObservation(
                patient=patient,
                task_description=task_desc,
                current_step=step,
                max_steps=max_steps,
                feedback=feedback,
                score_so_far=score,
                done=done,
                reward=reward,
                time_elapsed_minutes=patient.arrival_time_minutes,
                qsofa_score=qsofa,
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def _compute_qsofa(self, patient) -> int:
        """Compute quick SOFA score (0-3)."""
        score = 0
        if patient.vitals.systolic_bp <= 100:
            score += 1
        if patient.vitals.respiratory_rate >= 22:
            score += 1
        if patient.vitals.glasgow_coma_scale < 15:
            score += 1
        return score

    def _get_drug_reference_info(self) -> Dict[str, Any]:
        """
        Returns a simulated drug reference/formulary information
        relevant to the current scenario's medications.
        """
        patient = self._scenario["patient"]
        reference = {}
        for med in patient.current_medications:
            name = med.name.lower()
            if "warfarin" in name:
                reference["warfarin"] = {
                    "class": "anticoagulant",
                    "interactions": ["aspirin (increased bleeding)", "NSAIDs", "amiodarone", "fluconazole"],
                    "monitoring": "INR target 2-3 (mechanical valve: 2.5-3.5)"
                }
            if "metformin" in name:
                reference["metformin"] = {
                    "class": "biguanide_antidiabetic",
                    "contraindications": ["eGFR<30 (hold if <45 in some guidelines)", "IV contrast (hold)", "hepatic failure", "sepsis"],
                    "max_dose": "2000mg/day"
                }
            if "simvastatin" in name:
                reference["simvastatin"] = {
                    "class": "HMG-CoA reductase inhibitor (statin)",
                    "interactions": ["CYP3A4 inhibitors (ritonavir, itraconazole, erythromycin) - CONTRAINDICATED",
                                     "amiodarone", "diltiazem"],
                    "FDA_note": "Max 20mg with CYP3A4 inhibitors; 80mg dose restricted since 2011",
                    "toxicity": "Myopathy, rhabdomyolysis (CK monitoring)"
                }
            if "ritonavir" in name:
                reference["ritonavir"] = {
                    "class": "HIV protease inhibitor (PI) + pharmacokinetic booster",
                    "CYP3A4": "POTENT INHIBITOR - increases levels of CYP3A4 substrates by up to 3000%",
                    "statin_interaction": "Simvastatin/lovastatin CONTRAINDICATED. Use pravastatin or rosuvastatin."
                }
            if "clopidogrel" in name:
                reference["clopidogrel"] = {
                    "class": "P2Y12 antiplatelet",
                    "interactions": ["warfarin + aspirin = triple therapy (major bleed risk)"],
                    "note": "Triple antithrombotic therapy guideline: minimize duration, consider PPI"
                }
        return reference

    @staticmethod
    def list_tasks() -> Dict[str, Dict]:
        """List all available tasks with metadata."""
        return {
            k: {
                "name": v["name"],
                "type": v["type"],
                "difficulty": v["difficulty"],
                "description": v["description"][:100] + "..."
            }
            for k, v in TASK_REGISTRY.items()
        }
