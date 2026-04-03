import random
import math
from collections import defaultdict, deque

# ── Constants ──────────────────────────────────────────────────────────────

ACTIONS = [
    "Assign High Priority",
    "Assign Medium Priority",
    "Assign Low Priority",
    "Discharge Patient",
]

ACTION_IDX  = {a: i for i, a in enumerate(ACTIONS)}
IDX_ACTION  = {i: a for i, a in enumerate(ACTIONS)}

# ── State Featuriser ───────────────────────────────────────────────────────

def featurise(state: dict) -> tuple:
    """
    Convert a raw env state dict into a discrete hashable tuple
    suitable as a Q-table key.
    Bins:
      spo2_zone:  0=crisis(<90), 1=low(90-94), 2=ok(95-100)
      hr_zone:    0=brady(<60),  1=normal(60-100), 2=tachy(>100)
      bp_zone:    0=hypo(<90),   1=normal, 2=hyper(>160)
      age_zone:   0=child(<18),  1=adult, 2=elderly(>65)
      red_flag:   0/1
      amber_flag: 0/1
    """
    spo2 = state.get("oxygen_level", 98)
    hr   = state.get("heart_rate", 75)
    bp   = state.get("blood_pressure", "120/80")
    age  = state.get("age", 35)
    syms = [s.lower() for s in state.get("symptoms", [])]

    spo2_zone = 0 if spo2 < 90 else (1 if spo2 < 95 else 2)
    hr_zone   = 0 if hr < 60  else (2 if hr > 100 else 1)

    try:
        sys_bp = int(bp.split("/")[0])
    except Exception:
        sys_bp = 120
    bp_zone = 0 if sys_bp < 90 else (2 if sys_bp > 160 else 1)

    age_zone = 0 if age < 18 else (2 if age > 65 else 1)

    red_kw = ["chest pain","loss of consciousness","unresponsive","stroke","anaphylaxis",
              "massive bleeding","cyanosis","respiratory distress","throat swelling",
              "facial droop","slurred speech","shortness of breath"]
    amb_kw = ["fever","stiff neck","abdominal pain","vomiting","confusion",
              "headache","wheezing","broken bone","back pain"]

    red_flag  = int(any(k in s for k in red_kw for s in syms))
    amber_flag = int(any(k in s for k in amb_kw for s in syms))

    return (spo2_zone, hr_zone, bp_zone, age_zone, red_flag, amber_flag)


# ── Replay Buffer ──────────────────────────────────────────────────────────

class ReplayBuffer:
    """Circular experience replay buffer."""
    def __init__(self, capacity: int = 500):
        self._buf = deque(maxlen=capacity)

    def push(self, state_feat, action_idx, reward, next_feat, done):
        self._buf.append((state_feat, action_idx, reward, next_feat, done))

    def sample(self, n: int):
        return random.sample(self._buf, min(n, len(self._buf)))

    def __len__(self):
        return len(self._buf)


# ── Q-Learning Agent ───────────────────────────────────────────────────────

class QLearningAgent:
    """
    Tabular Q-Learning agent with:
      - ε-greedy exploration with exponential decay
      - Experience replay (batch Q-updates)
      - Episode tracking & analytics
      - Policy extraction & confidence scoring
    """

    def __init__(
        self,
        lr:           float = 0.15,
        gamma:        float = 0.90,
        epsilon:      float = 1.0,
        epsilon_min:  float = 0.05,
        epsilon_decay: float = 0.97,
        replay_batch: int   = 16,
    ):
        self.lr            = lr
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.replay_batch  = replay_batch

        # Q-table: defaultdict → array of Q-values per action
        self.q_table: dict = defaultdict(lambda: [0.0] * len(ACTIONS))

        # Replay buffer
        self.buffer = ReplayBuffer(capacity=1000)

        # Analytics
        self.episode_rewards:   list = []
        self.episode_accuracies: list = []
        self.epsilon_history:   list = []
        self.total_episodes:    int  = 0
        self.total_steps:       int  = 0

        # Visit counts per (state, action) for policy confidence
        self.visit_counts: dict = defaultdict(lambda: [0] * len(ACTIONS))

        # Current episode tracking
        self._ep_reward    = 0.0
        self._ep_correct   = 0
        self._ep_steps     = 0
        self._last_feat    = None

    # ── Core API ──────────────────────────────────────────────────────────

    def select_action(self, state: dict) -> tuple[str, str, float]:
        """
        ε-greedy action selection.
        Returns (action_str, mode_label, confidence_score).
        """
        feat = featurise(state)
        self._last_feat = feat

        if random.random() < self.epsilon:
            idx  = random.randrange(len(ACTIONS))
            mode = f"🎲 Exploring (ε={self.epsilon:.3f})"
        else:
            q_vals = self.q_table[feat]
            idx    = q_vals.index(max(q_vals))
            mode   = f"🧠 Exploiting (ε={self.epsilon:.3f})"

        self.visit_counts[feat][idx] += 1
        confidence = self._confidence(feat, idx)
        return ACTIONS[idx], mode, confidence

    def update(self, state: dict, action: str, reward: float,
               next_state: dict, done: bool):
        """Store transition and perform batch Q-update."""
        feat      = featurise(state)
        next_feat = featurise(next_state)
        act_idx   = ACTION_IDX[action]

        self.buffer.push(feat, act_idx, reward, next_feat, done)
        self._ep_reward += reward
        self._ep_steps  += 1
        self.total_steps += 1

        # Batch replay update
        batch = self.buffer.sample(self.replay_batch)
        for sf, ai, r, nf, d in batch:
            target = r if d else r + self.gamma * max(self.q_table[nf])
            self.q_table[sf][ai] += self.lr * (target - self.q_table[sf][ai])

        if done:
            self._end_episode()

    def _end_episode(self):
        self.total_episodes    += 1
        self.episode_rewards.append(round(self._ep_reward, 2))
        self.epsilon_history.append(round(self.epsilon, 4))
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        # Reset episode state
        self._ep_reward  = 0.0
        self._ep_steps   = 0

    # ── Policy & Analytics ────────────────────────────────────────────────

    def get_policy(self) -> dict:
        """Return greedy policy: state_feat → best_action."""
        return {
            feat: IDX_ACTION[q_vals.index(max(q_vals))]
            for feat, q_vals in self.q_table.items()
        }

    def get_q_values(self, state: dict) -> dict:
        """Return Q-values for all actions given current state."""
        feat = featurise(state)
        q    = self.q_table[feat]
        return {a: round(q[i], 3) for i, a in enumerate(ACTIONS)}

    def get_value_estimate(self, state: dict) -> float:
        """V(s) = max_a Q(s,a)."""
        feat = featurise(state)
        return round(max(self.q_table[feat]), 3)

    def _confidence(self, feat, idx) -> float:
        """Confidence = visits to best action / total visits for this state."""
        visits = self.visit_counts[feat]
        total  = sum(visits) + 1e-9
        return round(visits[idx] / total * 100, 1)

    def get_analytics(self) -> dict:
        """Return full analytics dict for UI display."""
        rewards = self.episode_rewards
        n = len(rewards)
        if n == 0:
            return {
                "total_episodes": 0,
                "total_steps": 0,
                "mean_reward": 0,
                "best_reward": 0,
                "worst_reward": 0,
                "recent_mean": 0,
                "q_table_size": 0,
                "epsilon": round(self.epsilon, 4),
                "trend": "No data yet",
                "rewards_history": [],
                "epsilon_history": [],
            }

        recent = rewards[-min(10, n):]
        trend  = "📈 Improving" if (len(recent) > 1 and recent[-1] > recent[0]) else \
                 "📉 Declining" if (len(recent) > 1 and recent[-1] < recent[0]) else "➡️ Stable"

        return {
            "total_episodes":  self.total_episodes,
            "total_steps":     self.total_steps,
            "mean_reward":     round(sum(rewards) / n, 2),
            "best_reward":     max(rewards),
            "worst_reward":    min(rewards),
            "recent_mean":     round(sum(recent) / len(recent), 2),
            "q_table_size":    len(self.q_table),
            "epsilon":         round(self.epsilon, 4),
            "trend":           trend,
            "rewards_history": rewards[-50:],
            "epsilon_history": self.epsilon_history[-50:],
        }

    def get_policy_heatmap_data(self) -> list[dict]:
        """
        Return policy table rows for display.
        Each row: spo2_zone, hr_zone, best_action, confidence, visits.
        """
        rows = []
        spo2_labels = ["🔴 Crisis (<90%)", "🟡 Low (90-94%)", "🟢 Normal (95%+)"]
        hr_labels   = ["⬇️ Brady (<60)", "✅ Normal (60-100)", "⬆️ Tachy (>100)"]
        action_emojis = {"Assign High Priority":"🔴","Assign Medium Priority":"🟡",
                         "Assign Low Priority":"🟢","Discharge Patient":"⚪"}

        seen = set()
        for feat, q_vals in self.q_table.items():
            key = (feat[0], feat[1])  # spo2_zone, hr_zone
            if key in seen: continue
            seen.add(key)
            best_idx = q_vals.index(max(q_vals))
            best_act = IDX_ACTION[best_idx]
            visits   = sum(self.visit_counts[feat])
            conf     = self._confidence(feat, best_idx)
            rows.append({
                "SpO₂ Zone":    spo2_labels[feat[0]],
                "HR Zone":      hr_labels[feat[1]],
                "Policy Action": f"{action_emojis[best_act]} {best_act}",
                "Confidence":   f"{conf}%",
                "Visits":       visits,
                "Q-Value":      round(max(q_vals), 2),
            })

        rows.sort(key=lambda r: -r["Visits"])
        return rows[:20]  # top 20 most-visited states

    def run_training_episode(self, env_class) -> dict:
        """
        Run one full autonomous training episode.
        Returns episode summary dict.
        """
        from environment import ClinicalTriageEnv
        env   = ClinicalTriageEnv()
        state = env.reset()
        ep_reward = 0.0

        for _ in range(3):
            action, mode, conf = self.select_action(state)
            result = env.step(action)
            next_state = result["state"]
            reward     = result["reward"]
            done       = result["done"]

            self.update(state, action, reward, next_state, done)
            ep_reward += reward
            state = next_state
            if done:
                break

        summary = env.get_summary()
        return {
            "episode":    self.total_episodes,
            "reward":     round(ep_reward, 2),
            "severity":   summary["true_severity"],
            "accuracy":   summary["decision_accuracy"],
            "safety":     summary["safety_score"],
        }
