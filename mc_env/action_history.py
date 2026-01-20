from collections import deque

import numpy as np

HISTORY_LENGTH = 1

class ActionHistory:
    def __init__(self):
        self.action_dim = 5  # 5D action vector
        self.history = deque(maxlen=HISTORY_LENGTH)

    def save_action(self, action: np.ndarray):
        self.history.append(action)

    def clear(self):
        self.history.clear()

    def get_action_history_features(self, decay: float = 0.8) -> np.ndarray:
        """
        Return concatenated, decayed history (up to HISTORY_LENGTH actions).
        Most recent action has the highest weight (1.0), older decay exponentially.
        """
        if not self.history:
            return np.zeros(HISTORY_LENGTH * self.action_dim, dtype=np.float32)

        # Convert deque → list, take up to max_length (most recent)
        hist = list(self.history)[-HISTORY_LENGTH:]

        # Reverse to most recent first for weighting
        recent_first = hist[::-1]  # newest → oldest

        weighted = []
        weight = 1.0
        for action_vec in recent_first:
            weighted.append(weight * action_vec)
            weight *= decay

        # Reverse back to chronological order (oldest → newest)
        weighted = weighted[::-1]

        # Pad if too short
        while len(weighted) < HISTORY_LENGTH:
            weighted.append(np.zeros(self.action_dim, dtype=np.float32))

        return np.concatenate(weighted).astype(np.float32)
