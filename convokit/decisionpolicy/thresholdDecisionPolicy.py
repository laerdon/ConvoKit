from typing import Callable

import numpy as np
from sklearn.metrics import roc_curve

from .decisionPolicy import DecisionPolicy


class ThresholdDecisionPolicy(DecisionPolicy):
    """
    A simple decision policy that predicts 1 when score > threshold.
    """

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = float(threshold)

    def decide(self, context, score_fn: Callable) -> int:
        return int(score_fn(context) > self.threshold)

    def fit(self, contexts, val_contexts=None, score_fn: Callable = None):
        if val_contexts is None or score_fn is None or self.labeler is None:
            return {"best_threshold": self.threshold}

        val_contexts = list(val_contexts)
        if len(val_contexts) == 0:
            return {"best_threshold": self.threshold}

        highest_convo_scores = {}
        convo_labels = {}
        for context in val_contexts:
            convo_id = context.conversation_id
            score = score_fn(context)
            label = int(self.labeler(context.current_utterance.get_conversation()))
            if convo_id not in highest_convo_scores:
                highest_convo_scores[convo_id] = score
            else:
                highest_convo_scores[convo_id] = max(highest_convo_scores[convo_id], score)
            convo_labels[convo_id] = label

        convo_ids = list(highest_convo_scores.keys())
        y_true = np.asarray([convo_labels[c] for c in convo_ids])
        y_score = np.asarray([highest_convo_scores[c] for c in convo_ids])

        # roc_curve can fail when only one class is present; keep current threshold in that case.
        try:
            _, _, thresholds = roc_curve(y_true, y_score)
        except ValueError:
            return {"best_threshold": self.threshold}

        if len(thresholds) == 0:
            return {"best_threshold": self.threshold}

        accs = [((y_score > t).astype(int) == y_true).mean() for t in thresholds]
        best_idx = int(np.argmax(accs))
        self.threshold = float(thresholds[best_idx])
        return {"best_threshold": self.threshold, "best_val_accuracy": float(accs[best_idx])}
