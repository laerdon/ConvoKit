from abc import ABC, abstractmethod
from typing import Callable, Tuple, Optional, Dict, Any

import numpy as np
from sklearn.metrics import roc_curve
from tqdm import tqdm


class DecisionPolicy(ABC):
    """
    Abstract interface for converting a conversational context into an action.
    """

    def __init__(self):
        self._labeler = None

    @property
    def labeler(self):
        return self._labeler

    @labeler.setter
    def labeler(self, value: Callable):
        self._labeler = value

    def _fit_with_model_checkpoint_selection(self, val_contexts, score_fn: Callable = None):
        if score_fn is None:
            return None
        forecaster_model = getattr(score_fn, "__self__", None)
        if forecaster_model is None:
            return None
        get_checkpoints = getattr(forecaster_model, "get_checkpoints", None)
        load_checkpoint = getattr(forecaster_model, "load_checkpoint", None)
        finalize_best_checkpoint_selection = getattr(
            forecaster_model, "finalize_best_checkpoint_selection", None
        )
        if not callable(get_checkpoints) or not callable(load_checkpoint):
            return None

        checkpoints = list(get_checkpoints())
        if len(checkpoints) == 0:
            return None

        best_config = None
        best_checkpoint = None
        best_val_accuracy = -1.0
        for checkpoint_name in checkpoints:
            load_checkpoint(checkpoint_name)
            fit_result = self._fit_threshold_for_loaded_model(val_contexts, score_fn=score_fn)
            print(f"accuracy: {checkpoint_name} {fit_result['best_val_accuracy']}")
            if fit_result["best_val_accuracy"] > best_val_accuracy:
                best_checkpoint = checkpoint_name
                best_val_accuracy = fit_result["best_val_accuracy"]
                best_config = {
                    "best_checkpoint": checkpoint_name,
                    "best_threshold": float(fit_result["best_threshold"]),
                    "best_val_accuracy": float(fit_result["best_val_accuracy"]),
                }

        if best_config is None:
            return None

        if hasattr(self, "threshold"):
            self.threshold = float(best_config["best_threshold"])
        load_checkpoint(best_checkpoint)
        if callable(finalize_best_checkpoint_selection):
            finalize_best_checkpoint_selection(
                best_checkpoint,
                best_config,
                val_contexts=val_contexts,
                score_fn=score_fn,
            )
        return best_config

    def _fit_threshold_for_loaded_model(self, val_contexts, score_fn: Callable):
        y_true, y_score = self._get_validation_arrays(val_contexts, score_fn)
        default_threshold = float(getattr(self, "threshold", 0.5))
        if len(y_true) == 0:
            return {"best_threshold": default_threshold, "best_val_accuracy": 0.0}

        try:
            _, _, thresholds = roc_curve(y_true, y_score)
        except ValueError:
            thresholds = np.asarray([default_threshold], dtype=float)

        if len(thresholds) == 0:
            thresholds = np.asarray([default_threshold], dtype=float)

        accs = [((y_score > t).astype(int) == y_true).mean() for t in thresholds]
        best_idx = int(np.argmax(accs))
        best_threshold = float(thresholds[best_idx])
        return {"best_threshold": best_threshold, "best_val_accuracy": float(accs[best_idx])}

    def _get_validation_arrays(self, val_contexts, score_fn: Callable):
        highest_convo_scores = {}
        convo_labels = {}
        for context in tqdm(val_contexts):
            convo_id = context.conversation_id
            score = float(score_fn(context))
            label = int(self.labeler(context.current_utterance.get_conversation()))
            if convo_id not in highest_convo_scores:
                highest_convo_scores[convo_id] = score
            else:
                highest_convo_scores[convo_id] = max(highest_convo_scores[convo_id], score)
            convo_labels[convo_id] = label

        convo_ids = list(highest_convo_scores.keys())
        y_true = np.asarray([convo_labels[c] for c in convo_ids])
        y_score = np.asarray([highest_convo_scores[c] for c in convo_ids])
        return y_true, y_score

    @abstractmethod
    def decide(self, context, score_fn: Callable) -> Tuple[float, int, Optional[Dict[str, Any]]]:
        """
        Decide whether to intervene for a context.

        :param context: context tuple supplied by Forecaster
        :param score_fn: callable that maps a context tuple to a scalar score
        :return: tuple containing the score, the integer action label (currently 0/1), and any additional metadata
        """
        pass

    @abstractmethod
    def fit(self, contexts, val_contexts=None, score_fn: Callable = None):
        """
        Fit policy-specific parameters if needed.

        :param contexts: training contexts for policy fitting
        :param val_contexts: optional validation contexts
        :param score_fn: optional scorer callable exposed by ForecasterModel
        """
        pass