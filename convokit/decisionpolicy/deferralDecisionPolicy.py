from itertools import tee
from typing import Callable, List, Optional

import numpy as np
from sklearn.metrics import roc_curve

from .decisionPolicy import DecisionPolicy


class _synthetic_speaker:
    def __init__(self, speaker_id: str):
        self.id = speaker_id


class _synthetic_utterance:
    def __init__(self, text: str, utterance_id: str, speaker_id: str):
        self.text = text
        self.id = utterance_id
        self.speaker_ = _synthetic_speaker(speaker_id)
        self.meta = {}

    def get_conversation(self):
        return None


class DeferralDecisionPolicy(DecisionPolicy):
    """
    Decision policy that can defer intervention using simulated next utterances.
    """

    def __init__(
        self,
        simulator=None,
        threshold: float = 0.5,
        num_simulations: int = 3,
        aggregation: str = "mean",
    ):
        super().__init__()
        self.simulator = simulator
        self.threshold = float(threshold)
        self.num_simulations = int(num_simulations)
        self.aggregation = aggregation

    def _aggregate_scores(self, scores: List[float]) -> float:
        if len(scores) == 0:
            return 0.0
        if self.aggregation == "max":
            return float(np.max(scores))
        if self.aggregation == "min":
            return float(np.min(scores))
        return float(np.mean(scores))

    def get_simulations(self, context, simulator=None, k: Optional[int] = None) -> List[str]:
        simulator = simulator if simulator is not None else self.simulator
        if k is None:
            k = self.num_simulations
        if simulator is None:
            return []
        if callable(simulator):
            sims = simulator(context, k)
            return list(sims)[:k]
        if hasattr(simulator, "get_simulations"):
            sims = simulator.get_simulations(context, k)
            return list(sims)[:k]
        if hasattr(simulator, "transform"):
            sims = simulator.transform(iter([context]))
            if context.current_utterance.id in sims.index:
                col_name = sims.columns[0]
                return list(sims.loc[context.current_utterance.id][col_name])[:k]
        return []

    def _build_simulated_context(self, context, simulation_text: str, simulation_idx: int):
        current_utt = context.current_utterance
        synthetic_utt = _synthetic_utterance(
            text=simulation_text,
            utterance_id=f"{current_utt.id}__sim_{simulation_idx}",
            speaker_id="simulator",
        )
        new_context_utts = list(context.context) + [synthetic_utt]
        context_cls = context.__class__
        return context_cls(
            context=new_context_utts,
            current_utterance=synthetic_utt,
            future_context=None,
            conversation_id=context.conversation_id,
        )

    def _decision_score(self, context, score_fn: Callable) -> float:
        current_score = float(score_fn(context))
        simulations = self.get_simulations(context)
        if len(simulations) == 0:
            return current_score
        simulation_scores = []
        for idx, sim_text in enumerate(simulations):
            sim_context = self._build_simulated_context(context, sim_text, idx)
            simulation_scores.append(float(score_fn(sim_context)))
        return self._aggregate_scores([current_score] + simulation_scores)

    def decide(self, context, score_fn: Callable) -> int:
        decision_score = self._decision_score(context, score_fn)
        return int(decision_score > self.threshold)

    def fit(self, contexts, val_contexts=None, score_fn: Callable = None):
        if self.simulator is not None and hasattr(self.simulator, "fit"):
            if val_contexts is None:
                sim_contexts = contexts
                sim_val_contexts = None
            else:
                sim_contexts, contexts = tee(contexts, 2)
                sim_val_contexts, val_contexts = tee(val_contexts, 2)
            self.simulator.fit(sim_contexts, sim_val_contexts)

        if val_contexts is None or score_fn is None or self.labeler is None:
            return {"threshold": self.threshold}
        val_contexts = list(val_contexts)
        if len(val_contexts) == 0:
            return {"threshold": self.threshold}

        highest_convo_scores = {}
        convo_labels = {}
        for context in val_contexts:
            convo_id = context.conversation_id
            score = self._decision_score(context, score_fn)
            label = int(self.labeler(context.current_utterance.get_conversation()))
            if convo_id not in highest_convo_scores:
                highest_convo_scores[convo_id] = score
            else:
                highest_convo_scores[convo_id] = max(highest_convo_scores[convo_id], score)
            convo_labels[convo_id] = label

        convo_ids = list(highest_convo_scores.keys())
        y_true = np.asarray([convo_labels[c] for c in convo_ids])
        y_score = np.asarray([highest_convo_scores[c] for c in convo_ids])
        try:
            _, _, thresholds = roc_curve(y_true, y_score)
        except ValueError:
            return {"threshold": self.threshold}
        if len(thresholds) == 0:
            return {"threshold": self.threshold}

        accs = [((y_score > t).astype(int) == y_true).mean() for t in thresholds]
        best_idx = int(np.argmax(accs))
        self.threshold = float(thresholds[best_idx])
        return {"threshold": self.threshold, "best_val_accuracy": float(accs[best_idx])}
