from abc import ABC, abstractmethod
from itertools import tee
from typing import Callable

from convokit.decisionpolicy import ThresholdDecisionPolicy


class ForecasterModel(ABC):
    """
    An abstract class defining an interface that Forecaster can call into to invoke a conversational forecasting algorithm.
    The “contract” between Forecaster and ForecasterModel means that ForecasterModel can expect to receive conversational data
    in a consistent format, defined above.
    """

    def __init__(self, decision_policy=None, **kwargs):
        self._labeler = None
        self._decision_policy = decision_policy or ThresholdDecisionPolicy()

    @property
    def labeler(self):
        return self._labeler

    @labeler.setter
    def labeler(self, value: Callable):
        self._labeler = value
        if self._decision_policy is not None:
            self._decision_policy.labeler = value

    @property
    def decision_policy(self):
        return self._decision_policy

    @decision_policy.setter
    def decision_policy(self, value):
        self._decision_policy = value
        if self._decision_policy is not None:
            self._decision_policy.labeler = self._labeler

    def fit(self, contexts, val_contexts=None):
        """
        Train this conversational forecasting model on the given data by fitting
        both the belief estimator and the decision policy.

        :param contexts: an iterator over context tuples
        :param val_contexts: an optional second iterator over context tuples to be used as a separate held-out validation set. Concrete ForecasterModel implementations may choose to ignore this, or conversely even enforce its presence.
        """
        belief_contexts, policy_contexts = tee(contexts, 2)
        if val_contexts is None:
            belief_val_contexts = None
            policy_val_contexts = None
        else:
            belief_val_contexts, policy_val_contexts = tee(val_contexts, 2)
        self.fit_belief_estimator(belief_contexts, belief_val_contexts)
        self.fit_decision_policy(policy_contexts, policy_val_contexts)

    @abstractmethod
    def fit_belief_estimator(self, contexts, val_contexts=None):
        """
        Fit only the belief estimator component that produces continuous scores.
        """
        pass

    def fit_decision_policy(self, contexts, val_contexts=None):
        """
        Fit only the decision policy component.
        """
        if self.decision_policy is not None:
            return self.decision_policy.fit(
                contexts=contexts, val_contexts=val_contexts, score_fn=self.score
            )
        return None

    @abstractmethod
    def score(self, context) -> float:
        """
        Produce the belief estimator score for a context.
        """
        pass

    def _predict(self, context):
        """
        Return both belief score and policy action for a context.
        """
        utt_score = self.score(context)
        utt_pred = self.decision_policy.decide(context, self.score)
        return utt_score, utt_pred

    @abstractmethod
    def transform(self, contexts, forecast_attribute_name, forecast_prob_attribute_name):
        """
        Apply this trained conversational forecasting model to the given data, and return its forecasts
        in the form of a DataFrame indexed by (current) utterance ID

        :param contexts: an iterator over context tuples

        :return: a Pandas DataFrame, with one row for each context, indexed by the ID of that context's current utterance. Contains two columns, one with raw probabilities named according to forecast_prob_attribute_name, and one with discretized (binary) forecasts named according to forecast_attribute_name. Subclass implementations of ForecasterModel MUST adhere to this return value specification!
        """
        pass
