from abc import ABC, abstractmethod
from typing import Callable


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

    @abstractmethod
    def decide(self, context, score_fn: Callable) -> int:
        """
        Decide whether to intervene for a context.

        :param context: context tuple supplied by Forecaster
        :param score_fn: callable that maps a context tuple to a scalar score
        :return: integer action label (currently 0/1)
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
