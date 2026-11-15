"""Small models for dictionary meaning selection."""

from .schema import Candidate, Example, Target

__all__ = ["Candidate", "Example", "Target", "Predictor"]


def __getattr__(name):
    if name == "Predictor":
        from .predict import Predictor

        return Predictor
    raise AttributeError(name)
