from .reward_shaping import PoolRewardShaper, RewardOutcome

__all__ = [
    "CurriculumStage",
    "FormulaCurriculum",
    "GeneratorPretrainer",
    "MultiAgentTrainer",
    "MultiAgentTrainingEpisode",
    "MultiAgentTrainingSummary",
    "PoolRewardShaper",
    "PretrainingSummary",
    "RewardOutcome",
    "SingleAgentTrainer",
    "SyntheticRecoveryDatasetBuilder",
    "SyntheticRecoveryExample",
    "TrainingEpisode",
    "TrainingSummary",
]


def __getattr__(name: str):
    if name in {"CurriculumStage", "FormulaCurriculum"}:
        from .curriculum import CurriculumStage, FormulaCurriculum

        mapping = {
            "CurriculumStage": CurriculumStage,
            "FormulaCurriculum": FormulaCurriculum,
        }
        return mapping[name]
    if name in {
        "MultiAgentTrainer",
        "MultiAgentTrainingEpisode",
        "MultiAgentTrainingSummary",
        "SingleAgentTrainer",
        "TrainingEpisode",
        "TrainingSummary",
    }:
        from .trainer import (
            MultiAgentTrainer,
            MultiAgentTrainingEpisode,
            MultiAgentTrainingSummary,
            SingleAgentTrainer,
            TrainingEpisode,
            TrainingSummary,
        )

        mapping = {
            "MultiAgentTrainer": MultiAgentTrainer,
            "MultiAgentTrainingEpisode": MultiAgentTrainingEpisode,
            "MultiAgentTrainingSummary": MultiAgentTrainingSummary,
            "SingleAgentTrainer": SingleAgentTrainer,
            "TrainingEpisode": TrainingEpisode,
            "TrainingSummary": TrainingSummary,
        }
        return mapping[name]
    if name in {
        "GeneratorPretrainer",
        "PretrainingSummary",
        "SyntheticRecoveryDatasetBuilder",
        "SyntheticRecoveryExample",
    }:
        from .pretraining import GeneratorPretrainer, PretrainingSummary
        from .synthetic_recovery import SyntheticRecoveryDatasetBuilder, SyntheticRecoveryExample

        mapping = {
            "GeneratorPretrainer": GeneratorPretrainer,
            "PretrainingSummary": PretrainingSummary,
            "SyntheticRecoveryDatasetBuilder": SyntheticRecoveryDatasetBuilder,
            "SyntheticRecoveryExample": SyntheticRecoveryExample,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
