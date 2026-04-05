"""Model training/evaluation/quantization scaffolding."""

# 中文说明：集中放置训练、评估、量化入口脚本。

from .tiny_cnn import TinyCNN, TinyCNNConfig, fit_tiny_cnn, predict_from_artifact, predict_from_loaded_artifact

__all__ = ["TinyCNN", "TinyCNNConfig", "fit_tiny_cnn", "predict_from_artifact", "predict_from_loaded_artifact"]
