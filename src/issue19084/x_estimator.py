from typing import Optional, Dict, Any
from os import PathLike
import tensorflow as tf
from tensorflow.python.estimator.canned.head import _Head, _RegressionHeadWithMeanSquaredErrorLoss

from .model import Model


class XEstimator(tf.estimator.Estimator):
  def __init__(self, model_dir: PathLike, config: Optional[tf.estimator.RunConfig] = None):
    model = Model()
    head: _Head = _RegressionHeadWithMeanSquaredErrorLoss(label_dimension=1)

    def model_fn(
        features: Dict[str, tf.Tensor],
        labels: Optional[tf.Tensor],
        mode: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        config: Optional[tf.estimator.RunConfig] = None
    ) -> tf.estimator.EstimatorSpec:
      x = features["x"]
      logits = model.apply(x, mode == tf.estimator.ModeKeys.TRAIN)
      return head.create_estimator_spec(
        features=features,
        mode=mode,
        logits=logits,
        labels=labels,
        optimizer=tf.train.AdamOptimizer()
      )

    super(XEstimator, self).__init__(model_fn=model_fn, model_dir=model_dir, config=config)
