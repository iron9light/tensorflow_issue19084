import tensorflow as tf
from pathlib import Path
from issue19084.dataset import Dataset
from issue19084.x_estimator import XEstimator


def main():
  tf.logging.set_verbosity(tf.logging.INFO)

  config = tf.estimator.RunConfig(
    session_config=tf.ConfigProto(
      device_count={'GPU': 1},
      gpu_options=tf.GPUOptions(visible_device_list="1")
    )
  )

  data = Dataset()
  model_dir = Path("..", "temp", "model")
  estimator = XEstimator(model_dir=model_dir, config=config)
  estimator.train(input_fn=data.input_fn)


if __name__ == "__main__":
  main()
