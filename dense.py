import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import numpy as np

class DenseLayer:
  def __init__(self, mi, mo):
    self.W = tf.Variable((np.random.randn(mi, mo) * np.sqrt(2.0 / mi)).astype(np.float32))
    self.b = tf.Variable(np.zeros(mo, dtype=np.float32))

  def forward(self, X):
    return tf.matmul(X, self.W) + self.b

  def copyFromKerasLayers(self, layer):
    W, b = layer.get_weights()
    op1 = self.W.assign(W)
    op2 = self.b.assign(b)
    self.session.run((op1, op2))

  def get_params(self):
    return [self.W, self.b]
