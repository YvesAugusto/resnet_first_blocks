import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import numpy as np

def init_filter(d, maps_input, maps_output, stride):
  return (np.random.randn(d, d, maps_input, maps_output) * np.sqrt(2.0 / (d * d * maps_input))).astype(np.float32)

class ConvLayer:
  def __init__(self, d, maps_input, maps_output, stride=2, padding='VALID'):
    self.W = tf.Variable(init_filter(d, maps_input, maps_output, stride))
    self.bias = tf.Variable(np.zeros(maps_output, dtype=np.float32))
    self.stride = stride
    self.padding = padding

  def forward(self, X):
    X = tf.nn.conv2d(
      X,
      self.W,
      strides=[self.stride, self.stride],
      padding=self.padding
    )
    X = X + self.bias
    return X

  def copyFromKerasLayers(self, layer):
    # only 1 layer to copy from
    W, bias = layer.get_weights()
    op1 = self.W.assign(W)
    op2 = self.bias.assign(bias)
    self.session.run((op1, op2))  
    
  # def copyFromWeights(self, W, b):
  #   op1 = self.W.assign(W)
  #   op2 = self.b.assign(b)
  #   self.session.run((op1, op2))

  def get_params(self):
    return [self.W, self.bias]

# init_op = tf.global_variables_initializer()
# conv=ConvLayer(1, 3, 64, 2)
# forward=conv.forward(np.random.random((1, 224, 224, 3)))
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     print(forward.eval().shape)
#     sess.close()
