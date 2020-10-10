# https://deeplearningcourses.com/c/advanced-computer-vision
# https://www.udemy.com/advanced-computer-vision
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future


# Let's go up to the end of the first conv block
# to make sure everything has been loaded correctly
# compared to keras
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt


from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

from conv_layer import ConvLayer
from batch_norm_layer import BatchNormLayer
from main import ConvBlock
import keras


class PartialResNet:
  def __init__(self, maps_input, fm_sizes, stride=2):
    # assert (len(fm_sizes) == 2)
    self.session = None
    self.f = tf.nn.relu
    self.stride=stride
    self.conv0=ConvLayer(1, maps_input, fm_sizes[0], stride)
    self.btn0=BatchNormLayer(fm_sizes[0])
    self.conv_block1 = ConvBlock(fm_sizes[0], [fm_sizes[0], 64, 256], stride=1,padding='SAME')
    self.btn_block1 = BatchNormLayer(256)

    self.layers=[
        self.conv0, self.btn0
    ]

    self.input_ = tf.placeholder(tf.float32, shape=(1, 224, 224, maps_input))
    self.output = self.forward(self.input_)
    # TODO
    pass

  def forward(self, X):
    FX = self.conv0.forward(X)
    FX = self.btn0.forward(FX)
    FX = self.f(FX)
    FX = tf.nn.max_pool2d(FX, strides=(self.stride, self.stride),ksize=[2,2], padding='VALID')
    FX = self.conv_block1.forward(FX)
    FX = self.btn_block1.forward(FX)

    return FX

  def copyFromKerasLayers(self, layers):
    # TODO
    pass

  def predict(self, X):
    assert (self.session is not None)
    return (
        self.session.run(
            self.output,
            feed_dict={self.input_: X}
        )
    )
    pass

  def set_session(self, session):
    self.session = session
    self.conv0.session = session
    self.conv_block1.session=session
    self.btn0.session=session
    # TODO: finish this

  def get_params(self):
    params = []
    # TODO: finish this

X = np.random.random((1, 224, 224, 3))

if __name__ == '__main__':

  my_net = PartialResNet(maps_input=3, fm_sizes=[64])
  X = X.astype(np.float32)
  a=my_net.forward(X)
  print(a.shape)