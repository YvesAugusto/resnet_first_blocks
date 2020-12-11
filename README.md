# resnet_first_blocks
Implements ConvLayer, BatchNorm, ResNet ConvBlock and concatenate then in order to compose ResNet50 first layers.

## Residual net justification
It is easier to neural networks to learn making the output of a function F(X) to be zero, than to learn the patterns of inputs X. So it was created the residual conv blocks, that combines the F(X) output with the X input, in order that the sum F(X) + X tends to equals X, since F(X) will be learned to be zero.

### Residual Net ConvBlock Scheme
![residual](https://user-images.githubusercontent.com/53539227/101893504-99155a00-3b83-11eb-8113-f5fc1153c59c.png)
