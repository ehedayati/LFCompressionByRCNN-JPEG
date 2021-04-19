import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers, models, initializers, Model, losses, optimizers
from tensorflow.keras.losses import Loss
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Input



# Full decompression Net
class LFDecompress(tf.keras.Model):
    """Light de-compression Model"""
    def __init__(self, spatialDim, angularDim):
        super(LFDecompress, self).__init__()
        self.spatialDim = spatialDim
        self.angularDim = angularDim
    def call(self, inputs):
        self.JpegHance = Jpeg_Hance(self.spatialDim,'centerNet')
        self.DepthNet = Depth_Net(self.spatialDim,self.angularDim,'depthNet')
    
    def jpegHance(self, jpegUncompressed,training=False):
        return self.JpegHance(jpegUncompressed,training)
        
    def depthEstimation(self, center,training=False):
        return self.DepthNet(center,training)
    
    def loadJpeg(self, path):
        self.JpegHance.load_weights(path)
    def loadDepth(self,path):
        self.DepthNet.load_weights(path)

# Jpeg-Hance
def Jpeg_Hance(SpatialDim, netName):
    centerInput = Input([SpatialDim[0],SpatialDim[1], 3])
    x1 = layers.Conv2D(64, (1,1), strides=(1,1),padding='same')(centerInput)
    x2 = JHBlock(filters=16, downSample=False)(x1)
    x3 = layers.Conv2D(128, (1,1), strides=(1,1),padding='same')(x2)
    x4 = JHBlock(filters=32, downSample=False)(x3)
    x5 = layers.Conv2D(256, (1,1), strides=(1,1),padding='same')(x4)
    x6 = JHBlock(filters=64, downSample=False)(x5)
    x7 = layers.Conv2D(128, (1,1), strides=(1,1),padding='same')(x6)+x4
    x8 = JHBlock(filters=32, downSample=False)(x7)+x3
    x9 = layers.Conv2D(64, (1,1), strides=(1,1),padding='same')(x8)+x2
    x10 = JHBlock(filters=16, downSample=False)(x9) + x1
    x11 = layers.Conv2D(filters=3, kernel_size=(3,3),padding='same')(x10)
    center = layers.Add()([x11,centerInput])
    center = layers.Activation('tanh', dtype='float32', name='last_tanh')(center)
    
    network = Model([centerInput],[center], name=netName)
    return network


# Jpeg-H residual block
class JHBlock(tf.keras.Model):
    def __init__(self, filters, downSample, **kwargs):
        super(JHBlock, self).__init__(**kwargs)

        self.conva = tf.keras.layers.Conv2D(filters, (1,1), padding='same')
        self.d = downSample
        self.bna = layers.BatchNormalization()
        self.bnb = layers.BatchNormalization()
        self.bnc = layers.BatchNormalization()
        self.convb = layers.Conv2D(filters, (3,3), padding='same')
        self.convc = layers.Conv2D(filters*4, (1,1), padding='same')
        self.convc_ds = layers.Conv2D(filters*4, (1,1), strides=(2,2),padding='same')
    
    def call(self, input_tensor, training=True):

        if self.d:
            ds = self.convc_ds(input_tensor)
            x = self.conva(ds)
        else:
            x = self.conva(input_tensor)
        x = self.bna(x, training=training)
        x = tf.nn.elu(x)
        x = self.convb(x)
        x = self.bnb(x, training=training)
        x = tf.nn.elu(x)
        x = self.convc(x)
        x = self.bnc(x, training=training)
        if self.d:
            x += ds
        else:
            x += input_tensor
        return tf.nn.elu(x)

# Depth-Net
def Depth_Net(spatialDim, angularDim, netName):
    initializer_net = models.Sequential(name='initializer')
    Input_shape = [spatialDim[0], spatialDim[1], 3]
    input_depth = Input(shape=Input_shape)
    filterSize = 64
    conv7by7 = layers.Conv2D(filterSize,kernel_size=(7,7), strides=(1,1),
                             input_shape=Input_shape, padding='same',activation='elu')(input_depth)
    filterSize = 128
    block128 = models.Sequential(name='resBlock128')
    dSample = True
    for i in range (1,4):
        block128.add(DepthBlock(filters=filterSize//2, downSample=dSample, name='resnetBlock1_' + str(i)))
        dSample = False
    b128 = block128(conv7by7)

    filterSize = 256    
    dSample = True
    block256 = models.Sequential(name='resBlock256')
    for i in range (1,5):
        block256.add(DepthBlock(filters=filterSize//2, downSample=dSample, name='resnetBlock2_' + str(i)))
        dSample = False
    b256 = block256(b128)

    block512 = models.Sequential(name='resBlock512')
    filterSize = 512
    dSample = True
    for i in range (1,7):
        block512.add(DepthBlock(filters=filterSize//2,downSample=dSample, name='resnetBlock3_' + str(i)))
        dSample = False
    b512 = block512(b256)

    block1024 = models.Sequential(name='resBlock1024')
    filterSize = 1024
    dSample = True
    for i in range (1,4):
        block1024.add(DepthBlock(filters=filterSize//2,downSample=dSample, name='resnetBlock4_' + str(i)))
        dSample = False
    b1024 = block1024(b512)
    
    u512 = UpsamplingBlock(filters=512, name='us_depth_512')(b1024) + b512

    u256 = UpsamplingBlock(filters=256, name='us_depth_256')(u512) \
    + UpsamplingBlock(filters=256, name='us_depth_256_2')(b512) + b256
    
    u128 = UpsamplingBlock(filters=128, name='us_depth_128')(u256) \
    + UpsamplingBlock(filters=128, name='us_depth_128_2')(b256) + b128
    
    u49 = UpsamplingBlock(filters=49, name='us_depth_64')(u128) \
    + UpsamplingBlock(filters=49, name='us_depth_64_2')(b128) 

    output_depth = layers.Reshape([spatialDim[0], spatialDim[1], angularDim[0],angularDim[0]])(u49)
    output_depth = layers.Activation('tanh', dtype='float32', name='last_tanh')(output_depth)
    model = Model([input_depth],[output_depth], name=netName)
    return model

# Depth ResBlk and Downsampler
class DepthBlock(tf.keras.Model):
    def __init__(self, filters, downSample, **kwargs):
        super(DepthBlock, self).__init__(**kwargs)

        self.conva = tf.keras.layers.Conv2D(filters, (1,1), padding='same')
        self.d = downSample
        self.bna = tfa.layers.InstanceNormalization(center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")
        self.bnb = tfa.layers.InstanceNormalization(center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")
        self.bnc = tfa.layers.InstanceNormalization(center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")
        self.convb = tf.keras.layers.Conv2D(filters, (3,3), padding='same')
        self.convc = tf.keras.layers.Conv2D(filters*2, (1,1), padding='same')
        self.conv_ds = tf.keras.layers.Conv2D(filters*2, (3,3), strides=(2,2),padding='same')
    
    def call(self, input_tensor, training=False):
        if self.d:
            ds = self.conv_ds(input_tensor)
            x = self.conva(ds)
        else:
            x = self.conva(input_tensor)
        x = tf.nn.elu(x)
        x = self.bna(x, training=training)
        x = self.convb(x)
        x = tf.nn.elu(x)
        x = self.bnb(x, training=training)
        x = self.convc(x)
#         x = self.bnc(x, training=training)
        if self.d:
            x += ds
        else:
            x += input_tensor
        return tf.nn.elu(x)

# Upsampler block
class UpsamplingBlock(tf.keras.Model):
    def __init__(self, filters, **kwargs):
        super(UpsamplingBlock, self).__init__(**kwargs)
        self.ups = layers.Conv2DTranspose(filters=filters, kernel_size=(3,3),strides=(2,2),padding='same')
        self.conva = layers.Conv2D(filters=filters, kernel_size=(3,3), padding='same')
        self.convb = layers.Conv2D(filters=filters, kernel_size=(3,3), padding='same')
        self.bna = tfa.layers.InstanceNormalization(center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")
        self.bna_u = tfa.layers.InstanceNormalization(center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")

    def call(self, input_tensor, training=False):
        x_u = self.ups(input_tensor)
        x_u = tf.nn.elu(x_u)
        x_u = self.bna_u(x_u,training=training)
        
        x = self.conva(x_u)
        x = tf.nn.elu(x)
        x = self.bna(x,training=training)
        
        x = self.convb(x)
        
        x +=x_u
        return tf.nn.elu(x)