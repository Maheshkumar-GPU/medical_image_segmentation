import tensorflow as tf
from tensorflow.keras.layers import(Conv2D,MaxPooling2D,Convolution2DTranspose,Input,concatenate )
from tensorflow.keras.models import Model

# convolutional block
def conv_block(input_tensor, num_filters):
    x = Conv2D(num_filters, (3, 3), padding='same',activation = "relu")(input_tensor)
    x = Conv2D(num_filters, (3, 3), padding='same', activation="relu")(x)
    return x
# encoder block
def encoder_block(input_tensor, num_filters):
    x = conv_block(input_tensor, num_filters)
    p = MaxPooling2D(pool_size=(2, 2))(x)
    return x,p
# decoder block
def decoder_block(input_tensor, num_filters, skip_tensor):
    x = Convolution2DTranspose(num_filters,(2,2),strides = 2,padding = 'same')(input_tensor)
    x = concatenate([x,skip_tensor])
    x = conv_block(x, num_filters)
    return x
# full u_net_model
def build_model(input_shape = (128,128,3)):
    inputs = Input(input_shape)
    # ENCODER
    s1,p1 = encoder_block(inputs,64)
    s2,p2 = encoder_block(p1,128)
    s3,p3 = encoder_block(p2,256)
    s4,p4 = encoder_block(p3,512)
    # BOTTLE NECK
    b1 = conv_block(p4,1024)
    # DECODER
    d1 = decoder_block(b1,512,s4)
    d2 = decoder_block(d1,256,s3)
    d3 = decoder_block(d2,128,s2)
    d4 = decoder_block(d3,64,s1)
    # output
    outputs = Conv2D(1, 1, padding = "same",activation='sigmoid')(d4)
    model = Model(inputs,outputs)
    return model
model = build_model()
model.summary()