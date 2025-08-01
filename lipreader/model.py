from keras.layers import Conv3D, ZeroPadding3D
from keras.layers import MaxPooling3D
from keras.layers import Dense, Activation, SpatialDropout3D, Flatten
from keras.layers import Bidirectional, TimeDistributed
from keras.layers import GRU
from keras.layers import BatchNormalization
from keras.layers import Input
from keras.models import Model
from lipreader.layers import CTC
from keras import backend as K
import numpy as np
import tensorflow as tf
from lipreader.common.constants import VIDEO_FRAME_NUM, NUM_PHONEMES, IMAGE_WIDTH, IMAGE_HEIGHT

class LipReader():
    def __init__(self, img_c=3, img_w=100, img_h=50, frames_n=VIDEO_FRAME_NUM, output_size=NUM_PHONEMES):
        #super().__init__()
        self.img_c = img_c
        self.img_w = img_w
        self.img_h = img_h
        self.frames_n = frames_n
        #self.absolute_max_string_len = absolute_max_string_len
        self.output_size = output_size
        
        if K.image_data_format() == 'channels_first':
            input_shape = (self.img_c, self.frames_n, self.img_w, self.img_h)
        else:
            # we are this one
            input_shape = (self.frames_n, IMAGE_WIDTH, IMAGE_HEIGHT, self.img_c)


        self.input_data = Input(name='the_input', shape=input_shape, dtype='float32')

        self.zero1 = ZeroPadding3D(padding=(1, 2, 2), name='zero1')(self.input_data)
        self.conv1 = Conv3D(32, (3, 5, 5), strides=(1, 2, 2), kernel_initializer='he_normal', name='conv1')(self.zero1)
        self.batc1 = BatchNormalization(name='batc1')(self.conv1)
        self.actv1 = Activation('relu', name='actv1')(self.batc1)
        self.drop1 = SpatialDropout3D(0.5)(self.actv1)
        self.maxp1 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max1')(self.drop1)

        self.zero2 = ZeroPadding3D(padding=(1, 2, 2), name='zero2')(self.maxp1)
        self.conv2 = Conv3D(64, (3, 5, 5), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv2')(self.zero2)
        self.batc2 = BatchNormalization(name='batc2')(self.conv2)
        self.actv2 = Activation('relu', name='actv2')(self.batc2)
        self.drop2 = SpatialDropout3D(0.5)(self.actv2)
        self.maxp2 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max2')(self.drop2)

        self.zero3 = ZeroPadding3D(padding=(1, 1, 1), name='zero3')(self.maxp2)
        self.conv3 = Conv3D(96, (3, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv3')(self.zero3)
        self.batc3 = BatchNormalization(name='batc3')(self.conv3)
        self.actv3 = Activation('relu', name='actv3')(self.batc3)
        self.drop3 = SpatialDropout3D(0.5)(self.actv3)
        self.maxp3 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max3')(self.drop3)

        # applies the flattening layer to each timestep
        self.resh1 = TimeDistributed(Flatten())(self.maxp3)

        # changed reset after to true
        self.gru_1 = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru1', reset_after=True), merge_mode='concat')(self.resh1)
        self.gru_2 = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru2', reset_after=True), merge_mode='concat')(self.gru_1)

        # transforms RNN output to character activations:
        self.dense1 = Dense(self.output_size, kernel_initializer='he_normal', name='dense1')(self.gru_2)
        #self.y_pred = self.dense1
        self.y_pred = Activation('softmax', name='softmax')(self.dense1)

        #self.labels = Input(name='the_labels', shape=[self.absolute_max_string_len], dtype='float32')
        #self.input_length = Input(name='input_length', shape=[1], dtype='int64')
        #self.label_length = Input(name='label_length', shape=[1], dtype='int64')

        #self.loss_out = CTC('ctc', [self.y_pred, self.labels, self.input_length, self.label_length])

        #self.model = Model(inputs=[self.input_data, self.labels, self.input_length, self.label_length], outputs=self.loss_out)
        self.model = Model(inputs=[self.input_data], outputs=self.y_pred)

    def summary(self):
        Model(inputs=self.input_data, outputs=self.y_pred).summary()

    def predict(self, input_batch):
        learning_phase = np.zeros((1,), dtype=np.int8)
        #print(input_batch)
        #return self.test_function()[0]  # the first 0 indicates test
        #print(type(self.input_data))
        out = K.function([self.input_data], [self.y_pred])
        return out(input_batch)[0]

    # @property
    # def test_function(self):
    #     # captures output of softmax so we can decode the output during visualization
    #     #print(self.input_data)
    #     return K.function([self.input_data, K.learning_phase()], [self.y_pred, K.learning_phase()])