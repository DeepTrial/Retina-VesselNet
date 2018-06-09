import keras
import theano.tensor as T
from keras.models import Model
from keras.layers.merge import add
from keras.layers import Input, Conv2D,ZeroPadding2D,Conv2DTranspose, MaxPooling2D, UpSampling2D,Cropping2D, core, Dropout,normalization,concatenate,Activation
from keras import backend as K
from keras.layers.core import Layer, InputSpec
from matplotlib import pyplot as plt
import configparser
from keras.layers.advanced_activations import LeakyReLU
from keras.constraints import maxnorm
config = configparser.RawConfigParser()
config.read('./configuration.txt')
num_lesion_class = int(config.get('data attributes', 'num_lesion_class'))

smooth=1.0

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

class LossHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get(fbeta_score))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get(fbeta_score))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure(figsize=(8, 6))
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()




def get_unet(n_ch,patch_height,patch_width):
    inputs = Input((n_ch,patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), padding='same')(inputs)#'valid'
    conv1 = LeakyReLU(alpha=0.3)(conv1)
    conv1 = Dropout(0.2)(conv1)
    conv1 = normalization.BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='RandomNormal', gamma_initializer='one')(conv1)
    conv1 = Conv2D(32, (3, 3), dilation_rate=2, padding='same')(conv1)
    conv1 = LeakyReLU(alpha=0.3)(conv1)
    conv1 = Conv2D(32, (3, 3), dilation_rate=4, padding='same')(conv1)
    conv1 = LeakyReLU(alpha=0.3)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)


    #pool1 = normalization.BatchNormalization(epsilon=1e-06, mode=1, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(pool1)
    conv2 = Conv2D(64, (3, 3), padding='same')(pool1) #,activation='relu', padding='same')(pool1)
    conv2 = normalization.BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='RandomNormal', gamma_initializer='one')(conv2)
    conv2 =LeakyReLU(alpha=0.3)(conv2)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), dilation_rate=2, padding='same')(conv2)
    conv2 = LeakyReLU(alpha=0.3)(conv2)
    conv2 = Conv2D(64, (3, 3), dilation_rate=4, padding='same')(conv2)#,W_regularizer=l2(0.01), b_regularizer=l2(0.01))(conv2)
    conv2 = LeakyReLU(alpha=0.3)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)


    #crop = Cropping2D(cropping=((int(3 * patch_height / 8), int(3 * patch_height / 8)), (int(3 * patch_width / 8), int(3 * patch_width / 8))))(conv1)
    #conv3 = concatenate([crop,pool2], axis=1)
    conv3 = Conv2D(128, (3, 3), padding='same')(pool2)   #, activation='relu', padding='same')(conv3)
    conv3 = normalization.BatchNormalization(epsilon=2e-05,axis=1, momentum=0.9, weights=None, beta_initializer='RandomNormal', gamma_initializer='one')(conv3)
    conv3 = LeakyReLU(alpha=0.3)(conv3)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), dilation_rate=2, padding='same')(conv3)#,W_regularizer=l2(0.01), b_regularizer=l2(0.01))(conv3)
    conv3 = normalization.BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None,beta_initializer='RandomNormal', gamma_initializer='one')(conv3)
    conv3 = LeakyReLU(alpha=0.3)(conv3)

    conv3 = Conv2D(128, (3, 3), dilation_rate=4, padding='same')(conv3)
    conv3 = normalization.BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None,beta_initializer='RandomNormal', gamma_initializer='one')(conv3)
    conv3 = LeakyReLU(alpha=0.3)(conv3)


    #up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=1)
    conv4 = Conv2D(64, (3, 3), padding='same')(up1)
    conv4 = LeakyReLU(alpha=0.3)(conv4)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), padding='same')(conv4)
    conv4 = LeakyReLU(alpha=0.3)(conv4)
    #conv4 = normalization.BatchNormalization(epsilon=1e-06, mode=1, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(conv4)
    #
    #up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=1)
    conv5 = Conv2D(32, (3, 3), padding='same')(up2)
    conv5 = LeakyReLU(alpha=0.3)(conv5)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), padding='same')(conv5)
    conv5 = LeakyReLU(alpha=0.3)(conv5)

    conv6 = Conv2D(num_lesion_class+1, (1, 1),padding='same')(conv5)
    conv6 = LeakyReLU(alpha=0.3)(conv6)
    #conv6 = normalization.BatchNormalization(epsilon=1e-06, mode=1, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(conv6)

    #for tensorflow
    #conv6 = core.Reshape((patch_height*patch_width,num_lesion_class+1))(conv6)
    #for theano
    conv6 = core.Reshape(((num_lesion_class+1),patch_height*patch_width))(conv6)
    conv6 = core.Permute((2,1))(conv6)
    ############
    act = Activation('softmax')(conv6)

    model = Model(inputs=inputs, outputs=act)
    return model

def ResBlock(inputs,outdim):
    inputshape=K.int_shape(inputs)
    conv1=Conv2D(outdim, (1, 1), activation='relu', padding='same')(inputs)
    conv2 = Conv2D(outdim, (3, 3), activation='relu', padding='same')(conv1)
    conv3 = Conv2D(outdim, (1, 1), padding='same')(conv2)

    if inputshape[1]!=outdim:
        shortcut=Conv2D(outdim, (1, 1), padding='same')(inputs)
    else:
        shortcut=inputs
    result=add([conv3,shortcut])
    result=Activation('relu')(result)
    return result

def DenseBlock(inputs,outdim):

    inputshape = K.int_shape(inputs)
    bn=normalization.BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='zero',gamma_initializer='one')(inputs)
    act=Activation('relu')(bn)
    conv1 = Conv2D(outdim, (3, 3), activation=None, padding='same')(act)

    if inputshape[1]!=outdim:
        shortcut=Conv2D(outdim, (1, 1), padding='same')(inputs)
    else:
        shortcut=inputs
    result1=add([conv1,shortcut])

    bn = normalization.BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='zero',gamma_initializer='one')(result1)
    act = Activation('relu')(bn)
    conv2 = Conv2D(outdim, (3, 3), activation=None, padding='same')(act)
    result=add([result1,conv2,shortcut])
    result=Activation('relu')(result)
    return result


def R_Unet(n_ch,patch_height,patch_width):
    inputs = Input((n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (1, 1), activation=None, padding='same')(inputs)
    conv1 =normalization.BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(conv1)
    conv1 = Activation('relu')(conv1)

    conv1=DenseBlock(conv1,32) #48
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2=DenseBlock(pool1,64)#24
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 =DenseBlock(pool2,64)#12
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)


    conv4 = DenseBlock(pool3, 64)  # 12


    up1=Conv2DTranspose(64,(3,3),strides=2,activation='relu',padding='same')(conv4)
    up1 = concatenate([up1, conv3], axis=1)

    conv5=DenseBlock(up1,64)

    up2 = Conv2DTranspose(64,(3,3),strides=2,activation='relu',padding='same')(conv5)
    up2 = concatenate([up2, conv2], axis=1)

    conv6=DenseBlock(up2,64)

    up3 = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(conv6)
    up3 = concatenate([up3, conv1], axis=1)

    conv7=DenseBlock(up3,32)

    conv8 = Conv2D(num_lesion_class + 1, (1, 1), activation='relu', padding='same')(conv7)
    # conv6 = normalization.BatchNormalization(epsilon=1e-06, mode=1, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(conv6)

    # for tensorflow
    # conv6 = core.Reshape((patch_height*patch_width,num_lesion_class+1))(conv6)
    # for theano
    conv8 = core.Reshape(((num_lesion_class + 1), patch_height * patch_width))(conv8)
    conv8 = core.Permute((2, 1))(conv8)
    ############
    act = Activation('softmax')(conv8)

    model = Model(inputs=inputs, outputs=act)
    return model



