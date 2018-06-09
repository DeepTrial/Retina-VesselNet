import keras,os
from keras.models import Model
from keras.layers.merge import add,multiply
from keras.layers import Lambda,Input, Conv2D,Conv2DTranspose, MaxPooling2D, UpSampling2D,Cropping2D, core, Dropout,BatchNormalization,concatenate,Activation
from keras import backend as K
from keras.layers.core import Layer, InputSpec
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import plot_model

from perception.bases.model_base import ModelBase

class SegmentionModel(ModelBase):
	def __init__(self,config=None):
		super(SegmentionModel, self).__init__(config)

		self.patch_height=config.patch_height
		self.patch_width = config.patch_width
		self.num_seg_class=config.seg_num

		self.build_model()
		self.save()

	def expend_as(self,tensor, rep):
		my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)
		return my_repeat

	def AttnGatingBlock(self,x, g, inter_shape):
		shape_x = K.int_shape(x)  # 32
		shape_g = K.int_shape(g)  # 16

		theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
		shape_theta_x = K.int_shape(theta_x)

		phi_g = Conv2D(inter_shape, (1, 1), padding='same')(g)
		upsample_g = Conv2DTranspose(inter_shape, (3, 3),strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),padding='same')(phi_g)  # 16

		concat_xg = add([upsample_g, theta_x])
		act_xg = Activation('relu')(concat_xg)
		psi = Conv2D(1, (1, 1), padding='same')(act_xg)
		sigmoid_xg = Activation('sigmoid')(psi)
		shape_sigmoid = K.int_shape(sigmoid_xg)
		upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

		# my_repeat=Lambda(lambda xinput:K.repeat_elements(xinput[0],shape_x[1],axis=1))
		# upsample_psi=my_repeat([upsample_psi])
		upsample_psi = self.expend_as(upsample_psi, shape_x[3])

		y = multiply([upsample_psi, x])

		# print(K.is_keras_tensor(upsample_psi))

		result = Conv2D(shape_x[3], (1, 1), padding='same')(y)
		result_bn = BatchNormalization()(result)
		return result_bn

	def UnetGatingSignal(self,input, is_batchnorm=False):
		shape = K.int_shape(input)
		x = Conv2D(shape[3] * 2, (1, 1), strides=(1, 1), padding="same")(input)
		if is_batchnorm:
			x = BatchNormalization()(x)
		x = Activation('relu')(x)
		return x

	def UnetConv2D(self,input, outdim, is_batchnorm=False):
		shape = K.int_shape(input)
		x = Conv2D(outdim, (3, 3), strides=(1, 1), padding="same")(input)
		if is_batchnorm:
			x =BatchNormalization()(x)
		x = Activation('relu')(x)

		x = Conv2D(outdim, (3, 3), strides=(1, 1), padding="same")(x)
		if is_batchnorm:
			x = BatchNormalization()(x)
		x = Activation('relu')(x)
		return x

	def build_model(self):
		inputs = Input((self.patch_height, self.patch_width,1))
		conv = Conv2D(16, (3, 3), padding='same')(inputs)  # 'valid'
		conv = LeakyReLU(alpha=0.3)(conv)

		conv1 = self.UnetConv2D(conv, 32, is_batchnorm=True)  # 32 128
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

		conv2 = self.UnetConv2D(pool1, 32, is_batchnorm=True)  # 32 64
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

		conv3 = self.UnetConv2D(pool2, 64, is_batchnorm=True)  # 64 32
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

		conv4 = self.UnetConv2D(pool3, 64, is_batchnorm=True)  # 64 16
		pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

		center = self.UnetConv2D(pool4, 128, is_batchnorm=True)  # 128 8
		gating = self.UnetGatingSignal(center, is_batchnorm=True)

		attn_1 = self.AttnGatingBlock(conv4, gating, 128)
		up1 = concatenate([Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(center), attn_1], axis=3)

		attn_2 = self.AttnGatingBlock(conv3, gating, 64)
		up2 = concatenate([Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(up1), attn_2], axis=3)

		attn_3 = self.AttnGatingBlock(conv2, gating, 32)
		up3 = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(up2), attn_3], axis=3)

		up4 = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(up3), conv1], axis=3)

		conv8 = Conv2D(self.num_seg_class + 1, (1, 1), activation='relu', padding='same')(up4)
		conv8 = core.Reshape((self.patch_height * self.patch_width,(self.num_seg_class + 1)))(conv8)
		############
		act = Activation('softmax')(conv8)

		model = Model(inputs=inputs, outputs=act)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
		plot_model(model, to_file=os.path.join(self.config.checkpoint, "model.png"), show_shapes=True)
		self.model = model