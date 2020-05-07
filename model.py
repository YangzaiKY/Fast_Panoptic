from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Activation, Concatenate, Add, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, ZeroPadding2D, GlobalAveragePooling2D
from tensorflow.keras.utils import get_source_inputs, get_file
from tensorflow.keras import backend as K
from tensorflow.keras.applications.imagenet_utils import preprocess_input


class SepConvBN(tf.keras.layers.Layer):
	def __init__(self, 
				 filters, 
				 prefix, 
				 stride=1, 
				 kernel_size=3, 
				 rate=1, 
				 depth_activation=False, 
				 epsilon=1e-3, 
				 **kwargs):
		super().__init__(**kwargs)
		self.stride = stride
		self.depth_activation = depth_activation
		if self.stride == 1:
			depth_padding = 'same'
		else:
			kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
			pad_total = kernel_size_effective - 1
			pad_beg = pad_total // 2
			pad_end = pad_total - pad_beg
			self.zero_padding = ZeroPadding2D((pad_beg, pad_end))
			depth_padding = 'valid'

		self.activation = Activation(tf.nn.relu)
		self.depthwise_conv = DepthwiseConv2D((kernel_size, kernel_size), 
											  strides=(stride, stride), 
											  dilation_rate=(rate, rate), 
											  padding=depth_padding,
											  use_bias=False,
											  name=prefix+'_depthwise')
		self.depthwise_bn = BatchNormalization(name=prefix+'_depthwise_BN', epsilon=epsilon)
		self.pointwise_conv = Conv2D(filters, 
									 (1, 1),
									 padding='same',
									 use_bias=False,
									 name=prefix+'_pointwise')
		self.pointwise_bn = BatchNormalization(name=prefix+'_pointwise_BN', epsilon=epsilon)

	def call(self, inputs, training=False):
		x = inputs
		if self.stride != 1:
			x = self.zero_padding(x)
		if not self.depth_activation:
			x = self.activation(x)
		x = self.depthwise_conv(x)
		x = self.depthwise_bn(x)
		if self.depth_activation:
			x = self.activation(x)
		x = self.pointwise_conv(x)
		x = self.pointwise_bn(x)
		if self.depth_activation:
			x = self.activation(x)
		return x


def _make_divisible(v, divisor, min_value=None):
	if min_value is None:
		min_value = divisor
	new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
	if new_v < 0.9 * v:
		new_v += divisor
	return new_v


class InvertedResBlock(tf.keras.layers.Layer):
	def __init__(self,
				 expansion,
				 stride, 
				 alpha,
				 filters,
				 block_id,
				 skip_connection,
				 rate=1,
				 **kwargs):
		super().__init__(**kwargs)
		pointwise_conv_filters = int(filters * alpha)
		pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
		self.block_id = block_id
		self.skip_connection = skip_connection
		self.expansion = expansion
		if self.block_id:
			self.prefix = 'expanded_conv_{}_'.format(self.block_id)
			# expand_conv = Conv2D(self.expansion * in_channels,
			#					 kernel_size=1, 
			#					 padding='same', 
			#					 use_bias=False, 
			#					 activation=None,
			#					 name=self.prefix+'expand')
			self.expand_bn = BatchNormalization(epsilon=1e-3, 
											    momentum=0.999,
											    name=self.prefix+'expand_BN')
			self.expand_activation = Activation(tf.nn.relu6, name=self.prefix+'expand_relu')
		else:
			self.prefix = 'expanded_conv_'
		self.depthwise_conv = DepthwiseConv2D(kernel_size=3,
										 	  strides=stride,
										 	  activation=None,
										 	  use_bias=False,
										 	  padding='same',
										 	  dilation_rate=(rate, rate),
										 	  name=self.prefix+'depthwise')
		self.depthwise_bn = BatchNormalization(epsilon=1e-3,
										  	   momentum=0.999,
										  	   name=self.prefix+'depthwise_BN')
		self.depthwise_activation = Activation(tf.nn.relu6, name=self.prefix+'depthwise_relu')

		self.project_conv = Conv2D(pointwise_filters,
								   kernel_size=1,
								   padding='same',
								   use_bias=False,
								   activation=None,
								   name=self.prefix+'project')
		self.project_bn = BatchNormalization(epsilon=1e-3,
											 momentum=0.999,
											 name=self.prefix+'project_BN')
		self.add = Add(name=self.prefix+'add')


	def call(self, inputs, training=False):
		x = inputs
		in_channels = x.shape[-1].value
		if self.block_id:
			x = Conv2D(self.expansion * in_channels,
					   kernel_size=1, 
					   padding='same', 
					   use_bias=False, 
					   activation=None,
					   name=self.prefix+'expand')(x)
			x = self.expand_bn(x)
			x = self.expand_activation(x)
		x = self.depthwise_conv(x)
		x = self.depthwise_bn(x)
		x = self.depthwise_activation(x)

		x = self.project_conv(x)
		x = self.project_bn(x)

		if self.skip_connection:
			return self.add([inputs, x])
		return x


class MobileNetV2(tf.keras.layers.Layer):
	def __init__(self,
				 alpha,
				 **kwargs):
		super().__init__(**kwargs)
		first_block_filters = _make_divisible(32 * alpha, 8)
		self.conv0 = Conv2D(first_block_filters,
						    kernel_size=3,
						    strides=(2, 2),
						    padding='same',
						    use_bias=False,
						    name='Conv')
		self.bn0 = BatchNormalization(epsilon=1e-3,
								 	  momentum=0.999,
								 	  name='Conv_BN')
		self.ac0 = Activation(tf.nn.relu6, name='Conv_Relu6')

		self.inverted_res_block0 = InvertedResBlock(filters=16,
											        alpha=alpha,
											        stride=1,
											        expansion=1,
											        block_id=0,
											        skip_connection=False)

		self.inverted_res_block1 = InvertedResBlock(filters=24,
											        alpha=alpha,
											        stride=2,
											        expansion=6,
											        block_id=1,
											        skip_connection=False)
		self.inverted_res_block2 = InvertedResBlock(filters=24,
											        alpha=alpha,
											        stride=2,
											        expansion=6,
											        block_id=2,
											        skip_connection=True)

		self.inverted_res_block3 = InvertedResBlock(filters=32,
											        alpha=alpha,
											        stride=2,
											        expansion=6,
											        block_id=3,
											        skip_connection=False)
		self.inverted_res_block4 = InvertedResBlock(filters=32,
											        alpha=alpha,
											        stride=1,
											        expansion=6,
											        block_id=4,
											        skip_connection=True)
		self.inverted_res_block5 = InvertedResBlock(filters=32,
											        alpha=alpha,
											        stride=1,
											        expansion=6,
											        block_id=5,
											        skip_connection=True)

		self.inverted_res_block6 = InvertedResBlock(filters=64,
											        alpha=alpha,
											        stride=1,
											        expansion=6,
											        block_id=6,
											        skip_connection=False)
		self.inverted_res_block7 = InvertedResBlock(filters=64,
											        alpha=alpha,
											        stride=1,
											        rate=2,
											        expansion=6,
											        block_id=7,
											        skip_connection=True)
		self.inverted_res_block8 = InvertedResBlock(filters=64,
											        alpha=alpha,
											        stride=1,
											        rate=2,
											        expansion=6,
											        block_id=8,
											        skip_connection=True)
		self.inverted_res_block9 = InvertedResBlock(filters=64,
											        alpha=alpha,
											        stride=1,
											        rate=2,
											        expansion=6,
											        block_id=9,
											        skip_connection=True)

		self.inverted_res_block10 = InvertedResBlock(filters=96,
											         alpha=alpha,
											         stride=1,
											         rate=2,
											         expansion=6,
											         block_id=10,
											         skip_connection=False)
		self.inverted_res_block11 = InvertedResBlock(filters=96,
											         alpha=alpha,
											         stride=1,
											         rate=2,
											         expansion=6,
											         block_id=11,
											         skip_connection=True)
		self.inverted_res_block12 = InvertedResBlock(filters=96,
											         alpha=alpha,
											         stride=1,
											         rate=2,
											         expansion=6,
											         block_id=12,
											         skip_connection=True)

		self.inverted_res_block13 = InvertedResBlock(filters=160,
											    	 alpha=alpha,
											    	 stride=1,
											    	 rate=2,
											    	 expansion=6,
											    	 block_id=13,
											    	 skip_connection=False)
		self.inverted_res_block14 = InvertedResBlock(filters=160,
											    	 alpha=alpha,
											    	 stride=1,
											    	 rate=4,
											    	 expansion=6,
											    	 block_id=14,
											    	 skip_connection=True)
		self.inverted_res_block15 = InvertedResBlock(filters=160,
											    	 alpha=alpha,
											    	 stride=1,
											    	 rate=4,
											    	 expansion=6,
											    	 block_id=15,
											    	 skip_connection=True)

		self.inverted_res_block16 = InvertedResBlock(filters=320,
													 alpha=alpha,
													 stride=1,
													 rate=4,
													 expansion=6,
													 block_id=16,
													 skip_connection=False)

	def call(self, inputs, training=False):
		x = inputs
		x = self.conv0(x)
		x = self.bn0(x)
		x = self.ac0(x)

		x = self.inverted_res_block0(x)
		x = self.inverted_res_block1(x)
		x = self.inverted_res_block2(x)
		x = self.inverted_res_block3(x)
		x = self.inverted_res_block4(x)
		x = self.inverted_res_block5(x)
		x = self.inverted_res_block6(x)
		x = self.inverted_res_block7(x)
		x = self.inverted_res_block8(x)
		x = self.inverted_res_block9(x)
		x = self.inverted_res_block10(x)
		x = self.inverted_res_block11(x)
		x = self.inverted_res_block12(x)
		x = self.inverted_res_block13(x)
		x = self.inverted_res_block14(x)
		x = self.inverted_res_block15(x)
		x = self.inverted_res_block16(x)

		return x


class ASPP(tf.keras.layers.Layer):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def call(self, inputs, training=False):
		shape_before = tf.shape(inputs)
		x = inputs
		x = GlobalAveragePooling2D()(x)
		x = Lambda(lambda v: K.expand_dims(v, 1))(x)
		x = Lambda(lambda v: K.expand_dims(v, 1))(x)
		x = Conv2D(256,
				   (1, 1),
				   padding='same',
				   use_bias=False,
				   name='image_pooling')(x)
		x = BatchNormalization(name='image_pooling_BN',
							   epsilon=1e-5)(x)
		x = Activation(tf.nn.relu)(x)

		size_before = tf.keras.backend.int_shape(inputs)
		x = Lambda(lambda x: tf.compat.v1.image.resize(x, size_before[1:3],
													   method='bilinear', align_corners=True))(x)

		x1 = Conv2D(256,
					(1, 1),
					padding='same',
					use_bias=False,
					name='aspp0')(inputs)
		x1 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(x1)
		x1 = Activation(tf.nn.relu, name='aspp0_activation')(x1)

		return Concatenate()([x, x1])



class DeeplabV3(Model):
	def __init__(self,
				 weights='pascal_voc',
				 input_tensor=None,
				 input_shape=(512, 512, 3),
				 classes=21,
				 backbone='mobilenetv2',
				 OS=16,
				 alpha=1.,
				 activation=None,
				 **kwargs):
		super().__init__(**kwargs)
		if backbone == 'mobilenetv2':
			self.backbone = MobileNetV2(alpha=alpha)
		self.aspp = ASPP()
		self.activation = activation

		if input_tensor is None:
			self.img_input = Input(shape=input_shape)
		else:
			self.img_input = input_tensor

		if input_tensor is not None:
			self.inputs = get_source_inputs(input_tensor)
		else:
			self.inputs = self.img_input

		if (weights == 'pascal_voc' and classes == 21) or (weights == 'cityscapes' and classes == 19):
			self.last_layer_name = 'logits_semantic'
		else:
			self.last_layer_name = 'custom_logits_semantic'


	def call(self, inputs, training=False):
		x = inputs
		x = self.backbone(x)
		x = self.aspp(x)
		x = Conv2D(256,
				   (1, 1),
				   padding='same',
				   use_bias=False,
				   name='concat_projectioin')(x)
		x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
		x = Activation(tf.nn.relu)(x)
		x = Dropout(0.1)(x)

		x = Conv2D(classes,
				   (1, 1),
				   padding='same',
				   name=self.last_layer_name)(x)
		size_before = tf.keras.backend.int_shape(img_input)
		x = Lambda(lambda v: tf.compat.v1.image.resize(v, size_before[1:3],
													   method='bilinear', align_corners=True))(x)

		if activation in {'softmax', 'sigmoid'}:
			x = Activation(activation)(x)
		return x


if __name__ == '__main__':
	model = DeeplabV3()
	model.build(input_shape=(None, 300, 300, 3))
	model.summary()
