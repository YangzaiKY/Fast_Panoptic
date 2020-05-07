import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, Add


def l2_regularizer(weight=0.00004 * 0.5):
	return tf.keras.regularizers.l2(weight)


def truncated_normal_initializer(mean=0.0, stddev=0.03):
	return tf.keras.initializers.TruncatedNormal(mean=mean, stddev=stddev)


def _make_divisible(v, divisor, min_value=None):
	if min_value is None:
		min_value = divisor
	new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
	if new_v < 0.9 * v:
		new_v += divisor
	return new_v


class Conv2DBlock(tf.keras.layers.Layer):
	def __init__(self,
				 filters,
				 kernel_size,
				 strides,
				 padding,
				 use_bias,
				 kernel_initializer=truncated_normal_initializer(stddev=0.09),
				 kernel_regularizer=l2_regularizer(),
				 ac=True,
				 bn=True,
				 momentum=0.997,
				 **kwargs):
		super().__init__(**kwargs)
		self.batch_normalization = None
		self.activation = None

		self.conv = Conv2D(filters=filters,
						   kernel_size=kernel_size,
						   strides=strides,
						   padding=padding,
						   use_bias=use_bias,
						   activation=None,
						   kernel_initializer=kernel_initializer,
						   kernel_regularizer=kernel_regularizer)

		if bn:
			self.batch_normalization = BatchNormalization(momentum=momentum)
		if ac:
			self.activation = ReLU(max_value=6)

	def call(self, inputs, training=False):
		x = self.conv(inputs)
		if self.batch_normalization is not None:
			x = self.batch_normalization(x, training=training)
		if self.activation is not None:
			x = self.activation(x)
		return x


class DepthwiseConv2DBN(tf.keras.layers.Layer):
	def __init__(self,
				 kernel_size,
				 strides,
				 padding,
				 use_bias,
				 depth_multiplier=1,
				 depthwise_initializer=truncated_normal_initializer(stddev=0.09),
				 momentum=0.997,
				 ac=True,
				 bn=True,
				 **kwargs):
		super().__init__(**kwargs)
		self.batch_normalization = None
		self.activation = None

		self.depthwise_conv = DepthwiseConv2D(kernel_size=kernel_size,
											  strides=strides,
											  padding=padding,
											  use_bias=use_bias,
											  depth_multiplier=depth_multiplier,
											  depthwise_initializer=depthwise_initializer)

		if bn:
			self.batch_normalization = BatchNormalization(momentum=momentum)
		if ac:
			self.activation = ReLU(max_value=6)

	def call(self, inputs, training=False):
		x = self.depthwise_conv(inputs)
		if self.batch_normalization is not None:
			x = self.batch_normalization(x)
		if self.activation is not None:
			x = self.activation(x)
		return x


class MobileNetV2(Model):
	def __init__(self,
				 alpha=1.0,
				 classes=1000,
				 **kwargs):
		super().__init__(**kwargs)
		with tf.name_scope('MobileNet') as scope:
			first_block_filters = _make_divisible(32 * alpha, 8)
			self.conv2d0 = Conv2DBlock(filters=first_block_filters,
									   kernel_size=3,
									   strides=2,
									   padding='same',
									   use_bias=False)
			self.inverted_res_group0 = InvertedResGroup(filters=16,
														in_channel=first_block_filters,
														strides=1,
														repeat=1,
														alpha=alpha,
														expansion=1)
			self.inverted_res_group1 = InvertedResGroup(filters=24,
														in_channel=16,
														strides=2,
														repeat=2,
														alpha=alpha)
			self.inverted_res_group2 = InvertedResGroup(filters=32,
														in_channel=24,
														strides=2,
														repeat=3,
														alpha=alpha)
			self.inverted_res_group3 = InvertedResGroup(filters=64,
														in_channel=32,
														strides=2,
														repeat=4,
														alpha=alpha)
			self.inverted_res_group4 = InvertedResGroup(filters=96,
														in_channel=64,
														strides=1,
														repeat=3,
														alpha=alpha)

			self.conv2d1 = Conv2DBlock(filters=6 * 96,
									   kernel_size=1,
									   strides=1,
									   padding='same',
									   use_bias=False)

			self.inverted_res_group5 = InvertedResGroup(filters=160,
														in_channel=96,
														strides=2,
														repeat=1,
														alpha=alpha,
														expansion=1)			
			self.inverted_res_group6 = InvertedResGroup(filters=160,
														in_channel=160,
														strides=1,
														repeat=2,
														alpha=alpha)
			self.inverted_res_group7 = InvertedResGroup(filters=320,
														in_channel=160,
														strides=1,
														repeat=1,
														alpha=alpha)
			last_block_filters = 1280
			if alpha > 1.0:
				last_block_filters = _make_divisible(1280 * alpha, 8)

			self.conv2d2 = Conv2DBlock(filters=last_block_filters,
									   kernel_size=1,
									   strides=1,
									   use_bias=False,
									   padding='same')

	def call(self, inputs, training=None, mask=None):
		x = inputs

		x = self.conv2d0(x, training=training)
		x = self.inverted_res_group0(x, training=training)
		x = self.inverted_res_group1(x, training=training)
		x = self.inverted_res_group2(x, training=training)
		x = self.inverted_res_group3(x, training=training)
		x = self.inverted_res_group4(x, training=training)

		branch1 = self.conv2d1(x, training=training)

		x = self.inverted_res_group5(x, training=training)
		x = self.inverted_res_group6(x, training=training)
		x = self.inverted_res_group7(x, training=training)

		branch2 = self.conv2d2(x, training=training)

		return branch1, branch2
		
class InvertedResGroup(tf.keras.layers.Layer):
	def __init__(self,
				 filters,
				 in_channel,
				 strides,
				 repeat,
				 alpha,
				 expansion=6,
				 **kwargs):
		super().__init__(**kwargs)
		self.ops = list()

		base_inverted = InvertedResBlock(filters=filters,
										 in_channel=in_channel,
										 expansion=expansion,
										 strides=strides,
										 alpha=alpha)
		self.ops.append(base_inverted)

		for i in range(repeat - 1):
			inverted = InvertedResBlock(filters=filters,
										in_channel=in_channel,
										expansion=expansion,
										strides=1,
										alpha=alpha)
			self.ops.append(inverted)

	def call(self, inputs, training=False):
		x = inputs
		for op in self.ops:
			x = op(x)
		return x


class InvertedResBlock(tf.keras.layers.Layer):
	def __init__(self,
				 filters,
				 in_channel,
				 expansion,
				 strides,
				 alpha,
				 **kwargs):
		super().__init__(**kwargs)
		with tf.name_scope('InvertedResBlock'):
			pointwise_filters = int(filters * alpha)
			pointwise_filters = _make_divisible(pointwise_filters, 8)

			self.ops = list()
			if expansion > 1:
				convblock0 = Conv2DBlock(filters=expansion * in_channel,
										 kernel_size=1,
										 strides=1,
										 padding='same',
										 use_bias=False)
				self.ops.append(convblock0)

			depthwise0 = DepthwiseConv2DBN(kernel_size=3,
										 strides=strides,
										 padding='same',
										 use_bias=False)

			self.ops.append(depthwise0)

			convblock1 = Conv2DBlock(filters=pointwise_filters,
									 kernel_size=1,
									 strides=1,
									 padding='same',
									 use_bias=False,
									 ac=False)
			self.ops.append(convblock1)

			self.add_op = None
			if (in_channel == pointwise_filters) and (strides == 1):
				self.add_op = Add(name='add')

	def call(self, inputs, training=False):
		x = inputs
		for op in self.ops:
			x = op(x, training)

		if self.add_op is not None:
			x = self.add_op([inputs, x])
		return x


if __name__ == '__main__':
	model = MobileNetV2()
	model.build(input_shape=(None, 300, 300, 3))
	model.summary()

