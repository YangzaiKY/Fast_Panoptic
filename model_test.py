from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Activation, Concatenate, Add, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, ZeroPadding2D, GlobalAveragePooling2D, ReLU
from tensorflow.keras.utils import get_source_inputs, get_file
from tensorflow.keras import backend as K
from tensorflow.keras.applications.imagenet_utils import preprocess_input


##########################################################################################################
# common
def l2_regularizer(weight = 0.00004 * 0.5):
    return tf.keras.regularizers.l2(weight)

def truncated_normal_initializer(mean = 0.0, stddev=0.03):
    return tf.keras.initializers.TruncatedNormal(mean = mean, stddev=stddev)

def constant_initializer():
    return tf.keras.initializers.constant()

def non_max_suppression_with_scores(bboxes,
                        scores,
                        max_output_size,
                        iou_threshold=0.6,
                        score_threshold=float("-inf")):
    selected_indices, selected_scores = \
        tf.image.non_max_suppression_with_scores(bboxes, scores,
                                                 max_output_size, iou_threshold,
                                                 score_threshold)

    return selected_indices, selected_scores

class Conv2DBlock(tf.keras.layers.Layer):

    def __init__(self,
                 filters,
                 kernel_size,
                 strides,
                 padding,
                 use_bias,
                 kernel_initializer = truncated_normal_initializer(stddev=0.09),
                 kernel_regularizer = l2_regularizer(),
                 ac = True,
                 bn = True,
                 momentum = 0.997,
                 **kwargs):
        super().__init__(**kwargs)

        self.batch_normalization = None
        self.activation = None

        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            activation=None,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer
        )

        if bn:
            self.batch_normalization = tf.keras.layers.BatchNormalization(
                momentum=momentum
            )

        if ac:
            self.activation = tf.keras.layers.ReLU(max_value=6)

    def call(self, inputs, training = False):
        outputs = self.conv(inputs)

        if self.batch_normalization is not None:
            outputs = self.batch_normalization(outputs, training = training)

        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs


class DepthwiseConv2DBN(tf.keras.layers.Layer):

    def __init__(self,
                 kernel_size,
                 strides,
                 padding,
                 use_bias,
                 depth_multiplier = 1,
                 depthwise_initializer = truncated_normal_initializer(stddev=0.09),
                 momentum = 0.997,
                 ac = True,
                 bn = True,
                 **kwargs):
        super().__init__(**kwargs)

        self.batch_normalization = None
        self.activation = None

        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            depth_multiplier=depth_multiplier,
            activation=None,
            depthwise_initializer=depthwise_initializer
        )

        if bn:
            self.batch_normalization = tf.keras.layers.BatchNormalization(
                momentum=momentum
            )

        if ac:
            self.activation = tf.keras.layers.ReLU(max_value=6)

    def call(self, inputs, training = False):
        outputs = self.depthwise_conv(inputs)

        if self.batch_normalization is not None:
            outputs = self.batch_normalization(outputs)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation(tf.nn.relu)(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation(tf.nn.relu)(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation(tf.nn.relu)(x)

    return x

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
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
            #                    kernel_size=1, 
            #                    padding='same', 
            #                    use_bias=False, 
            #                    activation=None,
            #                    name=self.prefix+'expand')
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
        in_channels = x.shape[-1]
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


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):
    in_channels = inputs.shape[-1]  # inputs._keras_shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'expanded_conv_{}_'.format(block_id)
    if block_id:
        # Expand

        x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                   use_bias=False, activation=None,
                   name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'expand_BN')(x)
        x = Activation(tf.nn.relu6, name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'
    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                        use_bias=False, padding='same', dilation_rate=(rate, rate),
                        name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'depthwise_BN')(x)

    x = Activation(tf.nn.relu6, name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(pointwise_filters,
               kernel_size=1, padding='same', use_bias=False, activation=None,
               name=prefix + 'project')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'project_BN')(x)

    if skip_connection:
        return Add(name=prefix + 'add')([inputs, x])

    # if in_channels == pointwise_filters and stride == 1:
    #    return Add(name='res_connect_' + str(block_id))([inputs, x])

    return x


def _conv_2d_block(inputs, filters, kernel_size, strides, padding, use_bias, momentum=0.997):
	x = inputs
	x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias)(x)
	x = BatchNormalization(momentum=momentum)(x)
	x = ReLU(max_value=6)(x)
	return x


###################################################################################################################
# anchor_generator
def create_scales_and_ratios(num_layers = 6,
                             min_scale=0.2,
                             max_scale=0.95,
                             aspect_ratios=[1.0, 2.0, 0.5, 3.0, 0.3333]
                             ):
    scales = [min_scale + (max_scale - min_scale) * i / (num_layers - 1)
              for i in range(num_layers)] + [1.0]

    total_scales = list()
    total_ratios = list()

    interp_scale_aspect_ratio = 1.0
    for layer, scale, scale_next in zip(range(num_layers), scales[:-1], scales[1:]):
        layer_scales = list()
        layer_ratios = list()

        if layer == 0:
            layer_scales = [0.1, scale, scale]
            layer_ratios = [1.0, 2.0, 0.5]
        else:
            for aspect_ratio in aspect_ratios:
                layer_scales.append(scale)
                layer_ratios.append(aspect_ratio)
            layer_scales.append(tf.sqrt(scale * scale_next))
            layer_ratios.append(interp_scale_aspect_ratio)

        total_scales.append(layer_scales)
        total_ratios.append(layer_ratios)

    return total_scales, total_ratios

def tile_anchors(grid_width,
                  grid_height,
                  scales,
                  aspect_ratios,
                  anchor_stride,
                  anchor_offset,
                  base_anchor_size=[1, 1]):
        '''
        '''

        ratio_sqrts = tf.sqrt(aspect_ratios)
        widths = scales * ratio_sqrts * base_anchor_size[0]
        heights = scales / ratio_sqrts * base_anchor_size[1]

        x_centers = tf.range(grid_width, dtype=tf.float32)
        x_centers = x_centers * anchor_stride[0] + anchor_offset[0]
        y_centers = tf.range(grid_height, dtype=tf.float32)
        y_centers = y_centers * anchor_stride[1] + anchor_offset[1]
        x_centers, y_centers = tf.meshgrid(x_centers, y_centers)

        width_grid, x_center_grid = tf.meshgrid(widths, x_centers)
        height_grid, y_center_grid = tf.meshgrid(heights, y_centers)

        x_center_grid = tf.reshape(x_center_grid, (-1, 1))
        y_center_grid = tf.reshape(y_center_grid, (-1, 1))
        width_grid = tf.reshape(width_grid, (-1, 1))
        height_grid = tf.reshape(height_grid, (-1, 1))

        x_min_grid = x_center_grid - 0.5 * width_grid
        y_min_grid = y_center_grid - 0.5 * height_grid
        x_max_grid = x_center_grid + 0.5 * width_grid
        y_max_grid = y_center_grid + 0.5 * height_grid

        # 切换anchor到[y, x, h, w]
        # boxs = tf.concat([y_center_grid, x_center_grid, height_grid, width_grid], axis=1)
        boxs = tf.concat([y_min_grid, x_min_grid, y_max_grid, x_max_grid], axis=1)

        return boxs

def generate_anchors_per_locations(num_layers = 6):
    num_anchors_per_location = list()

    total_scales, total_ratios = create_scales_and_ratios(num_layers=num_layers)
    for scales in total_scales:
        num_anchors_per_location.append(len(scales))
    return num_anchors_per_location


def generate_anchors(feature_map_shapes):
    """生成多尺度的anchor，确定图像大小和特征图大小以后生成就不变
                """
    total_scales, total_ratios = create_scales_and_ratios(num_layers=len(feature_map_shapes))

    anchor_strides = [(1.0 / float(pair[0]), 1.0 / float(pair[1])) for pair in feature_map_shapes]
    anchor_offsets = [(0.5 * stride[0], 0.5 * stride[1]) for stride in anchor_strides]

    anchors = list()
    for grid_size, scales, aspect_ratios, anchor_stride, anchor_offset \
            in zip(feature_map_shapes, total_scales, total_ratios, anchor_strides, anchor_offsets):
        tail_anchors = tile_anchors(grid_size[0], grid_size[1], scales,
                                          aspect_ratios, anchor_stride, anchor_offset)

        for idx in range(tail_anchors.shape[0]):
            tail_anchor = tail_anchors[idx]
            anchors.append(tail_anchor)

    anchors = tf.stack(anchors)
    return anchors


class FastPanoptic(Model):
    def __init__(self,
                 weights='pascal_voc',
                 input_shape=(224, 224, 3),
                 classes=21,
                 backbone='mobilenet_v2',
                 alpha=1,
                 activation=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.classes = classes
        self.activation = activation
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
        self.inverted_res_block0 = InvertedResBlock(filters=16, alpha=alpha, stride=1, expansion=1, block_id=0, skip_connection=False)
        self.inverted_res_block1 = InvertedResBlock(filters=24, alpha=alpha, stride=2, expansion=6, block_id=1, skip_connection=False)
        self.inverted_res_block2 = InvertedResBlock(filters=24, alpha=alpha, stride=1, expansion=6, block_id=2, skip_connection=True)
        self.inverted_res_block3 = InvertedResBlock(filters=32, alpha=alpha, stride=2, expansion=6, block_id=3, skip_connection=False)
        self.inverted_res_block4 = InvertedResBlock(filters=32, alpha=alpha, stride=1, expansion=6, block_id=4, skip_connection=True)
        self.inverted_res_block5 = InvertedResBlock(filters=32, alpha=alpha, stride=1, expansion=6, block_id=5, skip_connection=True)
        self.inverted_res_block6 = InvertedResBlock(filters=64, alpha=alpha, stride=2, expansion=6, block_id=6, skip_connection=False)
        self.inverted_res_block7 = InvertedResBlock(filters=64, alpha=alpha, stride=1, expansion=6, block_id=7, skip_connection=True)
        self.inverted_res_block8 = InvertedResBlock(filters=64, alpha=alpha, stride=1, expansion=6, block_id=8, skip_connection=True)
        self.inverted_res_block9 = InvertedResBlock(filters=64, alpha=alpha, stride=1, expansion=6, block_id=9, skip_connection=True)
        self.inverted_res_block10 = InvertedResBlock(filters=96, alpha=alpha, stride=1, expansion=6, block_id=10, skip_connection=False)
        self.inverted_res_block11 = InvertedResBlock(filters=96, alpha=alpha, stride=1, expansion=6, block_id=11, skip_connection=True)
        self.inverted_res_block12 = InvertedResBlock(filters=96, alpha=alpha, stride=1, expansion=6, block_id=12, skip_connection=True)
        self.inverted_res_block13 = InvertedResBlock(filters=160, alpha=alpha, stride=2, expansion=6, block_id=13, skip_connection=False)
        self.inverted_res_block14 = InvertedResBlock(filters=160, alpha=alpha, stride=1, expansion=6, block_id=14, skip_connection=True)
        self.inverted_res_block15 = InvertedResBlock(filters=160, alpha=alpha, stride=1, expansion=6, block_id=15, skip_connection=True)
        self.inverted_res_block16 = InvertedResBlock(filters=320, alpha=alpha, stride=1, expansion=6, block_id=16, skip_connection=False)

        self.branch1 = Conv2DBlock(filters=6 * 96, kernel_size=1, strides=1, use_bias=False, padding='same')
        last_block_filters = 320 * 4
        if alpha > 1.0:
            last_block_filters = _make_divisible(last_block_filters * alpha, 8)
        self.branch2 = Conv2DBlock(filters=last_block_filters, kernel_size=1, strides=1, use_bias=False, padding='same')
        # you can use it with arbitary number of classes
        if (weights == 'pascal_voc' and classes == 21) or (weights == 'cityscapes' and classes == 19):
            self.last_layer_name = 'logits_semantic'
        else:
            self.last_layer_name = 'custom_logits_semantic'

        self.ssd_extend = SSDFeatureExpandLayer()

        self.anchors_per_locations = generate_anchors_per_locations()

        self.ssd_predictor = SSDPredictor(anchors_per_locations=self.anchors_per_locations,
                                          num_class_with_background=self.classes,
                                          use_depthwise=False)

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
        b1 = self.branch1(x)
        x = self.inverted_res_block13(x)
        x = self.inverted_res_block14(x)
        x = self.inverted_res_block15(x)
        x = self.inverted_res_block16(x)
        b2 = self.branch2(x)
        img_input = x
        # Image Feature branch
        shape_before = tf.shape(img_input)
        b3 = GlobalAveragePooling2D()(x)
        # from (b_size, channels)->(b_size, 1, 1, channels)
        b3 = Lambda(lambda x: K.expand_dims(x, 1))(b3)
        b3 = Lambda(lambda x: K.expand_dims(x, 1))(b3)
        b3 = Conv2D(256, (1, 1), padding='same',
                    use_bias=False, name='image_pooling')(b3)
        b3 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b3)
        b3 = Activation(tf.nn.relu)(b3)
        # upsample. have to use compat because of the option align_corners
        size_before = tf.keras.backend.int_shape(x)
        b3 = Lambda(lambda v: tf.compat.v1.image.resize(v, size_before[1:3],
                                                        method='bilinear', align_corners=True))(b3)
        # simple 1x1
        b4 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
        b4 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b4)
        b4 = Activation(tf.nn.relu, name='aspp0_activation')(b4)

        x = Concatenate()([b3, b4])

        x = Conv2D(256, (1, 1), padding='same',
                   use_bias=False, name='concat_projection')(x)
        x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
        x = Activation(tf.nn.relu)(x)
        x = Dropout(0.1)(x)
        # DeepLab v.3+ decoder

        x = Conv2D(self.classes, (1, 1), padding='same', name=self.last_layer_name)(x)
        size_before3 = tf.keras.backend.int_shape(img_input)
        x = Lambda(lambda v: tf.compat.v1.image.resize(v,
                                                        size_before3[1:3],
                                                        method='bilinear', align_corners=True))(x)

        if self.activation in {'softmax', 'sigmoid'}:
            x = Activation(self.activation)(x)

        #print(branch1.shape, branch2.shape)
        # SSD
        extend_features = self.ssd_extend(b2)
        extend_features = [b1] + extend_features
        
        labels_feature, bboxes_feature = self.ssd_predictor(extend_features)
        
        outputs = {
                   'Segmantic_Segmentation': x,
                   'detection_classes': labels_feature,
                   'detection_bboxes': bboxes_feature
                   }
        return outputs


def fast_panoptic(weights='pascal_voc', input_tensor=None, input_shape=(224, 224, 3), classes=21, backbone='mobilenetv2',
              OS=16, alpha=1., activation=None):
    
	# Deeplabv3
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor

    OS = 8
    first_block_filters = _make_divisible(32 * alpha, 8)
    print(img_input.shape)
    x = Conv2D(first_block_filters,
               kernel_size=3,
               strides=(2, 2), padding='same',
               use_bias=False, name='Conv')(img_input)
    x = BatchNormalization(
        epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
    x = Activation(tf.nn.relu6, name='Conv_Relu6')(x)
    print(x.shape)
    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                            expansion=1, block_id=0, skip_connection=False)
    print(x.shape)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                            expansion=6, block_id=1, skip_connection=False)
    print(x.shape)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=2, skip_connection=True)
    print(x.shape)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                            expansion=6, block_id=3, skip_connection=False)
    print(x.shape)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=4, skip_connection=True)
    print(x.shape)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=5, skip_connection=True)
    print(x.shape)

    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2,
                            expansion=6, block_id=6, skip_connection=False)
    print(x.shape)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=7, skip_connection=True)
    print(x.shape)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=8, skip_connection=True)
    print(x.shape)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=9, skip_connection=True)
    print(x.shape)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=10, skip_connection=False)
    print(x.shape)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=11, skip_connection=True)
    print(x.shape)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=12, skip_connection=True)
    print(x.shape)
    # for SSD
    branch1 = _conv_2d_block(x, filters=6 * 96, kernel_size=1, strides=1, padding='same', use_bias=False)

    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=2,
                            expansion=6, block_id=13, skip_connection=False)
    print(x.shape)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                            expansion=6, block_id=14, skip_connection=True)
    print(x.shape)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                            expansion=6, block_id=15, skip_connection=True)
    print(x.shape)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1,
                            expansion=6, block_id=16, skip_connection=False)
    print(x.shape)

    last_block_filters = 320 * 4
    if alpha > 1.0:
    	last_block_filters = _make_divisible(last_block_filters * alpha, 8)

    # for SSD
    branch2 = _conv_2d_block(x, filters=last_block_filters, kernel_size=1, strides=1, use_bias=False, padding='same')

    # end of feature extractor

    # branching for Atrous Spatial Pyramid Pooling

    # Image Feature branch
    shape_before = tf.shape(x)
    b4 = GlobalAveragePooling2D()(x)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Conv2D(256, (1, 1), padding='same',
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation(tf.nn.relu)(b4)
    # upsample. have to use compat because of the option align_corners
    size_before = tf.keras.backend.int_shape(x)
    b4 = Lambda(lambda x: tf.compat.v1.image.resize(x, size_before[1:3],
                                                    method='bilinear', align_corners=True))(b4)
    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation(tf.nn.relu, name='aspp0_activation')(b0)

    x = Concatenate()([b4, b0])

    x = Conv2D(256, (1, 1), padding='same',
               use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation(tf.nn.relu)(x)
    x = Dropout(0.1)(x)
    # DeepLab v.3+ decoder

    # you can use it with arbitary number of classes
    if (weights == 'pascal_voc' and classes == 21) or (weights == 'cityscapes' and classes == 19):
        last_layer_name = 'logits_semantic'
    else:
        last_layer_name = 'custom_logits_semantic'

    x = Conv2D(classes, (1, 1), padding='same', name=last_layer_name)(x)
    size_before3 = tf.keras.backend.int_shape(img_input)
    x = Lambda(lambda xx: tf.compat.v1.image.resize(xx,
                                                    size_before3[1:3],
                                                    method='bilinear', align_corners=True))(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    if activation in {'softmax', 'sigmoid'}:
        x = tf.keras.layers.Activation(activation)(x)

    #print(branch1.shape, branch2.shape)
    # SSD
    extend_features = SSDFeatureExpandLayer()(branch2)
    extend_features = [branch1] + extend_features
    
    anchors_per_locations = generate_anchors_per_locations()
    labels_feature, bboxes_feature = SSDPredictor(anchors_per_locations=anchors_per_locations,
    											  num_class_with_background=classes,
    											  use_depthwise=False)(extend_features)
    
    outputs = {'detection_classes': labels_feature,
    		   'detection_bboxes': bboxes_feature
    		   }


    #model1 = Model(inputs, x, name='deeplabv3plus')
    #model2 = Model(inputs, outputs, name='ssd_mobilenet_v2')
    model = Model(inputs, [x, outputs], name='fast_panoptic')
    #return model1, model2
    return model

# ========== Feature Expand ==========

class HeadOp(tf.keras.layers.Layer):
    def __init__(self,
                 kernel_size,
                 filters,
                 stride,
                 use_bias,
                 ac,
                 bn,
                 use_depthwise = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.ops = list()

        if use_depthwise:
            dp1 = DepthwiseConv2DBN(
                kernel_size=kernel_size,
                strides=stride,
                padding='SAME',
                use_bias=False
            )

            cv1 = Conv2DBlock(
                filters=filters,
                kernel_size=1,
                strides=1,
                padding='SAME',
                use_bias=use_bias,
                ac=ac,
                bn=bn
            )
            self.ops.append(dp1)
            self.ops.append(cv1)
        else:
            cv2 = Conv2DBlock(
                filters=filters,
                kernel_size=kernel_size,
                strides=stride,
                padding='SAME',
                use_bias=use_bias,
                ac = ac,
                bn = bn,
                kernel_initializer=truncated_normal_initializer()
            )
            self.ops.append(cv2)

    def call(self, inputs, training = False):
        outputs = inputs
        for op in self.ops:
            outputs = op(outputs, training)
        return outputs

class SSDConv2DStack(tf.keras.layers.Layer):

    def __init__(self,
                 depth,
                 use_depthwise = False,
                 **kwargs):
        super().__init__(**kwargs)

        self.ops = list()

        name0 = "{}_1_1x1_{}".format(self.name, int(depth/2))
        conv0 = Conv2DBlock(
            filters=int(depth/2),
            kernel_size=1,
            strides=1,
            padding='SAME',
            use_bias=False,
            kernel_initializer=truncated_normal_initializer(),
            name=name0
        )
        self.ops.append(conv0)

        name1 = "{}_2_3x3_{}".format(self.name, depth)
        op2 = HeadOp(kernel_size=3,
                     filters=depth,
                     stride=2,
                     use_bias=False,
                     ac = False,
                     bn = False,
                     use_depthwise=use_depthwise,
                     name=name1)
        self.ops.append(op2)

    def call(self, inputs, training=False):
        outputs = inputs
        for op in self.ops:
            outputs = op(outputs, training)
        return outputs



class SSDFeatureExpandLayer(tf.keras.layers.Layer):
    '''特征提取
    '''

    def __init__(self,
                 use_depthwise=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.use_depthwise = use_depthwise
        expand_depthes = [512, 256, 256, 128]

        # 生成一系列的ops
        self.ops = list()
        for expand_depth in expand_depthes:
            name_ext = "{}_{}".format(self.name, expand_depth)
            op = SSDConv2DStack(depth=expand_depth,
                                use_depthwise=self.use_depthwise,
                                name=name_ext)
            self.ops.append(op)

    def call(self, inputs, training=False):
        outputs = inputs
        total_outputs = list()
        total_outputs.append(outputs)
        for op in self.ops:
            outputs = op(outputs, training)
            total_outputs.append(outputs)
        return total_outputs


# ============ Predictor ============
class LabelBboxPredictor(tf.keras.layers.Layer):

    def __init__(self,
                 anchors_per_locations,
                 code_size,
                 use_depthwise=False,
                 **kwargs):
        super().__init__(**kwargs)

        kernel_size = 1 if not use_depthwise else 3
        strides = 1

        self.ops = list()
        for anchors_per_location in anchors_per_locations:
            filters = code_size * anchors_per_location

            if use_depthwise:
                dp1 = DepthwiseConv2DBN(
                    kernel_size=kernel_size,
                    strides=strides,
                    padding='SAME',
                    use_bias=False
                )

                cv1 = Conv2DBlock(
                    filters=filters,
                    kernel_size=1,
                    strides=1,
                    padding='SAME',
                    use_bias=True,
                    ac=False,
                    bn=False
                )
                self.ops.append([dp1, cv1])
            else:
                cv2 = Conv2DBlock(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding='SAME',
                    use_bias=True,
                    ac=False,
                    bn=False,
                    kernel_initializer=truncated_normal_initializer()
                )
                self.ops.append([cv2])

        self.reshape = tf.keras.layers.Reshape((-1, code_size))
        self.concatenate = tf.keras.layers.Concatenate(axis=1)

    def call(self, inputs, training = False):
        features = inputs
        if (len(features) != len(self.ops)):
            raise ValueError("feature map length not match ops length")

        outputs = list()
        for idx in range(len(features)):
            feature = features[idx]
            op = self.ops[idx]
            for p in op:
                feature = p(feature)
            feature = self.reshape(feature)
            outputs.append(feature)
        outputs = self.concatenate(outputs)
        return outputs



class SSDPredictor(tf.keras.layers.Layer):

    def __init__(self,
                 anchors_per_locations,
                 num_class_with_background,
                 box_code_size=4,
                 use_depthwise=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.bboxes_op = LabelBboxPredictor(anchors_per_locations=anchors_per_locations,
                                            code_size=box_code_size,
                                            use_depthwise = use_depthwise)

        self.labels_op = LabelBboxPredictor(anchors_per_locations=anchors_per_locations,
                                            code_size=num_class_with_background,
                                            use_depthwise=use_depthwise)


    def call(self, inputs, training = False):

        label_features = self.labels_op(inputs)
        bbox_features = self.bboxes_op(inputs)

        return label_features, bbox_features


def preprocess_input(x):
    """Preprocesses a numpy array encoding a batch of images.
    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].
    # Returns
        Input array scaled to [-1.,1.]
    """
    return preprocess_input(x, mode='tf')


if __name__ == '__main__':
	#model = fast_panoptic()
    model = FastPanoptic()
    model.build(input_shape=(None, 224, 224, 3))
    model.summary(line_length=200, positions=[.33, .5, .7, 1.])
    '''
    model1, model2 = FastPanoptic()
    model1.build(input_shape=(None, 300, 300, 3))
    model2.build(input_shape=(None, 300, 300, 3))
    model1.summary(line_length=200)
    model2.summary(line_length=200)
    '''