import tensorflow as tf
from tensorflow.contrib import slim

from nets import resnet_v1
from utils.training import get_valid_logits_and_labels
from nets import resnet_utils
num_classes = 2
FLAGS = tf.app.flags.FLAGS

def unpool(inputs,scale):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*scale,  tf.shape(inputs)[2]*scale])


def ResidualConvUnit(inputs,features=256,kernel_size=3):
    net=tf.nn.relu(inputs)
    net=slim.conv2d(net, features, kernel_size)
    net=tf.nn.relu(net)
    net=slim.conv2d(net,features,kernel_size)
    net=tf.add(net,inputs)

    return net

def MultiResolutionFusion(high_inputs=None,low_inputs=None,up0=2,up1=1,n_i=256):

    g0 = unpool(slim.conv2d(high_inputs, n_i, 3), scale=up0)

    if low_inputs is None:
        return g0

    g1=unpool(slim.conv2d(low_inputs,n_i,3),scale=up1)
    return tf.add(g0,g1)

def ChainedResidualPooling(inputs,n_i=256):
    net_relu=tf.nn.relu(inputs)
    net=slim.max_pool2d(net_relu, [5, 5],stride=1,padding='SAME')
    net=slim.conv2d(net,n_i,3)
    return tf.add(net,net_relu)

def RefineBlock(high_inputs=None,low_inputs=None):
    if low_inputs is not None:
        print(high_inputs.shape)
        rcu_high=ResidualConvUnit(high_inputs,features=256)
        rcu_low=ResidualConvUnit(low_inputs,features=256)
        fuse=MultiResolutionFusion(rcu_high,rcu_low,up0=2,up1=1,n_i=256)
        fuse_pooling=ChainedResidualPooling(fuse,n_i=256)
        output=ResidualConvUnit(fuse_pooling,features=256)
        return output
    else:
        rcu_high = ResidualConvUnit(high_inputs, features=256)
        fuse = MultiResolutionFusion(rcu_high, low_inputs=None, up0=1,  n_i=256)
        fuse_pooling = ChainedResidualPooling(fuse, n_i=256)
        output = ResidualConvUnit(fuse_pooling, features=256)
        return output
def build_pyramid(conv_layer=None, num_pyramid=4, is_training=True):
    '''
    :param conv_layer: the conv layer for building pyramids
    :param num_pyramid: the num of all the pyramids, the default number is 4
    :return:
    '''
    batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'is_training': is_training
    }
    conv_shape = conv_layer.get_shape().as_list()
    resize_shape = conv_shape[1:3]
    pyramid_dimension = conv_shape[-1]//num_pyramid
    pyramid_layers = []
    with tf.variable_scope('pyramid_module'):

        if conv_shape[1] != conv_shape[2]:
            raise ValueError('The input convolution layer must be square!')
        else:
            for size in range(num_pyramid, 0, -1):
                new_size = conv_shape[1]//2**(size-1)
                name = 'avg_pool_pyramid_%d' % new_size
                avg_conv_pool = slim.avg_pool2d(conv_layer, new_size, stride=new_size, scope=name)
                name = 'conv_avg_pool_pyramid_%d' % new_size
                conv_avg_conv_pool = slim.conv2d(avg_conv_pool, pyramid_dimension, kernel_size=1, stride=1, normalizer_fn=slim.batch_norm,
                                                normalizer_params = batch_norm_params,
                                                weights_regularizer = slim.l2_regularizer(1e-5), activation_fn=tf.nn.relu, scope=name)
                name = 'resize_avg_pool_pyramid_%d' % new_size
                resize_layer = tf.image.resize_bilinear(conv_avg_conv_pool, resize_shape, name=name)
                pyramid_layers.append(resize_layer)
        return pyramid_layers

def atrous_spatial_pyramd_pooling(inputs, output_stride, is_training, depth=256):
    batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'is_training': is_training
    }
    with tf.variable_scope('aspp'):
        atrous_rates = [6, 12, 18]
        if output_stride == 8:
            atrous_rates = [2*rate for rate in atrous_rates]
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            inputs_size = tf.shape(inputs)[1:3]
            conv_1x1 = slim.conv2d(inputs, depth, [1, 1], stride=1, scope='conv_1x1')
            conv_3x3_1 = resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[0], scope='conv_3x3_1')
            conv_3x3_2 = resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[1], scope='conv_3x3_2')
            conv_3x3_3 = resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[2], scope='conv_3x3_3')
            with tf.variable_scope('image_level_features'):
                image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keep_dims=True)
                image_level_features = slim.conv2d(image_level_features, depth, [1, 1], stride=1, scope='conv_1x1')
                image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')
            net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3, name='concat')
            net = slim.conv2d(net, depth, [1, 1], stride=1, scope='conv_1x1_concat')
            return net

def up_res_conv(input, depth, stride, name=None, is_training=True):
    batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'is_training': is_training
    }
    with tf.variable_scope(name):
        depth_in = slim.utils.last_dimension(input.get_shape(), min_rank=4)
        up_resdual = slim.conv2d(input, depth, kernel_size=[3, 3], stride=1, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm,
                                 normalizer_params=batch_norm_params,scope='up_conv1')
        up_resdual = slim.conv2d(up_resdual, depth, kernel_size=[3, 3], stride=stride, activation_fn=tf.nn.relu, scope='up_conv2')
        return up_resdual

def up_res_conv50(input, depth, stride, name=None, is_training=True):
    with tf.variable_scope(name):
        up_resdual = slim.conv2d(input, depth, [3, 3], stride=1, scope='conv1')
        up_resdual = slim.conv2d(up_resdual, depth, [3, 3], stride=1, activation_fn=tf.nn.relu, scope='conv2')
        up_resdual = slim.conv2d(up_resdual, depth, [3, 3], stride=1, activation_fn=tf.nn.relu, scope='conv3')
    return up_resdual

def up_concat_conv(low_layer=None, high_layer=None, name=None, is_merge=True):
    with tf.variable_scope(name) as sc:
        low_shape = low_layer.get_shape().as_list()
        high_shape = high_layer.get_shape().as_list()
        if low_shape[1] == high_shape[1]:
            up_low_layer = slim.conv2d_transpose(low_layer, high_shape[-1], kernel_size=2, stride=1, scope='up_low_layer')
        else:
            up_low_layer = slim.conv2d_transpose(low_layer, high_shape[-1], kernel_size=2, stride=2, scope='up_low_layer')
        if is_merge:
            merge = tf.concat([up_low_layer, high_layer], 3)
        else:
            merge = up_low_layer
        #up_resdual = up_res_conv50(merge, high_shape[-1], stride=1, name='up_resdual')
        #output = slim.conv2d(up_resdual, high_shape[-1], 1, 1, normalizer_fn=None, scope='activation_layer')
    return merge
def up_concat_bilinear(low_layer=None, high_layer=None, name=None, is_merge=True):
    with tf.variable_scope(name) as sc:
        low_shape = low_layer.get_shape().as_list()
        high_shape = high_layer.get_shape().as_list()
        up_low_size = high_shape[1: 3]
        up_low_layer = tf.image.resize_bilinear(low_layer, up_low_size, name='up_low_layer')
        up_low_layer = slim.conv2d(up_low_layer, high_shape[-1], 3, 1, scope='up_low_layer_conv')
        if is_merge:
            merge = tf.concat([up_low_layer, high_layer], 3)
        else:
            merge = up_low_layer
        up_resdual = up_res_conv(merge, high_shape[-1], stride=1, name='up_resdual')
    return up_resdual

def model(images, weight_decay=1e-5, is_training=True):
    images = mean_image_subtraction(images)

    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
        logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')

    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'is_training': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            f = [end_points['pool7'], end_points['pool6'], end_points['pool5'], end_points['pool4'], end_points['pool3']]
            for i in range(5):
                print('Shape of f_{} {}'.format(i, f[i].shape))
            """orginal resnet-50"""
            low_layer = up_res_conv(f[0], f[1].get_shape().as_list()[-1], stride=1, name='conv_f0', is_training=is_training)
            for j in range(1, 4, 1):
                name = 'up_low_layer_%d' % j
                output = up_concat_conv(low_layer, f[j], name=name, is_merge=True)
                if j == 3:
                    output = up_res_conv(output, f[j + 1].get_shape().as_list()[-1]*2, stride=1, name=name, is_training=is_training)
                else:
                    output = up_res_conv(output, f[j+1].get_shape().as_list()[-1], stride=1, name=name, is_training=is_training)
                low_layer = output
            name = 'up_low_layer_%d' % 4
            output = up_concat_conv(low_layer, f[4], name=name, is_merge=True)
            output = up_res_conv(output, f[4].get_shape().as_list()[-1], stride=1, name=name, is_training=is_training)
            output = tf.image.resize_bilinear(output, [256, 256])
            logit = slim.conv2d(output, num_classes, 1, 1)
            logit = tf.nn.relu(logit)

            """atrous_spatial_pyramd_pooling"""
            '''inputs_size = tf.shape(images)[1:3]
            net = end_points['pool7']
            net = atrous_spatial_pyramd_pooling(net, 16, is_training)
            with tf.variable_scope("upsampling_logits"):
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None,
                                        scope='conv_1x1')
                logit = tf.image.resize_bilinear(net, inputs_size, name='upsample')'''

            '''inputs_size = tf.shape(images)[1:3]
            pyramid_layer = end_points['pool7']
            pyramid_layers = build_pyramid(pyramid_layer)
            net = tf.concat(pyramid_layers, axis=3, name='concat_layers')
            net = tf.concat([net, pyramid_layer], axis=3, name='concat_layer')
            net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv_1x1')
            logit = tf.image.resize_bilinear(net, inputs_size, name='upsample')'''

    return logit


def mean_image_subtraction(images, means=[83.6150508876, 70.1435573964, 53.9194863905]): #, 106.58, 43.17
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)

def loss(annotation_batch,upsampled_logits_batch,class_labels):
    valid_labels_batch_tensor, valid_logits_batch_tensor = get_valid_logits_and_labels(
        annotation_batch_tensor=annotation_batch,
        logits_batch_tensor=upsampled_logits_batch,
        class_labels=class_labels)

    cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=valid_logits_batch_tensor,
                                                              labels=valid_labels_batch_tensor)

    cross_entropy_sum = tf.reduce_mean(cross_entropies)
    tf.summary.scalar('cross_entropy_loss', cross_entropy_sum)

    return cross_entropy_sum

