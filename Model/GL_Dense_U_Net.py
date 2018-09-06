# -*- coding: UTF-8 -*-
import sys
sys.path.append(r'/media/cugxyy/c139cfbf-11c3-4275-9602-b96afc28d10c/DL/Road_Segmentation')
import tensorflow as tf
from tensorflow.contrib import slim
from utils.training import get_valid_logits_and_labels

growth_k = 16
layers_per_block = [4, 5, 7, 10, 12]   # [2, 4, 5, 7, 10] get the feature map is 1/16*input_image
#layers_per_block = [2, 4, 5, 7, 10]
num_classes = 2
nb_blocks = len(layers_per_block)

def batch_norm(x, training, name):
    training = tf.cast(training, tf.bool)
    with tf.variable_scope(name):
        x = tf.cond(training, lambda: slim.batch_norm(x, decay=0.997, epsilon=1e-5, scale=True, is_training=training, scope=name + '_batch_norm',reuse=None),
                    lambda: slim.batch_norm(x, decay=0.997, epsilon=1e-5, scale=True, is_training=training, scope=name + '_batch_norm', reuse=True))
    return x

def conv_layer(x, training, filters, name):

    with tf.variable_scope(name):
        x = batch_norm(x, training=training, name='conv_layer_batch_norm')
        x = tf.nn.relu(x, name='conv_layer_relu')
        x = slim.conv2d(x, filters, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu, scope=name + '_conv3x3')
        #x = slim.dropout(x, keep_prob=0.2, is_training=training, scope=name+'_dropout')
    return x

def dense_block(x, training, block_nb, name):
    dense_out = []
    with tf.variable_scope(name):
        for i in range(layers_per_block[block_nb]):
            conv = conv_layer(x, training, growth_k, name=name+'_layer_'+str(i))
            x = tf.concat([conv, x], axis=3)
            dense_out.append(conv)
        x = tf.concat(dense_out, axis=3)
    return x

def transition_down(x, training, filters, name):
    with tf.variable_scope(name):
        x = slim.conv2d(x, filters, kernel_size=1, stride=1, padding='SAME', activation_fn=tf.nn.relu, scope=name + '_conv1x1')
        #x = slim.dropout(x, keep_prob=0.8, is_training=training, scope=name + '_dropout')
        x = slim.max_pool2d(x, kernel_size=4, stride=2, padding='SAME', scope=name+'_maxpool2x2')
    return x

def transition_up(x, filters, name):
    with tf.variable_scope(name):
        x = slim.conv2d_transpose(x, filters, kernel_size=3, stride=2,
                                  padding='SAME', activation_fn=tf.nn.relu, scope=name+'_trans_conv3x3')
    return x

def global_average_pooling(x, name='GlobalAvePool'):
    in_shape = x.get_shape().as_list()
    assert len(in_shape) == 4, 'Input tensor shape must be 4!'
    with tf.name_scope(name):
        #x_out = tf.reduce_mean(x, [1, 2])
        x_out = slim.avg_pool2d(x, kernel_size=in_shape[2], stride=1, padding='SAME', scope=name)
    return x_out
def feature_pyramid_attention1(last_conv_layer, name=None):
    with tf.variable_scope('FPA') as scope:
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            activation_fn=tf.nn.relu,
                            padding='SAME'):
            in_shape = last_conv_layer.get_shape().as_list()
            x_global_conv = global_average_pooling(last_conv_layer)
            x_global_conv = slim.conv2d(x_global_conv, in_shape[-1], 1, 1, scope=name+'global_ave_conv')
            x_conv_1x1 = slim.conv2d(last_conv_layer, in_shape[-1], 1, 1, scope=name+'conv_1x1')
            x_conv_7x7 = slim.conv2d(last_conv_layer, in_shape[-1], kernel_size=7, stride=2, scope=name+'conv_7x7')
            x_conv_5x5 = slim.conv2d(last_conv_layer, in_shape[-1], kernel_size=5, stride=4, scope=name+'conv_5x5')
            x_conv_3x3 = slim.conv2d(last_conv_layer, in_shape[-1], kernel_size=3, stride=8, scope=name+'conv_3x3')
            x_conv_3x3_up = slim.conv2d_transpose(x_conv_3x3, in_shape[-1], kernel_size=3, stride=2, scope=name+'up_conv_3x3')
            x_concate_3x5 = tf.concat([x_conv_3x3_up, x_conv_5x5], axis=3, name=name+'x_concate_3x5')
            x_concate_3x5_up = slim.conv2d_transpose(x_concate_3x5, in_shape[-1], kernel_size=3, stride=2, scope=name+'up_concate_3x5')
            x_concate_5x7 = tf.concat([x_concate_3x5_up, x_conv_7x7], axis=3, name=name+'x_concate_5x7')
            x_concate_5x7_up = slim.conv2d_transpose(x_concate_5x7, in_shape[-1], kernel_size=3, stride=2, scope=name+'up_concate_5x7')
            x_multi_1x7 = x_conv_1x1 * x_concate_5x7_up
            x_concate_global_multi = tf.concat([x_multi_1x7, x_global_conv], axis=3, name=name+'x_concate_global_multi')
            x_concate_global_multi = slim.conv2d(x_concate_global_multi, in_shape[-1], 1, 1, scope=name+'x_concate_global_multi_1x1')
    return x_concate_global_multi

#modified by xyy
def LPU(last_conv_layer, name=None):
    with tf.variable_scope('FPA') as scope:
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            activation_fn=tf.nn.relu,
                            padding='SAME'):
            in_shape = last_conv_layer.get_shape().as_list()

            x_conv_1x1 = slim.conv2d(last_conv_layer, in_shape[-1], 1, 1, scope=name+'conv_1x1')
            x_conv_7x7 = slim.conv2d(last_conv_layer, in_shape[-1], kernel_size=7, stride=2, scope=name+'conv_7x7')
            x_conv_5x5 = slim.conv2d(last_conv_layer, in_shape[-1], kernel_size=5, stride=4, scope=name+'conv_5x5')
            x_conv_3x3 = slim.conv2d(last_conv_layer, in_shape[-1], kernel_size=3, stride=8, scope=name+'conv_3x3')
            x_conv_3x3_up = slim.conv2d_transpose(x_conv_3x3, in_shape[-1], kernel_size=3, stride=2, scope=name+'up_conv_3x3')
            x_concate_3x5 = tf.concat([x_conv_3x3_up, x_conv_5x5], axis=3, name=name+'x_concate_3x5')
            x_concate_3x5_up = slim.conv2d_transpose(x_concate_3x5, in_shape[-1], kernel_size=3, stride=2, scope=name+'up_concate_3x5')
            x_concate_5x7 = tf.concat([x_concate_3x5_up, x_conv_7x7], axis=3, name=name+'x_concate_5x7')
            x_concate_5x7_up = slim.conv2d_transpose(x_concate_5x7, in_shape[-1], kernel_size=3, stride=2, scope=name+'up_concate_5x7')
            x_multi_1x7 = x_conv_1x1 * x_concate_5x7_up
            x_concate_global_multi = slim.conv2d(x_multi_1x7, in_shape[-1], 1, 1, scope=name+'x_concate_global_multi_1x1')
    return x_concate_global_multi
def global_pool_upsample(layer, name):
    with tf.variable_scope('GPU') as scope:
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            activation_fn=tf.nn.relu,
                            padding='SAME'):
            in_shape = layer.get_shape().as_list()
            x_global_pool = global_average_pooling(layer, name=name+'layer_global_pool')
            x_global_pool = slim.conv2d(x_global_pool, in_shape[-1], 1, 1, scope=name+'conv_1x1')
            x_global_pool = slim.conv2d_transpose(x_global_pool, in_shape[-1], kernel_size=3, stride=2, scope=name+'layer_global_pool_up')
            return x_global_pool

def global_attention_upsample(low_layer, high_layer, is_training, name):
    with tf.variable_scope('GAU') as scope:
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            padding='SAME'):
            low_shape = low_layer.get_shape().as_list()
            low_layer_conv_3x3 = slim.conv2d(low_layer, low_shape[-1], kernel_size=3, stride=2, scope=name + 'low_layer_conv_3x3')
            high_layer_global_pool = global_average_pooling(high_layer)
            high_layer_1x1 = slim.conv2d(high_layer_global_pool, low_shape[-1], kernel_size=1, stride=1, scope=name + 'high_layer_1x1')
            high_layer_1x1_bn = batch_norm(high_layer_1x1, training=is_training, name=name + 'high_layer_1x1_bn')
            high_layer_1x1_bn = tf.nn.relu(high_layer_1x1_bn, name=name + 'high_layer_1x1_bn_relu')
            low_high_multi = high_layer_1x1_bn * low_layer_conv_3x3
            #up_high_layer = slim.conv2d_transpose(high_layer, low_shape[-1], kernel_size=4, stride=2, scope='up_high_layer')
            concate_layer = tf.concat([high_layer, low_high_multi], axis=3, name=name + 'concat_layer')
            up_high_layer = slim.conv2d_transpose(concate_layer, low_shape[-1], kernel_size=3, stride=2, scope=name + 'up_high_layer')
            return up_high_layer


def model(x, training):
    with tf.variable_scope('Densenet') as sc:
        batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(1e-5)):
            concats = []
            x = slim.conv2d(x, 48, [3, 3], stride=1, padding='SAME', activation_fn=tf.nn.relu, normalizer_fn=None, scope='first_conv3x3')
            #x_first = x
            print(x.get_shape())
            print('Building downsample!')
            for block_nb in range(0, nb_blocks):
                dense = dense_block(x, training, block_nb, 'down_dense_block_' + str(block_nb))
                x = tf.concat([x, dense], axis=3, name='down_concat_' + str(block_nb))
                print(x.get_shape())
                concats.append(x)
                if block_nb !=nb_blocks-1:
                    x = transition_down(x, training, x.get_shape()[-1], 'trans_down_' + str(block_nb))
                    if block_nb == 0:
                        x_first = x
                '''if block_nb !=nb_blocks:

                else:
                    dense = slim.conv2d(x, 48, kernel_size=3, stride=1, padding='SAME',scope='last_dense')'''
            x_last = LPU(x, name='last_layer')
            print('Building upsample!')
            for i, block_nb in enumerate(range(nb_blocks-1, 0, -1)):
                n_c = x_last.get_shape()[-1]
                #gau_x = global_attention_upsample(concats[len(concats) - i - 2], x, is_training=training, name = 'gau_'+str(block_nb))
                x = transition_up(x_last, x_last.get_shape()[-1], 'trans_up_' + str(block_nb))
                for_layer = global_pool_upsample(x_last, name='GPU%s'%i)
                x = tf.concat([x, for_layer], axis=3, name='up_concat_' + str(block_nb))
                if i == 1:
                    for_fpa_layer = concats[len(concats) - i - 2]
                else:
                    for_fpa_layer = LPU(concats[len(concats) - i - 2], name='for_layer_%s'%i)
                x = tf.concat([x,for_fpa_layer], axis=3, name='up_concat_fpa_' + str(block_nb))
                print(x.get_shape())

                x = slim.conv2d(x, n_c, kernel_size=1, stride=1, padding='SAME', activation_fn=None,
                                normalizer_fn=None, scope='up_concat_1x1' + str(block_nb))
                #remove
                x = dense_block(x, training, block_nb, 'up_dense_block_' + str(block_nb))
                x_last = x
            x = slim.conv2d(x, num_classes, kernel_size=1, stride=1, padding='SAME', activation_fn=None, normalizer_fn=None, scope='last_conv1x1')
            print(x.get_shape())
    ### return the first conv layer
    return x, x_first
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











