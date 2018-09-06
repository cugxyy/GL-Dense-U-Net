import tensorflow as tf
from tensorflow.python.ops import gen_logging_ops
from tensorflow.python.framework import ops as _ops
import time
import shutil
import datetime
import os
import pickle
import cv2
import numpy as np
from sklearn import metrics
from tensorflow.contrib import slim
import sys

sys.path.append(os.getcwd())
sys.path.append(r'/media/cugxyy/c139cfbf-11c3-4275-9602-b96afc28d10c/DL/Road_Segmentation')
import Model.GL_Dense_U_Net as model

from utils.tf_records import read_tfrecord_and_decode_into_image_annotation_pair_tensors
from utils.pascal_voc import pascal_segmentation_lut
from utils.augmentation import (distort_randomly_image_color, flip_randomly_left_right_image_with_annotation,
                                scale_randomly_image_with_annotation_with_fixed_size_output)

tf.app.flags.DEFINE_integer('batch_size', 2, '')
tf.app.flags.DEFINE_integer('train_size', 256, '')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, '')
tf.app.flags.DEFINE_integer('max_steps', 1400000, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_integer('num_classes', 2, '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', r'/media/cugxyy/c139cfbf-11c3-4275-9602-b96afc28d10c/DL/Road_Segmentation/checkpoints_fcn/', '')
tf.app.flags.DEFINE_string('logs_path', r'/media/cugxyy/c139cfbf-11c3-4275-9602-b96afc28d10c/DL/Road_Segmentation/logs/', '')
tf.app.flags.DEFINE_boolean('restore', False, 'whether to resotre from checkpoint')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 10000, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 20, '')
tf.app.flags.DEFINE_integer('save_image_steps', 100, '')
tf.app.flags.DEFINE_string('training_data_path',
                           r'/media/cugxyy/c139cfbf-11c3-4275-9602-b96afc28d10c/DL/Road_Segmentation/traintemp.tfrecords',
                           '')
tf.app.flags.DEFINE_string('pretrained_model_path', None, '')
tf.app.flags.DEFINE_integer('decay_steps', 30000, '')
tf.app.flags.DEFINE_float('decay_rate', 0.1, '')
FLAGS = tf.app.flags.FLAGS


def tower_loss(images, annotation, class_labels, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        #logits = model.model(images, training=True)
        vgg32 = model.FCN8VGG()
        logits = vgg32.build(images, train=True, num_classes=2)
    pred = tf.argmax(logits, axis=3)

    model_loss = model.loss(annotation, logits, class_labels)
    total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # add summary
    if reuse_variables is None:
        tf.summary.scalar('model_loss', model_loss)
        tf.summary.scalar('total_loss', total_loss)
    return total_loss, model_loss, pred


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def build_image_summary():
    log_image_data = tf.placeholder(tf.uint8, [None, None, 3])
    log_image_name = tf.placeholder(tf.string)
    log_image = gen_logging_ops.image_summary(log_image_name, tf.expand_dims(log_image_data, 0), max_images=1)
    _ops.add_to_collection(_ops.GraphKeys.SUMMARIES, log_image)
    return log_image, log_image_data, log_image_name

def cal_acc_recall(pred_image, annotation):
    pred_img = pred_image.flatten()
    annotation_img = annotation.flatten()
    confusion_matrix = metrics.confusion_matrix(annotation_img, pred_img)
    mean_acc = np.mean(metrics.precision_score(annotation_img, pred_img, average=None))
    acc_for_1 = metrics.precision_score(annotation_img, pred_img, average=None)[1]
    recall_for_1 = confusion_matrix[1][1]/np.sum(confusion_matrix, axis=1)[1]
    # report = metrics.classification_report(annotation_img, pred_img)
    # print(confusion_matrix)
    # print(report)
    return mean_acc, acc_for_1, recall_for_1


def main(argv=None):
    gpus = range(len(FLAGS.gpu_list.split(',')))
    pascal_voc_lut = pascal_segmentation_lut()
    class_labels = pascal_voc_lut.keys()
    color_map = {'0': [0, 0, 0], '1': [255, 255, 255], '255': [255, 255, 255]}
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    now = datetime.datetime.now()
    StyleTime = now.strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(FLAGS.logs_path + StyleTime)

    # make a files queue
    filename_queue = tf.train.string_input_producer([FLAGS.training_data_path], num_epochs=1000)
    image, annotation = read_tfrecord_and_decode_into_image_annotation_pair_tensors(filename_queue)

    image, annotation = flip_randomly_left_right_image_with_annotation(image, annotation)
    # image = distort_randomly_image_color(image)
    image = tf.to_float(image)

    image_train_size = [FLAGS.train_size, FLAGS.train_size]
    resized_image, resized_annotation = scale_randomly_image_with_annotation_with_fixed_size_output(image, annotation,
                                                                                                    image_train_size)
    resized_annotation = tf.squeeze(resized_annotation)

    # Create batches by randomly shuffling tensors
    image_batch, annotation_batch = tf.train.shuffle_batch([resized_image, resized_annotation],
                                                           batch_size=FLAGS.batch_size * len(gpus), capacity=1000,
                                                           num_threads=4,
                                                           min_after_dequeue=500)
    # # split a tensor into sub tensors
    input_images_split = tf.split(image_batch, len(gpus))
    input_segs_split = tf.split(annotation_batch, len(gpus))

    learning_rate = tf.Variable(FLAGS.learning_rate, trainable=False)
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    # val_step = tf.get_variable('val_step', [], initializer=tf.constant_initializer(0), trainable=False)
    # add summary
    tf.summary.scalar('learning_rate', learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate)

    tower_grads = []
    reuse_variables = None
    for i, gpu_id in enumerate(gpus):
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('model_%d' % gpu_id) as scope:
                iis = input_images_split[i]
                isms = input_segs_split[i]
                total_loss, model_loss, output_pred = tower_loss(iis, isms, class_labels, reuse_variables)
                batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
                reuse_variables = True
                grads = opt.compute_gradients(total_loss)
                tower_grads.append(grads)

    grads = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)


    summary_op = tf.summary.merge_all()
    log_image, log_image_data, log_image_name = build_image_summary()
    # save moving average
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # batch norm updates
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    summary_writer = tf.summary.FileWriter(FLAGS.logs_path + StyleTime, tf.get_default_graph())

    if FLAGS.pretrained_model_path is not None:
        variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path,
                                                             slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)

    global_vars_init_op = tf.global_variables_initializer()
    local_vars_init_op = tf.local_variables_initializer()
    init = tf.group(local_vars_init_op, global_vars_init_op)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        restore_step = 0
        if FLAGS.restore:
            sess.run(init)
            print('continue training from previous checkpoint')
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            restore_step = int(ckpt.split('_')[6])
            saver.restore(sess, ckpt)
        else:
            sess.run(init)
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)

        # data_generator = get_batch(num_workers=8, batch_size=FLAGS.batch_size * len(gpus))
        # data_generator = generator(batch_size=FLAGS.batch_size * len(gpus))
        start = time.time()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop():
                temp_acc = 0.
                txt_record = open(os.path.join(FLAGS.logs_path, StyleTime, 'record.txt'), 'w')
                for step in range(restore_step, FLAGS.max_steps):
                    if step != 0 and step % FLAGS.decay_steps == 0:
                        sess.run(tf.assign(learning_rate, learning_rate.eval() * FLAGS.decay_rate))

                    ml, tl, _ = sess.run([model_loss, total_loss, train_op])
                    if np.isnan(tl):
                        print('Loss diverged, stop training')
                        break
                    if step % 10 == 0 and step != 0:
                        pred_image, annotation = sess.run([isms, output_pred])
                        _, acc, _ = cal_acc_recall(pred_image, annotation)
                        #acc = acc / FLAGS.batch_size
                        res = 'Step:%d, total loss %.3f, train acc %.3f\n'%(step, tl, acc)
                        txt_record.write(res)
                    if step % 10 == 0 and step != 0:
                        avg_time_per_step = (time.time() - start) / 10
                        start = time.time()
                        print(
                            'Step {:06d}, model loss {:.4f}, total loss {:.4f}, train acc {:.3f}, {:.3f} seconds/step, lr: {:.7f}'.format(
                                step, ml, tl, acc, avg_time_per_step, learning_rate.eval()))
                    if (step + 1) % FLAGS.save_checkpoint_steps == 0:
                        filename = ('Model' + '_step_{:d}'.format(step + 1) + '_acc_{:.3f}'.format(acc) + '.ckpt')
                        filename = os.path.join(FLAGS.checkpoint_path, filename)
                        saver.save(sess, filename)
                        print('Write model to: {:s}'.format(filename))

                    if step % FLAGS.save_summary_steps == 0:
                        _, tl, summary_str = sess.run([train_op, total_loss, summary_op])
                        summary_writer.add_summary(summary_str, global_step=step)

                    if step % FLAGS.save_image_steps == 0:
                        log_image_name_str = ('%06d' % step)
                        img_split, seg_split, pred = sess.run([iis, isms, output_pred])

                        img_split = np.squeeze(img_split)[0]
                        seg_split = np.squeeze(seg_split)[0]
                        # img_split = img_split[0]
                        # seg_split = seg_split[0]
                        pred = np.squeeze(pred)[0]

                        color_seg = np.zeros((seg_split.shape[0], seg_split.shape[1], 3))
                        for i in range(seg_split.shape[0]):
                            for j in range(seg_split.shape[1]):
                                key = str(seg_split[i][j])
                                color_seg[i, j, :] = color_map[key]

                        color_pred = np.zeros((pred.shape[0], pred.shape[1], 3))
                        for i in range(pred.shape[0]):
                            for j in range(pred.shape[1]):
                                color_pred[i, j, :] = color_map[str(pred[i][j])]

                        write_img = np.hstack((img_split, color_seg, color_pred))
                        log_image_summary_op = sess.run(log_image, feed_dict={log_image_name: log_image_name_str, \
                                                                              log_image_data: write_img})
                        summary_writer.add_summary(log_image_summary_op, global_step=step)
                txt_record.close()

        except tf.errors.OutOfRangeError:
            txt_record.close()
            print('finish')
        finally:
            coord.request_stop()
        txt_record.close()
        coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
