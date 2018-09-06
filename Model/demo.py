import re
import cv2
import time
import os,shutil
import numpy as np
import tensorflow as tf
from sklearn import metrics
slim = tf.contrib.slim
import sys
sys.path.append(os.getcwd())
sys.path.append(r'/media/cugxyy/c139cfbf-11c3-4275-9602-b96afc28d10c/DL/Road_Segmentation')


import Model.GL_Dense_U_Net as model

from matplotlib import pyplot as plt
from utils.pascal_voc import pascal_segmentation_lut
from utils.visualization import visualize_segmentation_adaptive

tf.app.flags.DEFINE_string('test_data_path', r'/home/cugxyy/DL/test/img/', '')
tf.app.flags.DEFINE_string('target_data_path', r'/home/cugxyy/DL/test/gt/', '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_integer('num_classes', 2, '')
tf.app.flags.DEFINE_string('checkpoint_path', r'../checkpoints_3/', '')
tf.app.flags.DEFINE_string('result_path', '/home/cugxyy/DL/test/result/', '')

FLAGS = tf.app.flags.FLAGS

def cal_acc_recall(pred_image, annotation):
    pred_img = pred_image.flatten()
    annotation_img = annotation.flatten()
    confusion_matrix = metrics.confusion_matrix(annotation_img, pred_img)
    mean_acc = np.mean(metrics.precision_score(annotation_img, pred_img, average=None))
    acc_for_1 = metrics.precision_score(annotation_img, pred_img, average=None)[1]
    recallfor1 = metrics.recall_score(annotation_img, pred_img, average=None)[1]
    # recall_for_1 = confusion_matrix[1][1]/np.sum(confusion_matrix, axis=1)[1]
    acc_for_0 = metrics.precision_score(annotation_img, pred_img, average=None)[0]
    # recall_for_0 = confusion_matrix[0][0] / np.sum(confusion_matrix, axis=1)[0]
    recallfor0 = metrics.recall_score(annotation_img, pred_img, average=None)[0]
    # report = metrics.classification_report(annotation_img, pred_img)
    # print(confusion_matrix)
    # print(report)
    return mean_acc, acc_for_1, recallfor1, acc_for_0, recallfor0


def get_images():
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG', 'tiff']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print ('Find {} images'.format(len(files)))
    return files

def resize_image(im, size=32, max_side_len=2400):
    h, w, _ = im.shape
    resize_w = w
    resize_h = h
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)
    resize_h = resize_h if resize_h % size == 0 else (resize_h // size) * size
    resize_w = resize_w if resize_w % size == 0 else (resize_w // size) * size
    im = cv2.resize(im, (int(resize_w), int(resize_h)))
    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)
    return im, (ratio_h, ratio_w)

def main(argv=None):
    if os.path.exists(FLAGS.result_path):
        shutil.rmtree(FLAGS.result_path)
    os.makedirs(FLAGS.result_path)
    # with tf.Session() as sess:
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    pascal_voc_lut = pascal_segmentation_lut()

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, 256, 256, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        with tf.name_scope('model_%d' % 0) as scope:
            # with tf.name_scope('model_%d' % 0) as scope:
            with tf.variable_scope(tf.get_variable_scope(), reuse=None):
                logits, conv_first = model.model(input_images, training=False)
                # vgg32 = model.FCN8VGG()
                # logits = vgg32.build(input_images, train=False, num_classes=2)
            pred = tf.argmax(logits, dimension=3)

        #
        # with tf.variable_scope(tf.get_variable_scope(), reuse=None):
        #     logits = model.model(input_images, training=True)
        # pred = tf.argmax(logits, dimension=3)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        #saver = tf.train.Saver()

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            #gv = tf.global_variables()
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn_list = get_images()
            save_str = ''
            total_acc_for_1 = 0
            total_recall_for_1 = 0
            total_acc_for_0 = 0
            total_recall_for_0 = 0
            all_mean_acc = 0
            for im_fn in im_fn_list:
                im = cv2.imread(im_fn, flags=-1)
                im_resized, (ratio_h, ratio_w) = resize_image(im, size=256)

                start = time.time()
                pred_re = sess.run([pred], feed_dict={input_images: [im]})
                # pred_re, first_conv = sess.run([pred, conv_first], feed_dict={input_images: [im]})
                # conv1_transpose = sess.run(tf.transpose(first_conv, [3, 0, 1, 2]))
                # fig3, ax3 = plt.subplots(nrows=1, ncols=1)
                # for i in range(48):
                #     ax3.imshow(conv1_transpose[i][0])
                #     plt.title('first conv')
                #     plt.savefig('test{0}.png'.format(i))
                # plt.show()

                pred_re = np.array(np.squeeze(pred_re))
                str_temp = os.path.basename(im_fn)
                taget_re = cv2.imread(os.path.join(FLAGS.target_data_path, (str_temp.split('.')[0]+r'.bmp')), flags=-1)//255
                mean_acc, acc_for_1, recall_for_1, acc_for_0, recall_for_0 = cal_acc_recall(pred_re, taget_re)
                total_acc_for_1 += acc_for_1
                total_recall_for_1 += recall_for_1
                total_acc_for_0 += acc_for_0
                total_recall_for_0 += recall_for_0
                all_mean_acc += mean_acc
                save_re = pred_re * 255
                cv2.imwrite(os.path.join(FLAGS.result_path, (str_temp.split('.')[0]+r'.bmp')), save_re)

                img=visualize_segmentation_adaptive(pred_re, pascal_voc_lut)
                _diff_time = time.time() - start
                cv2.imwrite(os.path.join(FLAGS.result_path, r'orginal_seg_' + os.path.basename(im_fn)), img)
                save_str += str_temp[0:len(str_temp)-1] + ' mean_acc:%f ,road accuracy:%f, road recall:%f, background accuracy:%f,background recall:%f' \
                                                          % (mean_acc, acc_for_1, recall_for_1, acc_for_0, recall_for_0) + '\n'

                print('{}: cost {:.0f}ms mean_acc:{:.4f}, acc_for_road:{:.4f}, recall for road:{:.4f}!'
                      .format(im_fn, _diff_time * 1000, mean_acc, acc_for_1, recall_for_1))
            picture_num = len(im_fn_list)
            all_mean_acc_1 = total_acc_for_1/picture_num
            all_mean_recall_1 = total_recall_for_1/picture_num
            all_mean_acc_0 = total_acc_for_0 / picture_num
            all_mean_recall_0 = total_recall_for_0 / picture_num
            oa_value = all_mean_acc / picture_num
            f1_score_1 = (2*all_mean_acc_1*all_mean_recall_1)/(all_mean_acc_1+all_mean_recall_1)
            f1_score_0 = (2 * all_mean_acc_0 * all_mean_recall_0) / (all_mean_recall_0 + all_mean_acc_0)
            print('handle {} pictures, the average acc for road:{:.4f}, the average recall for road:{:.4f}:,the f1 score:{:.4f}'
                  .format(picture_num, all_mean_acc_1, all_mean_recall_1, f1_score_1))
            print(
                'handle {} pictures, the average acc for background:{:.4f}, the average recall for background:{:.4f}:,the f1 score:{:.4f}'
                .format(picture_num, all_mean_acc_0, all_mean_recall_0, f1_score_0))
            save_str += 'handle {} pictures, OA:{:.4f}, the average acc for road:{:.4f}, the average recall for road:{:.4f},the f1 score:{:.4f}:'\
                            .format(picture_num, oa_value, all_mean_acc_1, all_mean_recall_1, f1_score_1) + '\n'

            save_str += 'handle {} pictures, OA:{:.4f}, the average acc for background:{:.4f}, the average recall for background:{:.4f},the f1 score:{:.4f}:' \
                            .format(picture_num, oa_value, all_mean_acc_0, all_mean_recall_0, f1_score_0) + '\n'
            if not os.path.exists(FLAGS.result_path + 'acc.txt'):
                os.mknod(FLAGS.result_path + 'acc.txt')
            with open(FLAGS.result_path + 'acc.txt', 'w') as fp:
                fp.writelines(save_str)



if __name__ == '__main__':
    tf.app.run()