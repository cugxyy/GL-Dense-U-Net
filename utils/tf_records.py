# Important: We are using PIL to read .png files later.
# This was done on purpose to read indexed png files
# in a special way -- only indexes and not map the indexes
# to actual rgb values. This is specific to PASCAL VOC
# dataset data. If you don't want thit type of behaviour
# consider using skimage.io.imread()
from PIL import Image
import numpy as np

import tensorflow as tf
import cv2
import os
target_dir = r'/home/xu/PycharmProjects/Road_Segmentation/Contour_detection_data/all_train_data/label'
img_dir = r'/home/xu/PycharmProjects/Road_Segmentation/Contour_detection_data/all_train_data/image'


# Helper functions for defining tf types
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_image_annotation_pairs_to_tfrecord(tfrecords_filename):
    """Writes given image/annotation pairs to the tfrecords file.
    The function reads each image/annotation pair given filenames
    of image and respective annotation and writes it to the tfrecord
    file.
    Parameters
    ----------
    filename_pairs : array of tuples (img_filepath, annotation_filepath)
        Array of tuples of image/annotation filenames
    tfrecords_filename : string
        Tfrecords filename to write the image/annotation pairs
    """
    work_path = os.path.abspath('.')
    work_path = os.path.join(work_path, 'cut_data')
    # work_path = os.path.join(work_path, 'train')
    filename_pairs = get_file_path_2()
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    num_img = 0

    for img_path, annotation_path in filename_pairs:

        img = np.array(cv2.imread(img_path, flags=-1))
        annotation = np.array(cv2.imread(annotation_path, flags=-1))
        # Unomment this one when working with surgical data
        # annotation = annotation[:, :, 0]

        # The reason to store image sizes was demonstrated
        # in the previous example -- we have to know sizes
        # of images to later read raw serialized string,
        # convert to 1d array and convert to respective
        # shape that image used to have.
        annotation_array = np.zeros(shape=(256, 256))
        mask = (annotation[:, :] > 200)
        annotation_array[mask] = 1
        annotation_array = annotation_array.astype(np.uint8)
        height = 256
        width = 256
        img_raw = img.tostring()
        annotation_raw = annotation_array.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(img_raw),
            'mask_raw': _bytes_feature(annotation_raw)}))

        writer.write(example.SerializeToString())
        num_img += 1
        '''
        for i in range(img.shape[0]//256):
            for j in range(img.shape[0]//256):
                annotation_array = annotation[i*256:(i+1)*256, j*256:(j+1)*256]//255
                count = 0
                for ii in range(256):
                    for jj in range(256):
                        if annotation_array[ii, jj] >= 0.5:
                            count += 1
                if count/65536 >= 0.05:
                    img_array = img[i*256:(i+1)*256, j*256:(j+1)*256]
                else:
                    continue

                height = 256
                width = 256


                img_raw = img_array.tostring()
                annotation_raw = annotation_array.tostring()

                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(height),
                    'width': _int64_feature(width),
                    'image_raw': _bytes_feature(img_raw),
                    'mask_raw': _bytes_feature(annotation_raw)}))

                writer.write(example.SerializeToString())
                num_img += 1'''

    writer.close()
    print('Create %d images!'%num_img)


def read_image_annotation_pairs_from_tfrecord(tfrecords_filename):
    """Return image/annotation pairs from the tfrecords file.
    The function reads the tfrecords file and returns image
    and respective annotation matrices pairs.
    Parameters
    ----------
    tfrecords_filename : string
        filename of .tfrecords file to read from
    
    Returns
    -------
    image_annotation_pairs : array of tuples (img, annotation)
        The image and annotation that were read from the file
    """
    
    image_annotation_pairs = []

    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

    for string_record in record_iterator:

        example = tf.train.Example()
        example.ParseFromString(string_record)

        height = int(example.features.feature['height']
                                     .int64_list
                                     .value[0])

        width = int(example.features.feature['width']
                                    .int64_list
                                    .value[0])

        img_string = (example.features.feature['image_raw']
                                      .bytes_list
                                      .value[0])

        annotation_string = (example.features.feature['mask_raw']
                                    .bytes_list
                                    .value[0])

        img_1d = np.fromstring(img_string, dtype=np.uint8)
        img = img_1d.reshape((height, width, -1))

        annotation_1d = np.fromstring(annotation_string, dtype=np.uint8)

        # Annotations don't have depth (3rd dimension)
        # TODO: check if it works for other datasets
        annotation = annotation_1d.reshape((height, width))

        image_annotation_pairs.append((img, annotation))
    
    return image_annotation_pairs


def read_tfrecord_and_decode_into_image_annotation_pair_tensors(tfrecord_filenames_queue):
    """Return image/annotation tensors that are created by reading tfrecord file.
    The function accepts tfrecord filenames queue as an input which is usually
    can be created using tf.train.string_input_producer() where filename
    is specified with desired number of epochs. This function takes queue
    produced by aforemention tf.train.string_input_producer() and defines
    tensors converted from raw binary representations into
    reshaped image/annotation tensors.
    Parameters
    ----------
    tfrecord_filenames_queue : tfrecord filename queue
        String queue object from tf.train.string_input_producer()
    
    Returns
    -------
    image, annotation : tuple of tf.int32 (image, annotation)
        Tuple of image/annotation tensors
    """
    
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(tfrecord_filenames_queue)

    features = tf.parse_single_example(
      serialized_example,
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
        'mask_raw': tf.FixedLenFeature([], tf.string)
        })

    
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    annotation = tf.decode_raw(features['mask_raw'], tf.uint8)
    
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    
    #image_shape = tf.pack([height, width, 3])
    image_shape = tf.stack([height, width, 3])
    # The last dimension was added because
    # the tf.resize_image_with_crop_or_pad() accepts tensors
    # that have depth. We need resize and crop later.
    # TODO: See if it is necessary and probably remove third
    # dimension
    #annotation_shape = tf.pack([height, width, 1])
    annotation_shape = tf.stack([height, width, 1])
    image = tf.reshape(image, image_shape)
    annotation = tf.reshape(annotation, annotation_shape)
    
    return image, annotation


def get_file_path(dir_path):
    file_path = []
    input_path = os.path.join(dir_path, r'Input')
    target_path = os.path.join(dir_path, r'Target')
    for _, _, filenames in os.walk(input_path):
        for file in filenames:
            tempfile = file[0:len(file)-1]
            if os.path.exists(os.path.join(target_path, tempfile)):
                temp_pair = []
                input_file = os.path.join(input_path, file)
                target_file = os.path.join(target_path, tempfile)
                temp_pair.append(input_file)
                temp_pair.append(target_file)
                file_path.append(temp_pair)
    print('find:%d images!'%(len(file_path)))
    return file_path

def get_file_path_2():
    file_path = []
    input_path = img_dir
    target_path = target_dir
    for _, _, filenames in os.walk(target_path):
        for file in filenames:

            if os.path.exists(os.path.join(input_path, file)):
                temp_pair = []
                input_file = os.path.join(input_path, file)
                target_file = os.path.join(target_path, file)
                temp_pair.append(input_file)
                temp_pair.append(target_file)
                file_path.append(temp_pair)
    print('find:%d images!'%(len(file_path)))
    return file_path