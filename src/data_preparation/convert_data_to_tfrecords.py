import os
import xml.etree.ElementTree
import random

import tensorflow as tf

from src.common import consts
from .dataset import *
from src.freezing import inception
from src.common import paths
from .tf_record_utils import *

images_root_dir = os.path.join(paths.STANFORD_DS_DIR, 'Training')

def parse_image(_dir, _filename):
    path = os.path.join(images_root_dir, _dir, _filename)
    img_raw = open(path, 'rb').read()

    return img_raw

def build_stanford_example(img_raw, inception_output, one_hot_label, _feature):
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': bytes_feature(_feature.encode()),
        consts.IMAGE_RAW_FIELD: bytes_feature(img_raw),
        consts.LABEL_ONE_HOT_FIELD: float_feature(one_hot_label),
        consts.INCEPTION_OUTPUT_FIELD: float_feature(inception_output)}))

    return example

if __name__ == '__main__':
    one_hot_encoder, _ = one_hot_label_encoder()

    with tf.Graph().as_default(), \
         tf.Session().as_default() as sess, \
         tf.python_io.TFRecordWriter(paths.STANFORD_DS_TF_RECORDS,
                                    tf.python_io.TFRecordCompressionType.NONE) as writer:

        incept_model = inception.inception_model()

        def get_inception_ouput(img):
            inception_output = incept_model(sess, img).reshape(-1).tolist()
            return inception_output

        for _dir in [d for d in os.listdir(images_root_dir)]:
            print(_dir)
            listdir = os.listdir(os.path.join(images_root_dir, _dir))
            random.shuffle(listdir)
            for image_file in listdir:      
                print('-' + image_file)
                one_hot_label = one_hot_encoder([_dir]).reshape(-1).tolist()
                image = parse_image(_dir, image_file)
                example = build_stanford_example(image, get_inception_ouput(image), one_hot_label, _dir)
                writer.write(example.SerializeToString())

        writer.flush()
        writer.close()

        print('Finished')