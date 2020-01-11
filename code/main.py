from model import Model

import numpy as np
import tensorflow as tf

import pprint
import os

flags = tf.app.flags
flags.DEFINE_integer("epoch", 1500, "Number of epoch [1500]")
flags.DEFINE_integer("image_height", 64, "The image height of training stage [64]")
flags.DEFINE_integer("image_width", 64, "The image width of training stage [64]")

flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")

flags.DEFINE_integer("stride", 60, "The size of stride to apply input image [60]")
flags.DEFINE_integer("batch_size", 16, "The batch size of training stage [32]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("train_dir", "dataset/train/", "The directory of train input image")
flags.DEFINE_string("val_dir", "dataset/val/", "The directory of val label image")
flags.DEFINE_string("dataset_dir", "dataset/", "The directory of dataset")
flags.DEFINE_string("train_h5_name", "train.h5", "Name of train h5 file")
flags.DEFINE_string("val_h5_name", "val.h5", "Name of val h5 file")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
flags.DEFINE_string("test_dir", "test", "Name of sample directory [sample]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")
FLAGS = flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main(_):
    if not os.path.exists(FLAGS.checkpoint_dir):
      os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
      os.makedirs(FLAGS.sample_dir)

    with tf.Session() as sess:
        model = Model(sess,
                      batch_size=FLAGS.batch_size,
                      image_height=FLAGS.image_height,
                      image_width=FLAGS.image_width,
                      c_dim=FLAGS.c_dim,
                      checkpoint_dir=FLAGS.checkpoint_dir,
                      sample_dir=FLAGS.sample_dir)
    
        if FLAGS.is_train:
            model.train(FLAGS)
    
if __name__ == '__main__':
    tf.app.run()
