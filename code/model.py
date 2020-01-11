import time
import os

from collections import OrderedDict

from layers import *
from utils import *

import numpy as np
import tensorflow as tf

class Model(object):
    def __init__(self,
                 sess,
                 batch_size=32,
                 image_height=64,
                 image_width=64,
                 c_dim=1,
                 checkpoint_dir=None,
                 sample_dir=None):

        self.sess = sess
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.c_dim = c_dim
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir

        self.build_model()


    def build_model(self):
        self.inputs = tf.placeholder(tf.float32, [None, None, None, self.c_dim], name='images')
        self.labels = tf.placeholder(tf.float32, [None, None, None, self.c_dim], name='labels')
    
        self.pred = self.model(self.inputs)

        # Loss function (L1 loss)
        self.psnr = tf.reduce_mean(tf.image.psnr(self.labels, self.pred, max_val=1.0))
        self.loss = tf.reduce_mean(tf.abs(self.labels - self.pred))
        self.saver = tf.train.Saver()

    def train(self, config):
        # Prepare train and val h5 file
        if not os.path.exists(os.path.join(os.getcwd(), config.dataset_dir, config.train_h5_name)):
            input_setup(self.sess, config, 'Train')
        if not os.path.exists(os.path.join(os.getcwd(), config.dataset_dir, config.val_h5_name)):
            input_setup(self.sess, config, 'Val')

        # read h5 file and training
        train_dir = os.path.join('./{}'.format(config.dataset_dir), "train.h5")
        val_dir = os.path.join('./{}'.format(config.dataset_dir), "val.h5")

        train_data, train_label = read_data(train_dir)
        val_data, val_label = read_data(val_dir)

        self.train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.loss)

        self.sess.run(tf.initialize_all_variables())
    
        counter = 0
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        if config.is_train:
            print("Training...")

        for ep in range(config.epoch):
            batch_train_idxs = len(train_data) // config.batch_size
            batch_val_idxs = len(train_data) // config.batch_size
            batch_images = np.zeros([config.batch_size, config.image_height, config.image_width, 1], dtype=np.float)
            batch_labels = np.zeros([config.batch_size, config.image_height, config.image_width, 1], dtype=np.float)
            for idx in range(0, batch_train_idxs):
                batch_images = train_data[idx*config.batch_size : (idx+1)*config.batch_size]
                batch_labels = train_label[idx*config.batch_size : (idx+1)*config.batch_size]

                counter += 1
                _, err, psnr = self.sess.run([self.train_op, self.loss, self.psnr], \
                                             feed_dict={self.inputs: batch_images, \
                                                        self.labels: batch_labels})

                if counter % 10 == 0:
                    print("Epoch: [%d], Counter: [%d/%d], Time: [%4.4f], Loss: [%4.4f] Train-Psnr: [%4.4f]" \
                           % ((ep+1), counter, batch_train_idxs, time.time()-start_time, err, psnr))

                if counter % 500 == 0:
                    psnr = 0
                    psnr_ave = 0
                    val_batch_images = np.zeros([config.batch_size, config.image_height, config.image_width, 1], dtype=np.float)
                    val_batch_labels = np.zeros([config.batch_size, config.image_height, config.image_width, 1], dtype=np.float)
                    for idx in range(0, batch_val_idxs):
                        val_batch_images = val_data[idx * config.batch_size: (idx + 1) * config.batch_size]
                        val_batch_labels = val_label[idx * config.batch_size: (idx + 1) * config.batch_size]
                        val_psnr = self.sess.run(self.psnr, feed_dict={self.inputs: val_batch_images, self.labels: val_batch_labels})
                        psnr += val_psnr

                    psnr_ave = psnr / batch_val_idxs
                    print("Epoch: [%d], Val-Psnr: [%4.4f]" %  (ep, psnr_ave))
                    self.save(config.checkpoint_dir, counter)

                    ### extract a validate to view the middle result and compare
                    eval_image = train_data[idx:idx+1]
                    eval_label = train_label[idx:idx+1]
                    eval_pred = self.sess.run(self.pred, feed_dict={self.inputs: eval_image, self.labels: eval_label})
                    save_images(os.path.join(self.sample_dir, 'eval_%d_%d.png' % (counter, ep)), \
                                eval_image, eval_label, eval_pred)

    def model(self, input_, channels=1, layers=3, features_root=16, filter_size=3, pool_size=2):
        pools = OrderedDict()
        deconv = OrderedDict()
        dw_h_convs = OrderedDict()
        up_h_convs = OrderedDict()
        
        in_node = input_

        # down layers
        for layer in range(0, layers):
            with tf.name_scope("down_conv_{}".format(str(layer))):
                features = 2 ** layer * features_root
                stddev = np.sqrt(2 / (filter_size ** 2 * features))
                if layer == 0:
                    w1 = weight_variable([filter_size, filter_size, channels, features], stddev, name="w1")
                else:
                    w1 = weight_variable([filter_size, filter_size, features // 2, features], stddev, name="w1")

                w2 = weight_variable([filter_size, filter_size, features, features], stddev, name="w2")
                b1 = bias_variable([features], name="b1")
                b2 = bias_variable([features], name="b2")

                conv1 = conv2d(in_node, w1, b1)
                tmp_h_conv = tf.nn.relu(conv1)
                conv2 = conv2d(tmp_h_conv, w2, b2)
                dw_h_convs[layer] = tf.nn.relu(conv2)

                if layer < layers - 1:
                    pools[layer] = max_pool(dw_h_convs[layer], pool_size)
                    in_node = pools[layer]

        in_node = dw_h_convs[layers - 1]

        # up layers
        for layer in range(layers - 2, -1, -1):
            with tf.name_scope("up_conv_{}".format(str(layer))):
                features = 2 ** (layer + 1) * features_root
                stddev = np.sqrt(2 / (filter_size ** 2 * features))

                wd = weight_variable_devonc([pool_size, pool_size, features // 2, features], stddev, name="wd")
                bd = bias_variable([features // 2], name="bd")
                h_deconv = tf.nn.relu(deconv2d(in_node, wd, pool_size) + bd)
                h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)
                deconv[layer] = h_deconv_concat

                w1 = weight_variable([filter_size, filter_size, features, features // 2], stddev, name="w1")
                w2 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w2")
                b1 = bias_variable([features // 2], name="b1")
                b2 = bias_variable([features // 2], name="b2")

                conv1 = conv2d(h_deconv_concat, w1, b1)
                h_conv = tf.nn.relu(conv1)
                conv2 = conv2d(h_conv, w2, b2)
                in_node = tf.nn.relu(conv2)
                up_h_convs[layer] = in_node

        # Output Map
        with tf.name_scope("output_map"):
            weight = weight_variable([1, 1, features_root, channels], stddev)
            bias = bias_variable([channels], name="bias")
            conv = conv2d(in_node, weight, bias)
            output_map = tf.nn.relu(conv)

        return output_map
    

    def save(self, checkpoint_dir, step):
        model_name = "UNET.model"
        model_dir = "%s_%s_%s" % ("unet", self.image_height, self.image_width)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s_%s" % ("unet", self.image_height, self.image_width)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

