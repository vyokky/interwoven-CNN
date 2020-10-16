import sys
import os
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import DataProvider
import NetFlow as nf

BATCH_SIZE = 50
BUFFER_SIZE = 5
INPUT_DIR_TRA = '/oldhome/s1500945/etri/self_driving_image/side_cam_tra'
INPUT_DIR_TRA2 = '/oldhome/s1500945/etri/self_driving_image/front_cam_tra'
INPUT_DIR_VAL = '/oldhome/s1500945/etri/self_driving_image/side_cam_val'
INPUT_DIR_VAL2 = '/oldhome/s1500945/etri/self_driving_image/front_cam_val'
TARGET_DIR = '/oldhome/s1500945/etri/frame_label/'


FRAME_NUM = 15
CROP = (0, 0, 300, 50)
RESIZE = (0.3, 0.3)
CROP2 = (0, 0, 100, 150)
RESIZE2 = (0.15, 0.15)
EPOCH = 50
RESAMPLE = 3
X_PIXEL = 108
Y_PIXEL = 87
X_PIXEL2 = 54
Y_PIXEL2 = 58

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None,  FRAME_NUM, X_PIXEL, Y_PIXEL, 3], name='x')
x2 = tf.placeholder(tf.float32, shape=[None,  FRAME_NUM, X_PIXEL2, Y_PIXEL2, 3], name='x2')
y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

tra_kwag = {
    'input_dir':INPUT_DIR_TRA,
    'input_dir2':INPUT_DIR_TRA2,
    'target_dir': TARGET_DIR,
    'frame_num': FRAME_NUM,
    'crop': CROP,
    'resize':RESIZE,
    'resample':RESAMPLE,
    'resize2':RESIZE2,
    'crop2':CROP2}

val_kwag = {
    'input_dir':INPUT_DIR_VAL,
    'input_dir2':INPUT_DIR_VAL2,
    'target_dir': TARGET_DIR,
    'frame_num': FRAME_NUM,
    'crop': CROP,
    'resize':RESIZE,
    'resample':RESAMPLE,
    'resize2':RESIZE2,
    'crop2':CROP2}

tra_provider = DataProvider.MultiVideoProvider(batchsize = BATCH_SIZE, buffer_size = BUFFER_SIZE,  shuffle = True)
val_provider = DataProvider.MultiVideoProvider(batchsize = BATCH_SIZE, buffer_size = 1,  shuffle = False)

filter_shape = 3
filter_shape_s = 2
filter_shape_l = 4
feature = 36
dense_unit = 200
merge_x = 47
merge_y = 37


def stream1(x, act=tf.nn.selu, reuse=False, name='stream1', is_train=True):
    with tf.variable_scope(name, reuse=reuse):
        network = tl.layers.InputLayer(x, name='input_layer')
        network = tl.layers.Conv3dLayer(prev_layer=network, act=tf.identity,
                                        shape=[filter_shape, filter_shape, filter_shape, 3, feature],
                                        strides=[1, 1, 1, 1, 1],
                                        padding='VALID', name='cnn1')
        network = tl.layers.BatchNormLayer(network, name='bn1', act=act, is_train=is_train)
        network = tl.layers.Conv3dLayer(prev_layer=network, act=tf.identity,
                                        shape=[filter_shape, filter_shape, filter_shape, feature, feature],
                                        strides=[1, 1, 1, 1, 1],
                                        padding='VALID', name='cnn2')
        network = tl.layers.BatchNormLayer(network, name='bn2', act=act, is_train=is_train)
        network = tl.layers.Conv3dLayer(prev_layer=network, act=tf.identity,
                                        shape=[filter_shape, filter_shape, filter_shape, feature, feature],
                                        strides=[1, 1, 1, 1, 1],
                                        padding='VALID', name='cnn3')
        network = tl.layers.BatchNormLayer(network, name='bn3', act=act, is_train=is_train)
        network = tl.layers.Conv3dLayer(prev_layer=network, act=tf.identity,
                                        shape=[filter_shape, filter_shape, filter_shape, feature, feature],
                                        strides=[1, 1, 1, 1, 1],
                                        padding='VALID', name='cnn4')
        network = tl.layers.BatchNormLayer(network, name='bn4', act=act, is_train=is_train)
        network = tl.layers.Conv3dLayer(prev_layer=network, act=tf.identity,
                                        shape=[filter_shape, filter_shape, filter_shape, feature, feature],
                                        strides=[1, 1, 1, 1, 1],
                                        padding='VALID', name='cnn5')
        network = tl.layers.BatchNormLayer(network, name='bn5', act=act, is_train=is_train)
        network = tl.layers.Conv3dLayer(prev_layer=network, act=tf.identity,
                                        shape=[filter_shape, filter_shape, filter_shape, feature, feature],
                                        strides=[1, 1, 1, 1, 1],
                                        padding='VALID', name='cnn6')
        network = tl.layers.BatchNormLayer(network, name='bn6', act=act, is_train=is_train)
        network = tl.layers.Conv3dLayer(prev_layer=network, act=tf.identity,
                                        shape=[filter_shape, filter_shape, filter_shape, feature, feature],
                                        strides=[1, 1, 2, 2, 1],
                                        padding='VALID', name='cnn7')
        network = tl.layers.BatchNormLayer(network, name='bn7', act=act, is_train=is_train)

        return network


def stream2(x, act=tf.nn.selu, reuse=False, name='stream2', is_train=True):
    with tf.variable_scope(name, reuse=reuse):
        network = tl.layers.InputLayer(x, name='input_layer')
        network = tl.layers.Conv3dLayer(prev_layer=network, act=tf.identity,
                                        shape=[filter_shape, filter_shape_s, filter_shape_l, 3, feature],
                                        strides=[1, 1, 1, 1, 1],
                                        padding='VALID', name='cnn1')
        network = tl.layers.BatchNormLayer(network, name='bn1', act=act, is_train=is_train)
        network = tl.layers.Conv3dLayer(prev_layer=network, act=tf.identity,
                                        shape=[filter_shape, filter_shape_s, filter_shape_l, feature, feature],
                                        strides=[1, 1, 1, 1, 1],
                                        padding='VALID', name='cnn2')
        network = tl.layers.BatchNormLayer(network, name='bn2', act=act, is_train=is_train)
        network = tl.layers.Conv3dLayer(prev_layer=network, act=tf.identity,
                                        shape=[filter_shape, filter_shape_s, filter_shape_l, feature, feature],
                                        strides=[1, 1, 1, 1, 1],
                                        padding='VALID', name='cnn3')
        network = tl.layers.BatchNormLayer(network, name='bn3', act=act, is_train=is_train)
        network = tl.layers.Conv3dLayer(prev_layer=network, act=tf.identity,
                                        shape=[filter_shape, filter_shape_s, filter_shape_l, feature, feature],
                                        strides=[1, 1, 1, 1, 1],
                                        padding='VALID', name='cnn4')
        network = tl.layers.BatchNormLayer(network, name='bn4', act=act, is_train=is_train)
        network = tl.layers.Conv3dLayer(prev_layer=network, act=tf.identity,
                                        shape=[filter_shape, filter_shape_s, filter_shape_l, feature, feature],
                                        strides=[1, 1, 1, 1, 1],
                                        padding='VALID', name='cnn5')
        network = tl.layers.BatchNormLayer(network, name='bn5', act=act, is_train=is_train)
        network = tl.layers.Conv3dLayer(prev_layer=network, act=tf.identity,
                                        shape=[filter_shape, filter_shape_s, filter_shape_l, feature, feature],
                                        strides=[1, 1, 1, 1, 1],
                                        padding='VALID', name='cnn6')
        network = tl.layers.BatchNormLayer(network, name='bn6', act=act, is_train=is_train)
        network = tl.layers.Conv3dLayer(prev_layer=network, act=tf.identity,
                                        shape=[filter_shape, filter_shape_s, filter_shape_l, feature, feature],
                                        strides=[1, 1, 1, 1, 1],
                                        padding='VALID', name='cnn7')
        network = tl.layers.BatchNormLayer(network, name='bn7', act=act, is_train=is_train)

        return network


def combine(l1, l2, act=tf.nn.selu, reuse=False, name='combine', is_train=True):
    with tf.variable_scope(name, reuse=reuse):
        network = tl.layers.ConcatLayer([l1, l2], -1, name='concat_layer')
        network = tl.layers.ReshapeLayer(network, [-1, merge_x, merge_y, feature * 2], name='reshape')
        network = tl.layers.Conv2dLayer(prev_layer=network, act=tf.identity,
                                        shape=[filter_shape, filter_shape, feature * 2, feature], strides=[1, 1, 1, 1],
                                        padding='VALID', name='cnn8')
        network = tl.layers.BatchNormLayer(network, name='bn8', act=act, is_train=is_train)
        network = tl.layers.Conv2dLayer(prev_layer=network, act=tf.identity,
                                        shape=[filter_shape, filter_shape, feature, feature], strides=[1, 1, 1, 1],
                                        padding='VALID', name='cnn9')
        network = tl.layers.BatchNormLayer(network, name='bn9', act=act, is_train=is_train)
        network = tl.layers.Conv2dLayer(prev_layer=network, act=tf.identity,
                                        shape=[filter_shape, filter_shape, feature, feature], strides=[1, 1, 1, 1],
                                        padding='VALID', name='cnn10')
        network = tl.layers.BatchNormLayer(network, name='bn10', act=act, is_train=is_train)
        network = tl.layers.Conv2dLayer(prev_layer=network, act=tf.identity,
                                        shape=[filter_shape, filter_shape, feature, feature], strides=[1, 1, 1, 1],
                                        padding='VALID', name='cnn11')
        network = tl.layers.BatchNormLayer(network, name='bn11', act=act, is_train=is_train)
        network = tl.layers.Conv2dLayer(prev_layer=network, act=tf.identity,
                                        shape=[filter_shape, filter_shape, feature, feature], strides=[1, 1, 1, 1],
                                        padding='VALID', name='cnn12')
        network = tl.layers.BatchNormLayer(network, name='bn12', act=act, is_train=is_train)
        network = tl.layers.Conv2dLayer(prev_layer=network, act=tf.identity,
                                        shape=[filter_shape, filter_shape, feature, feature], strides=[1, 1, 1, 1],
                                        padding='VALID', name='cnn13')
        network = tl.layers.BatchNormLayer(network, name='bn13', act=act, is_train=is_train)
        network = tl.layers.Conv2dLayer(prev_layer=network, act=tf.identity,
                                        shape=[filter_shape, filter_shape, feature, feature], strides=[1, 1, 1, 1],
                                        padding='VALID', name='cnn14')
        network = tl.layers.BatchNormLayer(network, name='bn14', act=act, is_train=is_train)
        network = tl.layers.Conv2dLayer(prev_layer=network, act=tf.identity,
                                        shape=[filter_shape, filter_shape, feature, feature], strides=[1, 1, 1, 1],
                                        padding='VALID', name='cnn15')
        network = tl.layers.BatchNormLayer(network, name='bn15', act=act, is_train=is_train)
        network = tl.layers.Conv2dLayer(prev_layer=network, act=tf.identity,
                                        shape=[filter_shape, filter_shape, feature, feature], strides=[1, 1, 1, 1],
                                        padding='VALID', name='cnn16')
        network = tl.layers.BatchNormLayer(network, name='bn16', act=act, is_train=is_train)
        network = tl.layers.Conv2dLayer(prev_layer=network, act=tf.identity,
                                        shape=[filter_shape, filter_shape, feature, feature], strides=[1, 1, 1, 1],
                                        padding='VALID', name='cnn17')
        network = tl.layers.BatchNormLayer(network, name='bn17', act=act, is_train=is_train)
        network = tl.layers.Conv2dLayer(prev_layer=network, act=tf.identity,
                                        shape=[filter_shape, filter_shape, feature, feature], strides=[1, 1, 1, 1],
                                        padding='VALID', name='cnn18')
        network = tl.layers.BatchNormLayer(network, name='bn18', act=act, is_train=is_train)
        network = tl.layers.Conv2dLayer(prev_layer=network, act=tf.identity,
                                        shape=[filter_shape, filter_shape, feature, feature], strides=[1, 1, 1, 1],
                                        padding='VALID', name='cnn19')
        network = tl.layers.BatchNormLayer(network, name='bn19', act=act, is_train=is_train)
        network = tl.layers.Conv2dLayer(prev_layer=network, act=tf.identity,
                                        shape=[filter_shape, filter_shape, feature, feature], strides=[1, 1, 1, 1],
                                        padding='VALID', name='cnn20')
        network = tl.layers.BatchNormLayer(network, name='bn20', act=act, is_train=is_train)
        network = tl.layers.FlattenLayer(network)
        network = tl.layers.DenseLayer(network, n_units=dense_unit)
        network = tl.layers.DenseLayer(network, n_units=9, name='softmax')
    return network


network = combine(stream1(x), stream2(x2))
y = network.outputs


cost = tl.cost.cross_entropy(y, y_, 'cost')
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# define the optimizer
train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999,
                            epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

# initialize all variables in the session

tl.layers.initialize_global_variables(sess)

nf.customfit_2(sess = sess, network = network, cost = cost, train_op = train_op, tra_provider = tra_provider, x = x, x2 = x2, y_ = y_, acc=acc, n_epoch=EPOCH,
              print_freq=1, val_provider=val_provider, save_model=-100, tra_kwag=tra_kwag, val_kwag=val_kwag,
              save_path='/oldhome/s1500945/etri/model/cnn_20_2s', epoch_identifier=None, baseline = 100000)
# sess.close()
