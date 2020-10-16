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
INPUT_DIR_VAL = '/oldhome/s1500945/etri/self_driving_image/side_cam_val'
TARGET_DIR = '/oldhome/s1500945/etri/frame_label/'
FRAME_NUM = 15
CROP = (0, 0, 300, 50)
RESIZE = (0.3, 0.3)
EPOCH = 50
RESAMPLE = 3
X_PIXEL = 108
Y_PIXEL = 87

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None,  FRAME_NUM, X_PIXEL, Y_PIXEL, 3], name='x')
y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

tra_kwag = {
    'input_dir':INPUT_DIR_TRA,
    'target_dir': TARGET_DIR,
    'frame_num': FRAME_NUM,
    'crop': CROP,
    'resize':RESIZE,
    'resample':RESAMPLE}

val_kwag = {
    'input_dir':INPUT_DIR_VAL,
    'target_dir': TARGET_DIR,
    'frame_num': FRAME_NUM,
    'crop': CROP,
    'resize':RESIZE,
    'resample':RESAMPLE}

tra_provider = DataProvider.VideoProvider(batchsize = BATCH_SIZE, buffer_size = BUFFER_SIZE,  shuffle = True)
val_provider = DataProvider.VideoProvider(batchsize = BATCH_SIZE, buffer_size = 1,  shuffle = False)

filter_shape = 3
feature = 36
dense_unit = 200


def mobilenet_v1(layer, n_filter, stride=1, filtersize = (filter_shape, filter_shape), name = 'mobile_1',
                 is_train = True, reuse=False, act=tf.nn.relu6):
    with tf.variable_scope(name, reuse=reuse):
        network = tl.layers.DepthwiseConv2d(layer, shape = filtersize, strides = (1, stride, stride, 1),
                                            name=(name + '_dwise1'), act=tf.identity, padding='VALID', depth_multiplier=1)
        network = tl.layers.BatchNormLayer(network, act=act, is_train=is_train, name=(name +'_bn2'))
        network = tl.layers.Conv2d(network, n_filter = n_filter, filter_size = (1, 1), name=(name +'_project'), act=tf.identity)
        network = tl.layers.BatchNormLayer(network, act=act, is_train=is_train, name=(name +'_bn3'))
        return network


def cnn(x, act = tf.nn.selu, reuse = False, name = '3DCNN', is_train = True):
    with tf.variable_scope(name, reuse=reuse):
        network = tl.layers.InputLayer(x, name='input_layer')
        network = tl.layers.Conv3dLayer(prev_layer=network, act = tf.identity, shape=[filter_shape, filter_shape, filter_shape, 3, feature], strides=[1, 1, 1, 1,1],
                                                                     padding='VALID', name='cnn1')
        network = tl.layers.BatchNormLayer(network, name = 'bn1', act = act, is_train = is_train)
        network = tl.layers.Conv3dLayer(prev_layer=network, act = tf.identity, shape=[filter_shape, filter_shape, filter_shape, feature, feature], strides=[1, 1, 1, 1,1],
                                                                             padding='VALID', name='cnn2')
        network = tl.layers.BatchNormLayer(network, name = 'bn2', act = act, is_train = is_train)
        network = tl.layers.Conv3dLayer(prev_layer=network, act = tf.identity, shape=[filter_shape, filter_shape, filter_shape, feature, feature], strides=[1, 1, 1, 1,1],
                                                                             padding='VALID', name='cnn3')
        network = tl.layers.BatchNormLayer(network, name = 'bn3', act = act, is_train = is_train)
        network = tl.layers.Conv3dLayer(prev_layer=network, act = tf.identity, shape=[filter_shape, filter_shape, filter_shape, feature, feature], strides=[1, 1, 1, 1,1],
                                                                             padding='VALID', name='cnn4')
        network = tl.layers.BatchNormLayer(network, name = 'bn4', act = act, is_train = is_train)
        network = tl.layers.Conv3dLayer(prev_layer=network, act = tf.identity, shape=[filter_shape, filter_shape, filter_shape, feature, feature], strides=[1, 1, 1, 1,1],
                                                                             padding='VALID', name='cnn5')
        network = tl.layers.BatchNormLayer(network, name = 'bn5', act = act, is_train = is_train)
        network = tl.layers.Conv3dLayer(prev_layer=network, act = tf.identity, shape=[filter_shape, filter_shape, filter_shape, feature, feature], strides=[1, 1, 1, 1,1],
                                                                             padding='VALID', name='cnn6')
        network = tl.layers.BatchNormLayer(network, name = 'bn6', act = act, is_train = is_train)
        network = tl.layers.Conv3dLayer(prev_layer=network, act = tf.identity, shape=[filter_shape, filter_shape, filter_shape, feature, feature], strides=[1, 1, 1, 1,1],
                                                                             padding='VALID', name='cnn7')
        network = tl.layers.BatchNormLayer(network, name = 'bn7', act = act, is_train = is_train)
        network = tl.layers.ReshapeLayer(network, [-1, X_PIXEL-(filter_shape-1)*7, Y_PIXEL-(filter_shape-1)*7, feature], name='reshape')
        network =  mobilenet_v1(network, feature, name='mobile8')
        network =  mobilenet_v1(network, feature, name='mobile9')
        network =  mobilenet_v1(network, feature, name='mobile10')
        network =  mobilenet_v1(network, feature, name='mobile11')
        network =  mobilenet_v1(network, feature, name='mobile12')
        network =  mobilenet_v1(network, feature, name='mobile13')
        network =  mobilenet_v1(network, feature, name='mobile14')
        network =  mobilenet_v1(network, feature, name='mobile15')
        network =  mobilenet_v1(network, feature, name='mobile16')
        network =  mobilenet_v1(network, feature, name='mobile17')
        network =  mobilenet_v1(network, feature, name='mobile18')
        network =  mobilenet_v1(network, feature, name='mobile19')
        network =  mobilenet_v1(network, feature, name='mobile20')
        network = tl.layers.FlattenLayer(network)
        network = tl.layers.DenseLayer(network, n_units=dense_unit)
        network = tl.layers.DenseLayer(network, n_units=9, name = 'softmax')
    return network


network = cnn(x)
y =  network.outputs

cost = tl.cost.cross_entropy(y, y_, 'cost')
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# define the optimizer
train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999,
                            epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

# initialize all variables in the session

tl.layers.initialize_global_variables(sess)

nf.customfit(sess = sess, network = network, cost = cost, train_op = train_op, tra_provider = tra_provider, x = x, y_ = y_, acc=acc, n_epoch=EPOCH,
              print_freq=1, val_provider=val_provider, save_model=-100, tra_kwag=tra_kwag, val_kwag=val_kwag ,
              save_path='/oldhome/s1500945/etri/model/mobile_v1_20', epoch_identifier=None, baseline = 100000)
# sess.close()
