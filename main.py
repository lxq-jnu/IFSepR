from __future__ import print_function
# from VGG19_trans import *
# import pydot
# import graphviz
# import keras
from tensorflow import keras
from tensorflow.python.keras import backend as K
import tensorflow as tf
import numpy as np
from PIL import Image
from scipy.misc import imread, imsave
import warnings
import h5py
import os
import time
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Flatten, Dense, Input
from tensorflow.python.keras.layers import GlobalMaxPooling2D
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.preprocessing import image
# from tensorflow.python.keras.utils import layer_utils
# from tensorflow.python.keras.utils.data_utils import get_file
# from tensorflow.python.keras import backend as K
# from tensorflow.python.keras.applications.imagenet_utils import decode_predictions
# from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
# from tensorflow.python.keras.applications.imagenet_utils import _obtain_input_shape
# from tensorflow.python.keras.engine.topology import get_source_inputs
# from pylab import *
# import matplotlib.pyplot as plt
# from tensorflow.python.keras.utils.vis_utils import plot_model
from utils import *
import scipy.io as sio
from tensorflow.python.keras.layers import TimeDistributed
from tensorflow.python.keras.layers import Input, Dense, concatenate, Add
from tensorflow.python.keras.optimizers import RMSprop, Adam
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend as K

# WEIGHTS_PATH = ''
# WEIGHTS_PATH_NO_TOP = h5py.File('vgg19.h5','r')
# myself define layer

# myself-define choose-max layer
def choosemax(x):
    output = x[0]
    for i in range(1, len(x)):
        output = K.maximum(output, x[i])

    # output=tf.zeros_like(output1)
    return output




# myself-define fangcha
def compare(x):
    # x0shape = x[0].shape
    # print(x0shape)
    [channel, chang, width, height] = x[0].shape
    print(x[0].shape, 'x[0].shape')

    # print(chang)
    # res=np.zeros((chang,width,height))

    # res=np.expand_dims(res, axis=0)
    # print(res.shape,'res.shape')
    # print(res)
    input_1 = x[0]
    print(input_1.shape, 'input_1.shape')
    input_2 = x[1]
    print(input_2.shape, 'input_2.shape')
    results = []
    for i in range(0, height):
        # con_1=K.eval(input_1[:,:,:,i])
        # input_1_reshape=K.reshape(input_1[:,:,:,i], (input_1[:,:,:,i].shape[0],,input_1[:,:,:,i].shape[IR], input_1[:,:,:,i].shape[VI]))
        # input_1_shape=input_1[:,:,:,i].shape
        # con_1 = tf.get_variable(input_1[:,:,:,i],shape=input_1_shape,dtype=dtypes.float32,trainable=False)
        # con_2=K.eval(input_2[:,:,:,i])
        # input_2_reshape=K.reshape(input_2[:,:,:,i], (input_2[:,:,:,i].shape[0],,input_2[:,:,:,i].shape[IR], input_2[:,:,:,i].shape[VI]))
        # input_2_shape=input_2[:,:,:,i].shape
        # con_2 = tf.get_variable(input_2[:,:,:,i],shape=input_2_shape,dtype=dtypes.float32,trainable=False)
        # con_2 = K.variable(value=input_2[:,:,:,i])
        # res_con=K.eval(res[:,:,:,i])
        # res_con = K.variable(value=res[:,:,:,i])
        var_1 = K.var(input_1[:, :, :, i])
        var_2 = K.var(input_2[:, :, :, i])

        def true_proc():
            result = input_1[:, :, :, i]
            return result

        def false_proc():
            result = input_2[:, :, :, i]
            return result

        results.append(tf.cond(var_1 > var_2, lambda: input_1[:, :, :, i:i + 1], lambda: input_2[:, :, :, i:i + 1]))
        # res[:,:,:,i] = tf.cond(var_1>var_2, lambda: res[:,:,:,i].assign(input_1[:,:,:,i]), lambda: res[:,:,:,i].assign(input_2[:,:,:,i]))
        # res[:,:,:,i] = tf.cond(var_1>var_2,lambda:K.update(res_con, con_1),lambda:K.update(res_con, con_2))
        # if var_1>var_2:
        #    res[:,:,i]=input_1[:,:,i]
        # else:
        #    res[:,:,i]=input_2[:,:,i]
    res = K.concatenate(results, axis=3)
    return res


def compare_output_shape(input_shape):
    return input_shape[0]


# myself-define bianyuanqiangdu
def bianyuanqiangdu(x):
    # x0shape = x[0].shape
    # print(x0shape)
    [channel, chang, width, height] = x[0].shape
    dimension = x[0].shape

    print(x[0].shape, 'x[0].shape')
    input_1 = x[0]
    print(input_1.shape, 'input_1.shape')
    input_2 = x[1]
    print(input_2.shape, 'input_2.shape')
    results = []

    for i in range(0, height):
        N = (chang * width) * (3 / 5)
        # caculate the map for source images

        # print(input_1[:,:,:,i].shape,'input_1[:,:,:,i].shape')
        # non_zero_nums=tf.count_nonzero(input_1[:,:,:,i])
        # non_zero_nums=tf.size(input_1[:,:,:,i])
        # sum=tf.reduce_sum(input_1[:,:,:,i])
        # print(tf.size(input_1[:,:,:,i]),'size')
        # var_1=K.var(input_1[:,:,:,i])
        # with tf.Session()as sess:
        #    print(sess.run(var_1))
        # print(non_zero_nums)
        # num=non_zero_nums-10000
        # K.eval(sum)
        # with tf.Session()as sess:
        #    print(sess.run(tf.Print(non_zero_nums,[non_zero_nums],"nums: ")))
        # num=tf.Print(non_zero_nums,[non_zero_nums],"nums: ")
        # print(num)
        # a=non_zero_nums
        # sess = tf.Session()
        # print(sess.run(a))
        # K.eval(non_zero_nums)

        # print(non_zero_nums,'non_zero_nums')
        # var_2=K.var(input_2[:,:,:,i])
        results.append(tf.cond(tf.count_nonzero(input_2[:, :, :, i]) < N,
                               lambda: K.maximum(input_1[:, :, :, i:i + 1], input_2[:, :, :, i:i + 1]),
                               lambda: 0.5 * input_1[:, :, :, i:i + 1] + 0.5 * input_2[:, :, :, i:i + 1]))
        # results.append(tf.cond(tf.count_nonzero(input_2[:,:,:,i])<N, lambda: K.maximum(input_1[:,:,:,i:i+IR], input_2[:,:,:,i:i+IR]), lambda: 0.7*input_1[:,:,:,i:i+IR]+0.6*input_2[:,:,:,i:i+IR]))
    # results = np.stack(results, axis=-IR)
    # res = np.reshape(results, (dimension[0], dimension[IR], dimension[VI], dimension[3]))
    res = K.concatenate(results, axis=3)

    # res=tf.zeros_like(res1)
    return res


def L1_norm(x):
    result = []
    narry_a = x[0]
    narry_b = x[1]

    dimension = x[0].shape
    print(dimension, 'dimension___shape')
    # caculate L1-norm
    temp_abs_a = tf.abs(narry_a)
    temp_abs_b = tf.abs(narry_b)
    _l1_a = tf.reduce_sum(temp_abs_a, 3)
    _l1_b = tf.reduce_sum(temp_abs_b, 3)

    # _l1_a = tf.reduce_sum(_l1_a, 0)
    # _l1_b = tf.reduce_sum(_l1_b, 0)
    l1_a = _l1_a
    l1_b = _l1_b

    # caculate the map for source images
    mask_value = l1_a + l1_b

    mask_sign_a = l1_a / mask_value
    mask_sign_b = l1_b / mask_value

    array_MASK_a = mask_sign_a
    array_MASK_b = mask_sign_b

    for i in range(dimension[3]):
        temp_matrix = array_MASK_a * narry_a[:, :, :, i] + array_MASK_b * narry_b[:, :, :, i]
        # print(temp_matrix.shape,'temp_matrix')#(?,248,316)
        result.append(temp_matrix)

    result = tf.stack(result, axis=-1)

    # resule_tf = np.reshape(result, (dimension[0], dimension[IR], dimension[VI], dimension[3]))

    return result


def test_gyh(x):
    # x0shape = x[0].shape
    # print(x0shape)
    # [channel,chang,width,height] = x[0].shape
    sz = x[0].shape
    # channel = int(sz[0])
    chang = int(sz[1])
    width = int(sz[2])
    height = int(sz[3])

    dimension = x[0].shape

    print(x[0].shape, 'x[0].shape')
    input_1 = x[0]
    print(input_1.shape, 'input_1.shape')
    input_2 = x[1]
    print(input_2.shape, 'input_2.shape')
    results = []
    # caculate L1-norm
    temp_abs_a = tf.abs(input_1)
    temp_abs_b = tf.abs(input_2)
    _l1_a = tf.reduce_sum(temp_abs_a, 3)
    _l1_b = tf.reduce_sum(temp_abs_b, 3)
    l1_a = _l1_a
    l1_b = _l1_b

    # caculate the map for source images
    mask_value = l1_a + l1_b

    mask_sign_a = l1_a / mask_value
    mask_sign_b = l1_b / mask_value

    array_MASK_a = mask_sign_a
    array_MASK_b = mask_sign_b
    ggyh = 0
    for i in range(0, height):
        # N=(chang*width)*(3/5)
        N = np.int64((chang * width))
        # ggyh = tf.reduce_mean(input_2[:,:,:,i])+tf.reduce_mean(input_1[:,:,:,i])
        # N=np.int64(N/5)
        # results.append(tf.cond(tf.count_nonzero(input_2[:,:,:,i])<N, lambda: K.maximum(input_1[:,:,:,i], input_2[:,:,:,i]), lambda: array_MASK_a*input_1[:,:,:,i] + array_MASK_b*input_2[:,:,:,i]))
        results.append(tf.cond((tf.reduce_mean(input_2[:, :, :, i]) + tf.reduce_mean(input_1[:, :, :, i])) < 3,
                               lambda: K.maximum(input_1[:, :, :, i], input_2[:, :, :, i]),
                               lambda: array_MASK_a * input_1[:, :, :, i] + array_MASK_b * input_2[:, :, :, i]))
        # with tf.Session() as sess:
        # m_a= sess.run([ggyh])
        # print(m_a)
    res = tf.stack(results, axis=-1)  # = K.concatenate(results,axis=3)
    # res=tf.zeros_like(res1)
    return res


def test_gyh2(x):
    # x0shape = x[0].shape
    # print(x0shape)
    # [channel,chang,width,height] = x[0].shape
    sz = x[0].shape
    # channel = int(sz[0])
    chang = int(sz[1])
    width = int(sz[2])
    height = int(sz[3])

    dimension = x[0].shape

    # print(x[0].shape,'x[0].shape')
    input_1 = x[0]
    # print(input_1.shape,'input_1.shape')
    input_2 = x[1]
    # print(input_2.shape,'input_2.shape')
    results = []
    # caculate L1-norm
    temp_abs_a = tf.abs(input_1)
    temp_abs_b = tf.abs(input_2)
    _l1_a = tf.reduce_sum(temp_abs_a, 3)
    _l1_b = tf.reduce_sum(temp_abs_b, 3)
    l1_a = _l1_a
    l1_b = _l1_b

    # caculate the map for source images
    mask_value = l1_a + l1_b

    mask_sign_a = l1_a / mask_value
    mask_sign_b = l1_b / mask_value

    array_MASK_a = mask_sign_a
    array_MASK_b = mask_sign_b
    ggyh = 0
    for i in range(0, height):
        # N=(chang*width)*(3/5)
        N = np.int64((chang * width))
        # ggyh = tf.reduce_mean(input_2[:,:,:,i])+tf.reduce_mean(input_1[:,:,:,i])
        # N=np.int64(N/5)
        # results.append(tf.cond(tf.count_nonzero(input_2[:,:,:,i])<N, lambda: K.maximum(input_1[:,:,:,i], input_2[:,:,:,i]), lambda: array_MASK_a*input_1[:,:,:,i] + array_MASK_b*input_2[:,:,:,i]))
        # results.append(tf.cond(tf.reduce_mean(input_2[:,:,:,i])>tf.reduce_mean(input_1[:,:,:,i]), lambda: input_2[:,:,:,i], lambda: input_1[:,:,:,i]))
        results.append(input_2[:, :, :, i] + input_1[:, :, :, i])
        # with tf.Session() as sess:
        # m_a= sess.run([ggyh])
        # print(m_a)
    res = tf.stack(results, axis=-1)  # = K.concatenate(results,axis=3)
    # res=tf.zeros_like(res1)
    return res




def difference_loss(private_samples, shared_samples, weight=1.0, name='difl'):
    # ggg = private_samples.shape[0]
    # ggg = private_samples
    # ggg1 = private_samples[IR].shape
    # print(ggg1.shape)
    sz = private_samples.shape
    height = int(sz[3])
    print(height)
    kuan = int(sz[2])
    print(kuan)
    chang = int(sz[1])
    print(chang)
    shuliang = int(chang * kuan * height)

    private_samples1 = tf.contrib.layers.flatten(private_samples)
    shared_samples1 = tf.contrib.layers.flatten(shared_samples)

    # private_samples -= tf.reduce_mean(private_samples1, 0)
    # shared_samples -= tf.reduce_mean(shared_samples1, 0)
    # private_samples = tf.nn.l2_normalize(private_samples1, IR)
    # shared_samples = tf.nn.l2_normalize(shared_samples1, IR)

    # bianhuan1 = private_samples.reshape[:,chang,kuan,height]
    # bianhuan2 = shared_samples.reshape[:,chang,kuan,height]

    # dianji = tf.multiply(bianhuan1,bianhuan2)

    dianji = tf.multiply(private_samples1, shared_samples1)
    # print(dianji.shape)
    dianji = tf.abs(dianji)

    sunshi = tf.reduce_sum(dianji, 1, keepdims=True)
    print(sunshi.shape)

    # correlation_matrix = tf.matmul( private_samples, shared_samples, name = None)
    # print(correlation_matrix.shape)
    # cost = tf.reduce_sum(tf.square(correlation_matrix)) * weight
    # cost = cost /( ggg * ggg )
    # print(cost)
    # cost = tf.where(cost > 0, cost, 0, name='value')
    # print(cost.shape)
    # tf.summary.scalar('losses/Difference Loss {}'.format(name),cost)
    # assert_op = tf.Assert(tf.is_finite(cost), [cost])
    # with tf.control_dependencies([assert_op]):
    # tf.losses.add_loss(cost)
    return sunshi


def similarity_loss(shared_samples1, shared_samples2, weight=1.0, name='samel'):
    # ggg = private_samples.shape[0]
    # ggg = private_samples
    # ggg1 = private_samples[IR].shape
    # print(ggg1.shape)
    sz2 = shared_samples1.shape
    height2 = int(sz2[3])
    print(height2)
    kuan2 = int(sz2[2])
    print(kuan2)
    chang2 = int(sz2[1])
    print(chang2)
    shuliang2 = int(chang2 * kuan2 * height2)
    input1 = shared_samples1
    input2 = shared_samples2
    ##############
    # sameres = 0
    shared_samples1 = tf.contrib.layers.flatten(shared_samples1)
    shared_samples2 = tf.contrib.layers.flatten(shared_samples2)
    # shared_samples1 = tf.abs(shared_samples1)
    # shared_samples2 = tf.abs(shared_samples2)

    shared_value = shared_samples1 - shared_samples2
    shared_value = tf.abs(shared_value)
    # shared_value = tf.abs(shared_value)
    sameres = tf.reduce_sum(shared_value, 1, keepdims=True)

    return sameres


def lossg(y_true, y_pred):
    pixel_loss = pixel_mse(y_true, y_pred)
    lossg = 1 * pixel_loss
    return lossg


###################
def gyh_method(img1, img2, save_path, ind):
    ###########################################image process   lytro-09-A

    img_1 = image.img_to_array(img1)
    chang_1 = img_1.shape[0]
    kuan_1 = img_1.shape[1]
    print(img_1.shape, 'img_to_array')
    img_1 = np.expand_dims(img_1, axis=0)
    print(img_1.shape, 'expand_dims')
    img_1 = preprocess_images(img_1, tmin=0, tmax=1)
    print(img_1.shape, 'preprocess_images')
    img2 = image.img_to_array(img2)
    chang_2 = img2.shape[0]
    kuan_2 = img2.shape[1]
    print(img2.shape, 'img_to_array')
    img_2 = np.expand_dims(img2, axis=0)
    print(img_2.shape, 'expand_dims')
    img_2 = preprocess_images(img_2, tmin=0, tmax=1)
    print(img_2.shape, 'preprocess_images')


    ###########################################network structure:
    input0_1 = Input(shape=(chang_1, kuan_1, 1))
    input0_2 = Input(shape=(chang_1, kuan_1, 1))
    #input0_3 = Input(shape=(chang_1, kuan_1, IR))

    input_1 = keras.layers.Concatenate()([input0_1, input0_1, input0_1])
    input_2 = keras.layers.Concatenate()([input0_2, input0_2, input0_2])
    #input_3 = keras.layers.Concatenate()([input0_1, input0_2, input0_3])
    # input_1 = Input(shape=(chang_1, kuan_1,3))

    # input_2 = Input(shape=(chang_2, kuan_2,3))
    # input_1 = Input(shape=(496,632,3))

    # input_2 = Input(shape=(496,632,3))
    #############################share layer
    sharelayer1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')
    sharelayer2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')
    # sharelayer3 = MaxPooling2D((VI, VI), strides=(VI, VI), name='block1_pool')
    sharelayer4 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')
    sharelayer5 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last',
                         name='xs_de2_conv1')
    # sharelayer6 = UpSampling2D((VI, VI))
    sharelayer7 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last',
                         name='xs_de1_conv2')
    sharelayer8 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last',
                         name='xs_de1_conv1')
    outshare9 = Conv2D(3, (3, 3), padding='same', name='xs_out')
    # shareout=Conv2D(IR, (3, 3),activation='tanh',padding='same',name='share_out')
    ##
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='x1_block1_conv1')(input_1)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='x1_block1_conv2')(x1)
    # x1 = MaxPooling2D((VI, VI), strides=(VI, VI), name='x1_block1_pool')(x1)
    x1_e = Conv2D(128, (3, 3), activation='relu', padding='same', name='x1_block2_conv1')(x1)
    # x1=Conv2D(3, (3, 3),activation='tanh',padding='same',name='x1_out')(out_x1)
    ##

    x2_1 = sharelayer1(input_1)
    x2_2 = sharelayer1(input_2)

    x2_1 = sharelayer2(x2_1)
    x2_2 = sharelayer2(x2_2)

    x2_c = keras.layers.Concatenate(axis=3)([x2_1, x2_2])
    x2_e = sharelayer4(x2_c)





    ##
    x4 = Conv2D(64, (3, 3), activation='relu', padding='same', name='x4_block1_conv1')(input_2)
    x4 = Conv2D(64, (3, 3), activation='relu', padding='same', name='x4_block1_conv2')(x4)
    # x4 = MaxPooling2D((VI, VI), strides=(VI, VI), name='x4_block1_pool')(x4)
    x4_e = Conv2D(128, (3, 3), activation='relu', padding='same', name='x4_block2_conv1')(x4)
    # x4=Conv2D(3, (3, 3),activation='tanh',padding='same',name='x1_out')(out_x4)

    losszj1 = Lambda(lambda x: difference_loss(*x), name='losszj1')([x1_e, x2_e])  # this

    losszj2 = Lambda(lambda x: difference_loss(*x), name='losszj2')([x2_e, x4_e])

    # losszj3 = Lambda(lambda x:similarity_loss(*x),name = 'losszj3')([x2_e,x3_e])

    losszj = Lambda(lambda x: 0.6 * x[0] + 0.6 * x[1], name='losszj')([losszj1, losszj2])
    # output_hinden_pri=Lambda(test_xmy, output_shape=compare_output_shape)([x1_e,x4_e])
    # output_hinden_pri=keras.layers.Maximum()([x1_e,x4_e])                                            ########
    # output_hinden_pri=Lambda(layer_cal_xmy, output_shape=compare_output_shape)([x1_e,x4_e])


    output_hinden_pri = Lambda(test_gyh, output_shape=compare_output_shape)([x1_e, x4_e])  ######            pri


    # print(x1_e.shape,'x1_e.shape')
    # output_hinden_pri=Lambda(compare, output_shape=compare_output_shape)([x1_e,x4_e])
    print('output_hinden_pri_______:', output_hinden_pri.shape)
    # output_hinden_com1=keras.layers.Maximum()([x2_e,x3_e])
    # output_hinden_com2=keras.layers.Minimum()([x2_e,x3_e])
    # output_hinden_com=keras.layers.Average()([output_hinden_com1,output_hinden_com2])
    # output_hinden_com=Lambda(bianyuanqiangdu, output_shape=compare_output_shape)([x2_e,x3_e])
    # test_xmy  layer_cal_xmy  test_xmy
    # output_hinden_com=keras.layers.Maximum()([x2_e,x3_e])
    output_hinden_com = x2_e  ######com
    # output_hinden_com=Lambda(test_xmy, output_shape=compare_output_shape)([x2_e,x3_e])
    print(output_hinden_com, 'output_hinden_com')
    # output_hinden_com=Lambda(compare, output_shape=compare_output_shape)([output_hinden_com1,output_hinden_com2])
    # output_hinden_com=Lambda(compare, output_shape=compare_output_shape)([x2_e,x3_e])
    # output_hinden_priandcom = Lambda(test_gyh2,output_shape=compare_output_shape)([output_hinden_pri,output_hinden_com])          ##
    #output_hinden_priandcom = keras.layers.Add()([output_hinden_pri, output_hinden_com])
    # print('output_hinden_com_______:',output_hinden_com.shape)
    # out_hinden=keras.layers.add([output_hinden_1, output_hinden_2])

    xs_pri = sharelayer5(output_hinden_pri)
    xs_pri = sharelayer7(xs_pri)
    xs_pri = sharelayer8(xs_pri)

    xs_com = sharelayer5(output_hinden_com)
    xs_com = sharelayer7(xs_com)
    xs_com = sharelayer8(xs_com)

    out_xs_priandcom = keras.layers.add([xs_pri, xs_com])
    share_out =outshare9(out_xs_priandcom)

    '''
    xs_priandcom = sharelayer5(output_hinden_priandcom)
    # xs_pri=sharelayer6(xs_pri)
    xs_priandcom = sharelayer7(xs_priandcom)
    out_xs_priandcom = sharelayer8(xs_priandcom)
    # out_xs_pri=outshare9(xs_pri)
    share_out = outshare9(out_xs_priandcom)
    '''

    x1 = sharelayer5(x1_e)
    # x1 = sharelayer6(x1)
    x1 = sharelayer7(x1)
    out_x1 = sharelayer8(x1)
    # = outshare9(x1)
    # x1=shareout(out_x1)
    ##
    x2 = sharelayer5(x2_e)
    # x2 = sharelayer6(x2)
    x2 = sharelayer7(x2)
    out_x2 = sharelayer8(x2)
    # = outshare9(x2)
    # out12sum = keras.layers.merge([out_x1,out_x2],mode='sum')
    out12sum = keras.layers.Add()([out_x1, out_x2])  # !!!!
    # x12=outshare9(out_x1)
    x12 = outshare9(out12sum)  # !!!!!!
    ##
    ##
    x4 = sharelayer5(x4_e)
    # x4 = sharelayer6(x4)
    x4 = sharelayer7(x4)
    out_x4 = sharelayer8(x4)
    # out_x4 = outshare9(x4)
    # out34sum = keras.layers.merge([out_x3,out_x4],mode='sum')
    out34sum = keras.layers.Add()([out_x2, out_x4])  # !!!!!
    # x34=outshare9(out_x2)
    x34 = outshare9(out34sum)  # !!!!!!



    JAEmodel = Model(inputs=[input0_1, input0_2], outputs=[x12, x34, losszj])
    loss_layer = JAEmodel.get_layer('losszj').output
    JAEmodel.add_loss(loss_layer)
    predictmodel = Model(inputs=[input0_1, input0_2], outputs=share_out)
    predictmodel.summary()
    JAEmodel.load_weights('model.h5', by_name=True)



    print('start timing')
    time_start = time.time()
    predsimg_1 = predictmodel.predict([img_1, img_2])
    time_end1 = time.time()
    print('totally cost1', time_end1 - time_start)
    print(predsimg_1.shape, 'predsimg_1')
    predsimg_1 = postprocess_images(predsimg_1, omin=0, omax=1)
    preds_reimg_1 = np.reshape(predsimg_1, (predsimg_1.shape[1], predsimg_1.shape[2], predsimg_1.shape[3],))
    print(preds_reimg_1.shape, 'preds_reimg_1')
    #img = Image.fromarray(preds_reimg_1.astype('uint8')).convert('RGB')
    #img.save('result/' + filename + 'preds_reimg.bmp')
    img = Image.fromarray(preds_reimg_1.astype('uint8')).convert('L')
    img.save('result/' + filename +  '.png')
    #sio.savemat('result/' + filename + 'preds_reimg_1_weight0.mat', {'preds_reimg_1': preds_reimg_1})
    #imsave('result/' + filename + 'preds_reimg.bmp',preds_reimg_1)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = 'IR'
    log_device_placement = True
    allow_soft_placement = True
    tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.99
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(graph=tf.get_default_graph(), config=config))

    #file_path1 = '/root/raid1/gyh/TNO 520/IR/'
    #file_path2 = '/root/raid1/gyh/TNO 520/VI/'
    #file_path3 = '/root/raid1/gyh/TNO 520/3/'

    file_path1 = 'IR/'
    file_path2 = 'VI/'

    save_path = '/root/raid1/gyh/123/'
    for ind in range(1,49):
        filename = '{:0}'.format(ind)
        img_1 = os.path.join(file_path1 + '{0}.bmp'.format(filename))
        img_2 = os.path.join(file_path2 + '{0}.bmp'.format(filename))

        # img_3 = os.path.join('T3-{0}.bmp'.format(file_path3 +filename))
        img1 = image.load_img(img_1, grayscale=True)  # (240,320)  496,632  270,360,target_size=(320,240)
        img2 = image.load_img(img_2, grayscale=True)  # ,target_size=(320,240)

        print(img1, '############## dealing image1  #################')
        gyh_method(img1, img2, filename, ind)
