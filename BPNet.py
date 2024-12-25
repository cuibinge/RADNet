# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 21:02:57 2023

@author: Administrator
"""

import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import Model
from keras import layers
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Dense, Activation, SeparableConv2D,BatchNormalization,add
from keras.layers import GlobalAvgPool2D,Lambda, Reshape,subtract, multiply, UpSampling2D, Concatenate,Add,AveragePooling2D,GlobalAveragePooling2D
from keras.optimizers import SGD, rmsprop, Adam, Adamax
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow as tf
import keras.backend as K
from loss import my_binary_crossentropy_loss,iou_loss


#通道注意力CA
def CAB(low_layer,high_layer,channel_num,name1=None,name2=None,name3=None):
    # 输入通道数为64
    high_layer = Conv2D(channel_num, 1, activation='relu',padding='same', kernel_initializer='he_normal',name= name1)(high_layer)
    c1 = low_layer.shape[-1].value
    c2 = high_layer.shape[-1].value
    high_layer = UpSampling2D(size=(2, 2))(high_layer)  
    conv1 = concatenate([low_layer,high_layer])
    squeeze = GlobalAveragePooling2D()(conv1)
    squeeze = Reshape((1, 1, c1+c2))(squeeze)
    excitation = Conv2D(channel_num, 1, activation='relu',padding='same', kernel_initializer='he_normal')(squeeze)
    excitation = Conv2D(channel_num, 1)(excitation)
    excitation = Activation('sigmoid',name =name2)(excitation)
    mul_conv = multiply([low_layer, excitation],name =name3)
    out = add([mul_conv,high_layer])
    return out

#通道注意力FAM
def FAM(low_layer,high_layer,channel_num,name1=None,name2=None,name3=None):
    # 输入通道数为64
    s_sig = Conv2D(1, 1, activation='sigmoid',padding='same', kernel_initializer='he_normal')(high_layer)
    high_layer = multiply([high_layer,s_sig])  
    high_layer = Conv2D(channel_num, 1, activation='sigmoid',padding='same', kernel_initializer='he_normal')(high_layer)
    c1 = low_layer.shape[-1].value
    c2 = high_layer.shape[-1].value
    high_layer = UpSampling2D(size=(2, 2),name=name1)(high_layer)  
    conv1 = concatenate([low_layer,high_layer])
    squeeze = GlobalAveragePooling2D()(conv1)
    squeeze = Reshape((1, 1, c1+c2))(squeeze)
    excitation = Conv2D(channel_num, 1, activation='relu',padding='same', kernel_initializer='he_normal')(squeeze)
    excitation = Conv2D(channel_num, 1)(excitation)
    excitation = Activation('sigmoid',name=name2)(excitation)
    mul_conv = multiply([low_layer, excitation],name = name3)
    out = add([mul_conv,high_layer])
    return out
    
    
def GM(layer,name1):
    channel_num = layer.shape[-1].value
    a = Conv2D(2,1,padding = 'same', kernel_initializer = 'he_normal')(layer)
    so = Activation('softmax')(a)
    entropy =  Lambda(lambda x: K.sum((K.log(x)/K.log(2.0))*x,axis=-1)*(-1))(so)
    x = Lambda(lambda x: tf.expand_dims(x, axis=-1))(entropy)
    x = Activation("linear")(x)
    x = multiply([x,layer])
    x1 = Conv2D(channel_num,3, activation = 'relu',padding = 'same', kernel_initializer = 'he_normal')(x)
    x2 = Conv2D(1,1,padding = 'same', kernel_initializer = 'he_normal')(x1)  
    out = concatenate([layer,x1],axis = 3)
    out = Conv2D(channel_num,3, activation = 'relu',padding = 'same', kernel_initializer = 'he_normal')(out)
    return x2,out

def GM1(layer,name1):
    channel_num = layer.shape[-1].value
    a = Conv2D(2,1, activation = 'relu',padding = 'same', kernel_initializer = 'he_normal', name=name1)(layer)
    a = Conv2D(1,1,padding = 'same', kernel_initializer = 'he_normal')(a)
    # a = BatchNormalization()(a)  #必须要有，以避免sigmoid出现0和1,导致训练参数无法更新的问题 或者在公式中加入K.epsilon()微小的正值
    so = Activation('sigmoid')(a)
    tt1 = Lambda(lambda x: 1-x)(so)
    # a1 =  Lambda(lambda x: np.log2(x) -(K.log(x)/K.log(2.0))*x)(so)
    a1 =  Lambda(lambda x: -(K.log(x+K.epsilon())/K.log(2.0))*x)(so)
    a2 =  Lambda(lambda x: -(K.log(x+K.epsilon())/K.log(2.0))*(x))(tt1)
    entropy = add([a1,a2])
    x = Activation("linear")(entropy)
    x = multiply([x,layer])
    x1 = Conv2D(channel_num,1, activation = 'relu',padding = 'same', kernel_initializer = 'he_normal')(x)
    x2 = Conv2D(1,1,padding = 'same', kernel_initializer = 'he_normal')(x1)  
    out = concatenate([layer,x1])
    out = Conv2D(channel_num,3, activation = 'relu',padding = 'same', kernel_initializer = 'he_normal')(out)
    return x2,out

def GM2(layer,name1):
    channel_num = layer.shape[-1].value
    layer = Conv2D(channel_num,1,padding = 'same', kernel_initializer = 'he_normal',name = name1)(layer)
    layer = Activation('relu')(layer)
    a = Conv2D(1,1,padding = 'same', kernel_initializer = 'he_normal')(layer)
    a = BatchNormalization()(a)  #必须要有，以避免sigmoid出现0和1,导致训练参数无法更新的问题
    so = Activation('sigmoid')(a)
    tt1 = Lambda(lambda x: 1-x)(so)
    # a1 =  Lambda(lambda x: np.log2(x) -(K.log(x)/K.log(2.0))*x)(so)
    a1 =  Lambda(lambda x: -(K.log(x)/K.log(2.0))*x)(so)
    a2 =  Lambda(lambda x: -(K.log(x)/K.log(2.0))*(x))(tt1)
    entropy = add([a1,a2])
    x = multiply([entropy,layer])
    x1 = Conv2D(channel_num,1, padding = 'same', kernel_initializer = 'he_normal')(x)
    x1 = Activation('relu')(x1)
    # x2 = Conv2D(1,1,padding = 'same', kernel_initializer = 'he_normal')(x1)  
    out = concatenate([layer,x1])
    out = Conv2D(channel_num,3, padding = 'same', kernel_initializer = 'he_normal')(out)
    out = Activation('relu')(out)
    return layer,out

def res_block(inputlayer,kernelnum,name=None):
    conv = Conv2D(kernelnum, 1, padding='same', kernel_initializer='he_normal')(inputlayer)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv1 = Conv2D(kernelnum, 3,padding='same', kernel_initializer='he_normal')(conv)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv2D(kernelnum, 3, padding='same', kernel_initializer='he_normal')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)  
    if name:        
        out = add([conv, conv2],name=name)
    else:
        out = add([conv, conv2])
    return out

def backbone12(classes, input_size, epochs, batch_size, LR,
          Falg_summary=False, Falg_plot_model=False, pretrained_weights = None):
    # ======================================编码器=============================================
    input1 = Input(input_size)
    conv = Conv2D(64, 7, padding='same', kernel_initializer='he_normal')(input1)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    encode1 = res_block(conv, 64,name='res1') 
    pool1 = MaxPooling2D(pool_size=(2, 2))(encode1)
    encode2 = res_block(pool1, 128,name='res2')
    pool2 = MaxPooling2D(pool_size=(2, 2))(encode2)
    encode3 = res_block(pool2, 256,name='res3')
    pool3 = MaxPooling2D(pool_size=(2, 2))(encode3)
    encode4 = res_block(pool3, 512,name='res4')
    pool4 = MaxPooling2D(pool_size=(2, 2))(encode4)
    encode5 = res_block(pool4, 512,name='res5') 
    # ======================================解码器=============================================
    up4 =UpSampling2D(size=(2, 2),name = 'up4')(encode5)
    merge4 = concatenate([encode4,up4])
    conv =  Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='conv4')(merge4)
    
    up3 =UpSampling2D(size=(2, 2),name = 'up3')(conv)
    merge3 = concatenate([encode3,up3])
    conv =  Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='conv3')(merge3) 
   
    up2 =UpSampling2D(size=(2, 2),name = 'up2')(conv)
    merge2 = concatenate([encode2,up2])
    conv =  Conv2D(128, 3, padding='same', kernel_initializer='he_normal',name='conv2')(merge2)
    conv = Activation('relu')(conv)

    up1 =UpSampling2D(size=(2, 2),name = 'up1')(conv)
    merge1 = concatenate([encode1,up1])
    conv =  Conv2D(64, 3, padding='same', kernel_initializer='he_normal',name='conv1')(merge1) 
    conv = Activation('relu')(conv)
    
    # ======================================输出=============================================
    out = Conv2D(64, 3,  padding='same', kernel_initializer='he_normal',name = 'conv_out')(conv)
    conv = Activation('relu')(conv)
    #=========================================融合模块====================================================================
    out = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='out')(out)
    output = Conv2D(1, 1, activation='sigmoid', name='output')(out)  
    model = Model(input = input1, output = output)
    opt = Adam(lr = 1e-4)
    model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    if Falg_summary:
        model.summary()
    if Falg_plot_model:
        K.utils.plot_model(model, to_file='logs/Unet/model.png', show_shapes=True, show_layer_names=True)
    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def backbone_GM(classes, input_size, epochs, batch_size, LR,
          Falg_summary=False, Falg_plot_model=False, pretrained_weights = None):
    # ======================================编码器=============================================
    input1 = Input(input_size)
    conv = Conv2D(64, 7, padding='same', kernel_initializer='he_normal')(input1)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    encode1 = res_block(conv, 64,name='res1') 
    pool1 = MaxPooling2D(pool_size=(2, 2))(encode1)
    encode2 = res_block(pool1, 128,name='res2')
    pool2 = MaxPooling2D(pool_size=(2, 2))(encode2)
    encode3 = res_block(pool2, 256,name='res3')
    pool3 = MaxPooling2D(pool_size=(2, 2))(encode3)
    encode4 = res_block(pool3, 512,name='res4')
    pool4 = MaxPooling2D(pool_size=(2, 2))(encode4)
    encode5 = res_block(pool4, 512,name='res5') 
    # ======================================解码器=============================================
    up4 =UpSampling2D(size=(2, 2),name = 'up4')(encode5)
    merge4 = concatenate([encode4,up4])
    conv =  Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='conv4')(merge4)
    x,x1 = GM(conv)
    up4_1 = UpSampling2D(size=(8, 8))(x)
    loss1 = Activation('sigmoid',name='loss1')(up4_1)                    
    
    up3 =UpSampling2D(size=(2, 2),name = 'up3')(x1)
    merge3 = concatenate([encode3,up3])
    conv =  Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='conv3')(merge3) 
    x,x2 = GM(conv)
    up3_1 = UpSampling2D(size=(4, 4))(x)
    loss2 = Activation('sigmoid',name='loss2')(up3_1)  
    
    up2 =UpSampling2D(size=(2, 2),name = 'up2')(x2)
    merge2 = concatenate([encode2,up2])
    conv =  Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='conv2')(merge2)
    x,x3 = GM(conv)
    up2_1 = UpSampling2D(size=(2, 2))(x)
    loss3 = Activation('sigmoid',name='loss3')(up2_1) 

    up1 =UpSampling2D(size=(2, 2),name = 'up1')(x3)
    merge1 = concatenate([encode1,up1])
    conv =  Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='conv1')(merge1)  
    
    x,x4 = GM(conv)
    loss4 = Activation('sigmoid',name='loss4')(x) 
    # ======================================输出=============================================
    out = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',name = 'conv_out')(x4)
    #=========================================融合模块====================================================================

    out = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(out) 
    out = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='out')(out)
    output = Conv2D(1, 1, activation='sigmoid', name='output')(out)  
    model = Model(input = input1, output = output, name = 'output')
    opt = Adam(lr = 1e-4)
    model = Model(inputs=input1, outputs=[output,loss1,loss2,loss3,loss4])
    opt = Adam(lr = 1e-4)
    model.compile(optimizer=opt,
                  loss={'output': 'binary_crossentropy',
                        'loss1': 'binary_crossentropy',
                        'loss2': 'binary_crossentropy',
                        'loss3': 'binary_crossentropy',
                        'loss4': 'binary_crossentropy',
                       
                        },
                  loss_weights={
                        'output': 1,
                        'loss1': 0.1,
                        'loss2': 0.1,
                        'loss3': 0.1,
                        'loss4': 0.1,
                  },
                  metrics={
                         'output': ['accuracy'],
                        'loss1': ['accuracy'],
                        'loss2': ['accuracy'],
                        'loss3': ['accuracy'],
                        'loss4': ['accuracy'],
                      
                          })
    if Falg_summary:
        model.summary()
    if Falg_plot_model:
        K.utils.plot_model(model, to_file='logs/DANet_model.png', show_shapes=True, show_layer_names=True)
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model


def backbone_CA(classes, input_size, epochs, batch_size, LR,
          Falg_summary=False, Falg_plot_model=False, pretrained_weights = None):
    # ======================================编码器=============================================
    input1 = Input(input_size)
    conv = Conv2D(64, 7, padding='same', kernel_initializer='he_normal')(input1)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    encode1 = res_block(conv, 64,name='res1') 
    pool1 = MaxPooling2D(pool_size=(2, 2))(encode1)
    encode2 = res_block(pool1, 128,name='res2')
    pool2 = MaxPooling2D(pool_size=(2, 2))(encode2)
    encode3 = res_block(pool2, 256,name='res3')
    pool3 = MaxPooling2D(pool_size=(2, 2))(encode3)
    encode4 = res_block(pool3, 512,name='res4')
    pool4 = MaxPooling2D(pool_size=(2, 2))(encode4)
    encode5 = res_block(pool4, 512,name='res5') 
    # ======================================解码器=============================================
    conv = CAB(encode4,encode5,512)
    conv1 =  Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='conv4')(conv)                 
    
    conv = CAB(encode3,conv1,256)
    conv2 =  Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='conv3')(conv) 
   
    
    conv = CAB(encode2,conv2,128)
    conv3 =  Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='conv2')(conv)
    
    conv = CAB(encode1,conv3,64)
    conv4 =  Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='conv1')(conv)  
    
   
    # ======================================输出=============================================
    out = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',name = 'conv_out')(conv4)
    #=========================================融合模块====================================================================

    out = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(out) 
    out = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='out')(out)
    output = Conv2D(1, 1, activation='sigmoid', name='output')(out)  
    model = Model(input = input1, output = output)
    opt = Adam(lr = 1e-4)
    model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    if Falg_summary:
        model.summary()
    if Falg_plot_model:
        K.utils.plot_model(model, to_file='logs/Unet/model.png', show_shapes=True, show_layer_names=True)
    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def backbone_FAM(classes, input_size, epochs, batch_size, LR,
          Falg_summary=False, Falg_plot_model=False, pretrained_weights = None):
    # ======================================编码器=============================================
    input1 = Input(input_size)
    conv = Conv2D(64, 7, padding='same', kernel_initializer='he_normal')(input1)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    encode1 = res_block(conv, 64,name='res1') 
    pool1 = MaxPooling2D(pool_size=(2, 2))(encode1)
    encode2 = res_block(pool1, 128,name='res2')
    pool2 = MaxPooling2D(pool_size=(2, 2))(encode2)
    encode3 = res_block(pool2, 256,name='res3')
    pool3 = MaxPooling2D(pool_size=(2, 2))(encode3)
    encode4 = res_block(pool3, 512,name='res4')
    pool4 = MaxPooling2D(pool_size=(2, 2))(encode4)
    encode5 = res_block(pool4, 512,name='res5') 
    # ======================================解码器=============================================
    conv = FAM(encode4,encode5,512)
    conv1 =  Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='conv4')(conv)                 
    
    conv = FAM(encode3,conv1,256)
    conv2 =  Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='conv3')(conv) 
   
    
    conv = FAM(encode2,conv2,128)
    conv3 =  Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='conv2')(conv)
    
    conv = FAM(encode1,conv3,64)
    conv4 =  Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='conv1')(conv)  
    
   
    # ======================================输出=============================================
    out = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',name = 'conv_out')(conv4)
    #=========================================融合模块====================================================================

    out = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(out) 
    out = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='out')(out)
    output = Conv2D(1, 1, activation='sigmoid', name='output')(out)  
    model = Model(input = input1, output = output)
    opt = Adam(lr = 1e-4)
    model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    if Falg_summary:
        model.summary()
    if Falg_plot_model:
        K.utils.plot_model(model, to_file='logs/Unet/model.png', show_shapes=True, show_layer_names=True)
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model

def backbone_GM_CA(classes, input_size, epochs, batch_size, LR,
          Falg_summary=False, Falg_plot_model=False, pretrained_weights = None):
    # ======================================编码器=============================================
    input1 = Input(input_size)
    conv = Conv2D(64, 7, padding='same', kernel_initializer='he_normal')(input1)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    encode1 = res_block(conv, 64,name='res1') 
    pool1 = MaxPooling2D(pool_size=(2, 2))(encode1)
    encode2 = res_block(pool1, 128,name='res2')
    pool2 = MaxPooling2D(pool_size=(2, 2))(encode2)
    encode3 = res_block(pool2, 256,name='res3')
    pool3 = MaxPooling2D(pool_size=(2, 2))(encode3)
    encode4 = res_block(pool3, 512,name='res4')
    pool4 = MaxPooling2D(pool_size=(2, 2))(encode4)
    encode5 = res_block(pool4, 512,name='res5') 
    # ======================================解码器=============================================
    conv = CAB(encode4,encode5,512)
    conv =  Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='conv4')(conv)
    x,x1 = GM(conv,name1='11')
    up4_1 = UpSampling2D(size=(8, 8))(x)
    loss1 = Activation('sigmoid',name='loss1')(up4_1)                    
    
    conv = CAB(encode3,x1,256)
    conv =  Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='conv3')(conv) 
    x,x2 =  GM(conv,name1='22')
    up3_1 = UpSampling2D(size=(4, 4))(x)
    loss2 = Activation('sigmoid',name='loss2')(up3_1)  
    
    conv = CAB(encode2,x2,128)
    conv =  Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='conv2')(conv)
    x,x3 =  GM(conv,name1='33')
    up2_1 = UpSampling2D(size=(2, 2))(x)
    loss3 = Activation('sigmoid',name='loss3')(up2_1) 

    conv = CAB(encode1,x3,64)
    conv =  Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='conv1')(conv)  
    
    x,x4 =  GM(conv,name1='44')
    loss4 = Activation('sigmoid',name='loss4')(x) 
    # ======================================输出=============================================
    out = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',name = 'conv_out')(x4)
    #=========================================融合模块====================================================================

    out = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(out) 
    out = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='out')(out)
    output = Conv2D(1, 1, activation='sigmoid', name='output')(out)  
    model = Model(input = input1, output = output, name = 'output')
    opt = Adam(lr = 1e-4)
    model = Model(inputs=input1, outputs=[output,loss1,loss2,loss3,loss4])
    opt = Adam(lr = 1e-4)
    model.compile(optimizer=opt,
                  loss={'output': 'binary_crossentropy',
                        'loss1': 'binary_crossentropy',
                        'loss2': 'binary_crossentropy',
                        'loss3': 'binary_crossentropy',
                        'loss4': 'binary_crossentropy',
                       
                        },
                  loss_weights={
                        'output': 1,
                        'loss1': 0.1,
                        'loss2': 0.1,
                        'loss3': 0.1,
                        'loss4': 0.1,
                  },
                  metrics={
                         'output': ['accuracy'],
                        'loss1': ['accuracy'],
                        'loss2': ['accuracy'],
                        'loss3': ['accuracy'],
                        'loss4': ['accuracy'],
                      
                          })
    if Falg_summary:
        model.summary()
    if Falg_plot_model:
        K.utils.plot_model(model, to_file='logs/DANet_model.png', show_shapes=True, show_layer_names=True)
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model


def backbone_GM_CA1(classes, input_size, epochs, batch_size, LR,
          Falg_summary=False, Falg_plot_model=False, pretrained_weights = None):
    # ======================================编码器=============================================
    input1 = Input(input_size)
    conv = Conv2D(64, 7, padding='same', kernel_initializer='he_normal')(input1)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    encode1 = res_block(conv, 64,name='res1') 
    pool1 = MaxPooling2D(pool_size=(2, 2))(encode1)
    encode2 = res_block(pool1, 128,name='res2')
    pool2 = MaxPooling2D(pool_size=(2, 2))(encode2)
    encode3 = res_block(pool2, 256,name='res3')
    pool3 = MaxPooling2D(pool_size=(2, 2))(encode3)
    encode4 = res_block(pool3, 512,name='res4')
    pool4 = MaxPooling2D(pool_size=(2, 2))(encode4)
    encode5 = res_block(pool4, 512,name='res5') 
    # ======================================解码器=============================================
    conv = CAB(encode4,encode5,512)
    conv =  Conv2D(512, 3, padding='same', kernel_initializer='he_normal',name='conv4')(conv)
    conv = Activation('relu')(conv)
    x,x1 = GM1(conv,name1='11')
    up4_1 = UpSampling2D(size=(8, 8))(x)
    loss1 = Activation('sigmoid',name='loss1')(up4_1)                    
    
    conv = CAB(encode3,x1,256)
    conv =  Conv2D(256, 3, padding='same', kernel_initializer='he_normal',name='conv3')(conv) 
    conv = Activation('relu')(conv)
    x,x2 =  GM1(conv,name1='22')
    up3_1 = UpSampling2D(size=(4, 4))(x)
    loss2 = Activation('sigmoid',name='loss2')(up3_1)  
    
    conv = CAB(encode2,x2,128)
    conv =  Conv2D(128, 3, padding='same', kernel_initializer='he_normal',name='conv2')(conv)
    conv = Activation('relu')(conv)
    x,x3 =  GM1(conv,name1='33')
    up2_1 = UpSampling2D(size=(2, 2))(x)
    loss3 = Activation('sigmoid',name='loss3')(up2_1) 

    conv = CAB(encode1,x3,64)
    conv =  Conv2D(64, 3, padding='same', kernel_initializer='he_normal',name='conv1')(conv)  
    conv = Activation('relu')(conv)
    
    x,x4 =  GM1(conv,name1='44')
    loss4 = Activation('sigmoid',name='loss4')(x) 
    # ======================================输出=============================================
    out = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x4)
    #=========================================融合模块====================================================================

    out = Conv2D(64, 3, padding='same', kernel_initializer='he_normal',name='tt')(out) 
    out = Activation('relu')(out)
    out = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='out')(out)
    out = Activation('relu',name = 'conv_out')(out)
    output = Conv2D(1, 1, activation='sigmoid', name='output')(out)  
    model = Model(input = input1, output = output, name = 'output')
    opt = Adam(lr = 1e-4)
    model = Model(inputs=input1, outputs=[output,loss1,loss2,loss3,loss4])
    opt = Adam(lr = 1e-4)
    model.compile(optimizer=opt,
                  loss={'output': 'binary_crossentropy',
                        'loss1': my_binary_crossentropy_loss,
                        'loss2': my_binary_crossentropy_loss,
                        'loss3': my_binary_crossentropy_loss,
                        'loss4': my_binary_crossentropy_loss,
                       
                        },
                  loss_weights={
                        'output': 1,
                        'loss1': 0.1,
                        'loss2': 0.1,
                        'loss3': 0.1,
                        'loss4': 0.1,
                  },
                  metrics={
                          'output': ['accuracy'],
                        'loss1': ['accuracy'],
                        'loss2': ['accuracy'],
                        'loss3': ['accuracy'],
                        'loss4': ['accuracy'],
                      
                          })
    if Falg_summary:
        model.summary()
    if Falg_plot_model:
        K.utils.plot_model(model, to_file='logs/DANet_model.png', show_shapes=True, show_layer_names=True)
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model

def BPNet(classes, input_size, epochs, batch_size, LR,
          Falg_summary=False, Falg_plot_model=False, pretrained_weights = None):
    # ======================================编码器=============================================
    input1 = Input(input_size)
    conv = Conv2D(64, 7, padding='same', kernel_initializer='he_normal')(input1)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    encode1 = res_block(conv, 64,name='res1') 
    pool1 = MaxPooling2D(pool_size=(2, 2))(encode1)
    encode2 = res_block(pool1, 128,name='res2')
    pool2 = MaxPooling2D(pool_size=(2, 2))(encode2)
    encode3 = res_block(pool2, 256,name='res3')
    pool3 = MaxPooling2D(pool_size=(2, 2))(encode3)
    encode4 = res_block(pool3, 512,name='res4')
    pool4 = MaxPooling2D(pool_size=(2, 2))(encode4)
    encode5 = res_block(pool4, 512,name='res5') 
    # ======================================解码器=============================================
    x = Conv2D(1, 3, padding='same', kernel_initializer='he_normal',name='conv5')(encode5)
    up5_1 = UpSampling2D(size=(16, 16))(x)
    loss0 = Activation('sigmoid',name='loss0')(up5_1)  
    
    
    conv = CAB(encode4,encode5,512,name1 = 'c11',name2 = 'c12',name3 = 'c13')
    conv =  Conv2D(512, 3, padding='same', kernel_initializer='he_normal',name='conv4')(conv)
    conv = Activation('relu')(conv)
    x,x1 = GM1(conv,name1='11')
    up4_1 = UpSampling2D(size=(8, 8))(x)
    loss1 = Activation('sigmoid',name='loss1')(up4_1)                    
    
    conv = CAB(encode3,x1,256,name1 = 'c21',name2 = 'c22',name3 = 'c23')
    conv =  Conv2D(256, 3, padding='same', kernel_initializer='he_normal',name='conv3')(conv) 
    conv = Activation('relu')(conv)
    x,x2 =  GM1(conv,name1='22')
    up3_1 = UpSampling2D(size=(4, 4))(x)
    loss2 = Activation('sigmoid',name='loss2')(up3_1)  
    
    conv = CAB(encode2,x2,128,name1 = 'c31',name2 = 'c32',name3 = 'c33')
    conv =  Conv2D(128, 3, padding='same', kernel_initializer='he_normal',name='conv2')(conv)
    conv = Activation('relu')(conv)
    x,x3 =  GM1(conv,name1='33')
    up2_1 = UpSampling2D(size=(2, 2))(x)
    loss3 = Activation('sigmoid',name='loss3')(up2_1) 

    conv = CAB(encode1,x3,64,name1 = 'c41',name2 = 'c42',name3 = 'c43')
    conv =  Conv2D(64, 3, padding='same', kernel_initializer='he_normal',name='conv1')(conv)  
    conv = Activation('relu')(conv)
    
    x,x4 =  GM1(conv,name1='44')
    loss4 = Activation('sigmoid',name='loss4')(x) 
    # ======================================输出=============================================
    out = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x4)
    #=========================================融合模块====================================================================

    out = Conv2D(64, 3, padding='same', kernel_initializer='he_normal',name='tt')(out) 
    out = Activation('relu')(out)
    out = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='out')(out)
    out = Activation('relu',name = 'conv_out')(out)
    output = Conv2D(1, 1, activation='sigmoid', name='output')(out)  
    model = Model(input = input1, output = output, name = 'output')
    opt = Adam(lr = 1e-4)
    model = Model(inputs=input1, outputs=[output,loss0,loss1,loss2,loss3,loss4])
    opt = Adam(lr = 1e-4)
    model.compile(optimizer=opt,
                  loss={'output': 'binary_crossentropy',
                        'loss0':iou_loss,
                        'loss1': my_binary_crossentropy_loss,
                        'loss2': my_binary_crossentropy_loss,
                        'loss3': my_binary_crossentropy_loss,
                        'loss4': my_binary_crossentropy_loss,
                       
                        },
                  loss_weights={
                        'output': 1,
                        'loss0':0.1,
                        'loss1': 0.1,
                        'loss2': 0.1,
                        'loss3': 0.1,
                        'loss4': 0.1,
                  },
                  metrics={
                          'output': ['accuracy'],
                        'loss0': ['accuracy'],
                        'loss1': ['accuracy'],
                        'loss2': ['accuracy'],
                        'loss3': ['accuracy'],
                        'loss4': ['accuracy'],
                      
                          })
    if Falg_summary:
        model.summary()
    if Falg_plot_model:
        K.utils.plot_model(model, to_file='logs/DANet_model.png', show_shapes=True, show_layer_names=True)
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model


def backbone_GM_CA2(classes, input_size, epochs, batch_size, LR,
          Falg_summary=False, Falg_plot_model=False, pretrained_weights = None):
    # ======================================编码器=============================================
    input1 = Input(input_size)
    conv = Conv2D(64, 7, padding='same', kernel_initializer='he_normal')(input1)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    encode1 = res_block(conv, 64,name='res1') 
    pool1 = MaxPooling2D(pool_size=(2, 2))(encode1)
    encode2 = res_block(pool1, 128,name='res2')
    pool2 = MaxPooling2D(pool_size=(2, 2))(encode2)
    encode3 = res_block(pool2, 256,name='res3')
    pool3 = MaxPooling2D(pool_size=(2, 2))(encode3)
    encode4 = res_block(pool3, 512,name='res4')
    pool4 = MaxPooling2D(pool_size=(2, 2))(encode4)
    encode5 = res_block(pool4, 512,name='res5') 
    # ======================================解码器=============================================
    conv = CAB(encode4,encode5,512)
    conv =  Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='conv4')(conv)
    x,x1 = GM2(conv,name1='11')
    up4_1 = UpSampling2D(size=(8, 8))(x)
    loss1 = Conv2D(1, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal',name='loss1')(up4_1)
    # Activation('sigmoid',name='loss1')(up4_1)                    
    
    conv = CAB(encode3,x1,256)
    conv =  Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='conv3')(conv) 
    x,x2 =  GM2(conv,name1='22')
    # x =  Conv2D(1, 3,padding='same', kernel_initializer='he_normal')(conv)
    up3_1 = UpSampling2D(size=(4, 4))(x)
    loss2 = Conv2D(1, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal',name='loss2')(up3_1)
    
    conv = CAB(encode2,x2,128)
    conv =  Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='conv2')(conv)
    x,x3 =  GM2(conv,name1='33')
    # x =  Conv2D(1, 3,padding='same', kernel_initializer='he_normal')(conv)
    up2_1 = UpSampling2D(size=(2, 2))(x)
    loss3 = Conv2D(1, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal',name='loss3')(up2_1) 

    conv = CAB(encode1,x3,64)
    conv =  Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='conv1')(conv)  
    
    x,x4 =  GM2(conv,name1='44')
    loss4 = Conv2D(1, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal',name='loss4')(x) 
    # x =  Conv2D(1, 3,padding='same', kernel_initializer='he_normal')(conv)
    # ======================================输出=============================================
    out = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',name = 'conv_out')(x4)
    #=========================================融合模块====================================================================

    out = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(out) 
    out = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='out')(out)
    output = Conv2D(1, 1, activation='sigmoid', name='output')(out)  
    model = Model(input = input1, output = output, name = 'output')
    opt = Adam(lr = 1e-4)
    model = Model(inputs=input1, outputs=[output,loss1,loss2,loss3,loss4])
    opt = Adam(lr = 1e-4)
    model.compile(optimizer=opt,
                  loss={'output': 'binary_crossentropy',
                        'loss1': 'binary_crossentropy',
                        'loss2': 'binary_crossentropy',
                        'loss3': 'binary_crossentropy',
                        'loss4': 'binary_crossentropy',
                       
                        },
                  loss_weights={
                        'output': 1,
                        'loss1': 0.1,
                        'loss2': 0.1,
                        'loss3': 0.1,
                        'loss4': 0.1,
                  },
                  metrics={
                         'output': ['accuracy'],
                        'loss1': ['accuracy'],
                        'loss2': ['accuracy'],
                        'loss3': ['accuracy'],
                        'loss4': ['accuracy'],
                      
                          })
    if Falg_summary:
        model.summary()
    if Falg_plot_model:
        K.utils.plot_model(model, to_file='logs/DANet_model.png', show_shapes=True, show_layer_names=True)
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model

BPNet(2, (128,128,3), epochs=1, batch_size=1, LR=0.01, Falg_summary=True, Falg_plot_model=False)
