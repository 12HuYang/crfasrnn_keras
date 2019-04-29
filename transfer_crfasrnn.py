'''
Transfer learning satterlite image using the crfasrnn
by Yang Hu
'''
import tensorflow as tf
import os
import keras.backend as K
def assignGPU(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"]="%s" % (gpu)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))

assignGPU(2)

import sys
sys.path.insert(1, './src')
from keras.models import Model
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, \
    Dropout, Conv2DTranspose, Cropping2D, Add
from crfrnn_layer import CrfRnnLayer
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import h5py
import util
from PIL import Image

IMAGE_PATH = './images02/'
MASKS_PATH = './masks02/'

image_ids=[]
mask_ids=[]

channels, height, width = 3, 250, 250
for root,dirs,image_ids in os.walk(IMAGE_PATH):
    print(image_ids)
for rroot,rdirs,mask_ids in os.walk(MASKS_PATH):
    print(mask_ids)
image_ids.sort()
mask_ids.sort()
X = np.zeros((len(mask_ids), height, width,3), dtype=np.float32)
for i in range(len(image_ids)):
    X[i],imgh,imgw=util.get_preprocessed_image(IMAGE_PATH+image_ids[i])

Y = np.zeros((len(mask_ids), height, width,5), dtype=np.int32)
labelpattern=[[255,255,255,255],    #white, background
              [0,128,0,255],        #irrigate 01
              [0,0,255,255],        #non-irrigate
              [0,0,0,255],          #irrigate 02
              [255,192,203,255]]    #irrigate 03
for i in range(len(mask_ids)):
    im = np.array(Image.open(MASKS_PATH+mask_ids[i])).astype(np.int32)
    for j in range(len(labelpattern)):
        temp=im-labelpattern[j]
        temp=np.sum(temp,axis=2)
        b=np.where(temp==0)
        temparray=np.zeros(5)
        temparray[j]=1
        Y[i][b]=temparray
    print(im)
    #assert im.ndim == 3, 'Only RGB images are supported.'
    #im = im - _IMAGENET_MEANS
    #im = im[:, :, ::-1]  # Convert to BGR
    #img_h, img_w, img_c = im.shape
    #print(img_h,img_w,img_c)
    #assert img_c == 3, 'Only RGB images are supported.'
    #if img_h > 500 or img_w > 500:
    #    raise ValueError('Please resize your images to be not bigger than 500 x 500.')


    #pad_h = 500 - img_h
    #pad_w = 500 - img_w
    #im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
    #print(im)

def iou_loss_core(true,pred):  #this can be used as a loss if you make it negative
    intersection = true * pred
    notTrue = 1 - true
    union = true + (notTrue * pred)
    return (K.sum(intersection, axis=-1) + K.epsilon()) / (K.sum(union, axis=-1) + K.epsilon())

def getmodel():
    saved_model_path = 'crfrnn_keras_model.h5'
    f=h5py.File(saved_model_path)
    #a=list(f.keys())
    #print(a)
    #for i in range(len(a)):
    #    b=list(f[a[i]])
    #    print(b)
    #    if len(b)>0:
    #        c=list(f[a[i]][b[0]])
    #        print(c)
    #        if len(c)>0:
    #            for j in range(len(c)):
    #                d=list(f[a[i]][b[0]][c[j]])
    #                print('size of d')
    #                print(len(d))
    #                #print(d)



    input_shape = (height, width, 3)
    img_input = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(250, 250))(img_input)
    # VGG-16 convolution block 1
    x = Conv2D(3, (3, 3), activation='relu', padding='valid', name='convpre1_1')(x)
    x = Conv2D(3, (3, 3), activation='relu', padding='same', name='convpre1_2')(x)
    x = MaxPooling2D((2, 2), strides=(1, 1), name='poolpre1')(x)

    # VGG-16 convolution block 2
    x = Conv2D(3, (3, 3), activation='relu', padding='same', name='convpre2_1')(x)
    x = Conv2D(3, (3, 3), activation='relu', padding='same', name='convpre2_2')(x)
    x = MaxPooling2D((2, 2), strides=(1, 1), name='poolpre2', padding='same')(x)

    # VGG-16 convolution block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # VGG-16 convolution block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2', padding='same')(x)

    # VGG-16 convolution block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3', padding='same')(x)
    pool3 = x

    # VGG-16 convolution block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4', padding='same')(x)
    pool4 = x

    # VGG-16 convolution block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5', padding='same')(x)

    # Fully-connected layers converted to convolution layers
    x = Conv2D(4096, (7, 7), activation='relu', padding='valid', name='fc6')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='valid', name='fc7')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(5, (1, 1), padding='valid', name='score-fr')(x)

    score2 = Conv2DTranspose(5, (3, 3), strides=2, name='score2')(x)

    # Skip connections from pool4
    score_pool4 = Conv2D(5, (1, 1), name='score-pool4')(pool4)
    score_pool4c = Cropping2D((5, 5))(score_pool4)
    score_fused = Add()([score2, score_pool4c])
    score4 = Conv2DTranspose(5, (5, 5), strides=2, name='score4', use_bias=False)(score_fused)

    # Skip connections from pool3
    score_pool3 = Conv2D(5, (2, 2), name='score-pool3')(pool3)
    score_pool3c = Cropping2D((8, 8))(score_pool3)

    # Fuse things together
    score_final = Add()([score4, score_pool3c])

    # Final up-sampling and cropping
    upsample = Conv2DTranspose(5, (16, 16), strides=8, name='upsample', use_bias=False)(score_final)
    upscore = Cropping2D(((39, 37), (39, 37)))(upsample)

    output = CrfRnnLayer(image_dims=(height, width),
                         num_classes=5,
                         theta_alpha=160.,
                         #theta_beta=3.,
                         #theta_gamma=3.,
                         theta_beta=160.,
                         theta_gamma=160.,
                         num_iterations=10,
                         name='crfrnn')([upscore, img_input])

    # Build the model
    model = Model(img_input, output, name='crfrnn_net')


    #model = Model(input = img_input, output = x)
    print(model.summary())

    #transfer learning from https://medium.com/@14prakash/transfer-learning-using-keras-d804b2e04ef8
    layer_names = [layer.name for layer in model.layers]
    print(layer_names)
    last=layer_names.index('score-fr')
    for i in range(8+3,last-2):
    #for i in range(8,last):
        name=layer_names[i]
        dropname=name.find('dropout')
        if dropname!=-1:
            print(name)
            nameindex=int(name[8])
            print(nameindex)
            if nameindex>2 and nameindex%2!=0:
                nameindex=1
            if nameindex>2 and nameindex%2==0:
                nameindex=2
            print('change dropout layer names')
            name=name[:8]+str(nameindex)

        c=list(f[name])
        model.layers[i].trainable=False
        if len(c)>0:
            print(c)
            d=list(f[name][c[0]])
            print(d)
            weight=[f[name][c[0]][d[1]],f[name][c[0]][d[0]]]
            weight=np.asarray(weight)
            print(i)
            model.layers[i].set_weights(weight)
            #print(weight[0].shape)
            #print(weight[1].shape)
            #test=model.layers[i].get_weights()
            #test=np.array(test)
            #print(test.shape)
    #for i in range(5,last):
    #    model.layers[i].trainable=False
    model.compile(loss = "categorical_crossentropy", optimizer = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),metrics=["accuracy",iou_loss_core])#optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

    return model
def main():
    for i in range(len(mask_ids)):
        X_test=np.asarray([X[i]])
        Y_test=np.asarray([Y[i]])
        print (X_test.shape)
        print (Y_test.shape)
        tempX=X.tolist()
        tempY=Y.tolist()
        X_train=tempX[0:i]+tempX[i+1:]
        Y_train=tempY[0:i]+tempY[i+1:]
        X_train=np.asarray(X_train)
        Y_train=np.asarray(Y_train)
        print(X_train.shape)
        print(Y_train.shape)
        model=getmodel()
        model.fit(X_train,Y_train,epochs=4000,batch_size=16)
        #preds=model.predict(X_test,Y_test)
        #preds=model.evaluate(X_test,Y_test)
        #print ("Loss = " + str(preds[0]))
        #print ("Test Accuracy = " + str(preds[1]))
        probs = model.predict(X_test, verbose=False)[0, :, :, :]
        print("Test IU score:"+str(iou_loss_core(Y_test,probs)))
        output_file = 'labels'+str(i)+'.png'
        segmentation = util.get_label_image(probs, 100, 100)
        segmentation.save(output_file)



    '''
    base_model = get_crfrnn_model_def()
    base_model.load_weights(saved_model_path)   #model is made from Keras.Model function
    print(base_model.summary())
    print(len(base_model.layers))
    print(base_model.layers[24].get_config())  #beginning of RNN, beginning of chunk of output layers
    base_model.layers[0].trainable=False    #input
    base_model.layers[1].trainable=False    #zero_padding
    for layer in base_model.layers[7:24]:    #from https://medium.com/@14prakash/transfer-learning-using-keras-d804b2e04ef8
        layer.trainable = False
    '''






if __name__ == '__main__':
    main()




