'''
Transfer learning satterlite image using the crfasrnn
by Yang Hu
'''


import sys
import os
sys.path.insert(1, './src')
from keras.models import Model
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, \
    Dropout, Conv2DTranspose, Cropping2D, Add
from crfrnn_layer import CrfRnnLayer
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import h5py

IMAGE_PATH = './images/'
MASKS_PATH = '../masks/'

image_ids=os.walk(IMAGE_PATH)
mask_ids=os.walk(MASKS_PATH)
print(image_ids)
print(mask_ids)

def main():
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

    channels, height, width = 3, 100, 100
    input_shape = (height, width, 3)
    img_input = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(100, 100))(img_input)
    # VGG-16 convolution block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='valid', name='conv1_1')(x)
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
    score4 = Conv2DTranspose(5, (4, 4), strides=2, name='score4', use_bias=False)(score_fused)

    # Skip connections from pool3
    score_pool3 = Conv2D(5, (1, 1), name='score-pool3')(pool3)
    score_pool3c = Cropping2D((9, 9))(score_pool3)

    # Fuse things together
    score_final = Add()([score4, score_pool3c])

    # Final up-sampling and cropping
    upsample = Conv2DTranspose(5, (16, 16), strides=8, name='upsample', use_bias=False)(score_final)
    upscore = Cropping2D(((31, 37), (31, 37)))(upsample)

    output = CrfRnnLayer(image_dims=(height, width),
                         num_classes=5,
                         theta_alpha=160.,
                         theta_beta=3.,
                         theta_gamma=3.,
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
    for i in range(5,last):
        name=layer_names[i]
        c=list(f[name])
        model.layers[i].trainable=False
        if len(c)>0:
            print(c)
            d=list(f[name][c[0]])
            print(d)
            weight=[f[name][c[0]][d[1]],f[name][c[0]][d[0]]]
            weight=np.asarray(weight)
            model.layers[i].set_weights(weight)
            #print(weight[0].shape)
            #print(weight[1].shape)
            #test=model.layers[i].get_weights()
            #test=np.array(test)
            #print(test.shape)
    model.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])




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




