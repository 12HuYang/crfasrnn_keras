'''
Transfer learning satterlite image using the crfasrnn
by Yang Hu
'''


import sys
sys.path.insert(1, './src')
from crfrnn_model import get_crfrnn_model_def
import util

def main():
    saved_model_path = 'crfrnn_keras_model.h5'
    model = get_crfrnn_model_def()
    model.load_weights(saved_model_path)   #model is made from Keras.Model function
    
