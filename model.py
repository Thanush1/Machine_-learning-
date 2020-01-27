import tensorflow as tf 
import numpy as np 
from tensorflow import keras 
num_digits = 10
def binary_encode(num,num_digits):
    activations = np.array([num>>d&1 for d in range(num_digits)])
    return activations 
    

def fizz_buzz_encode(num):  
    if num % 3 == 0 and num %5 == 0:
        return np.array([1,0,0,0])
    if num % 3 == 0:
        return np.array([0,1,0,0])
    if num % 5 == 0:
        return np.array([0,0,1,0])
    return np.array([0,0,0,1])

training_X = np.array([binary_encode(x,num_digits) for x in range(101,1001)])
training_Y = np.array([fizz_buzz_encode(i) for i in range(101,1001)])

def init_weights(shape):
    return tf.Variable(tf.random.normal(shape,stddev = 0.01 ))


model1 = tf.keras.models.Sequential([keras.layers.Dense(units =250 ,input_shape =([10]),activation  = 'relu'),
                                    keras.layers.Dense(units = 120,activation  = 'relu'),
                                    #keras.layers.Dense(units = 50 ,input_shape = ([10]),activation  = 'relu'),
                                    #keras.layers.Dense(units = 20 ,input_shape = ([10]),activation  = 'relu'),
                            keras.layers.Dense(units = 4,activation = 'softmax')
                            ])
model1.compile(loss = 'categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
model1.fit(training_X,training_Y,epochs = 100)
