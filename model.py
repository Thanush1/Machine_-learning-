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


model2= tf.keras.models.Sequential([keras.layers.Dense(units =250 ,input_shape =([10]),activation  = 'relu'),
                                    keras.layers.Dense(units = 120,activation  = 'relu'),
                                    #keras.layers.Dense(units = 50 ,input_shape = ([10]),activation  = 'relu'),
                                    #keras.layers.Dense(units = 20 ,input_shape = ([10]),activation  = 'relu'),
                            keras.layers.Dense(units = 4,activation = 'softmax')
                            ])
model2.compile(loss = 'categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
model2.fit(training_X,training_Y,epochs = 100)

#model2.predict(binary_encode(5,10).reshape(1,10))


model_json = model2.to_json()
with open("weight.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model2.save_weights("model.h5")
