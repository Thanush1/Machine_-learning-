import tensorflow as tf 
import numpy as np 
import sys 

def fizz_buzz_encode(num):  
    if num % 3 == 0 and num %5 == 0:
        return np.array([1,0,0,0])
    if num % 3 == 0:
        return np.array([0,1,0,0])
    if num % 5 == 0:
        return np.array([0,0,1,0])
    return np.array([0,0,0,1])

def binary_encode(num,num_digits):
    activations = np.array([num>>d&1 for d in range(num_digits)])
    return activations 


test_input = open(str(sys.argv[1]),'r').read().strip()
string = test_input.split("\n")
values = [int(x) for x in string]

def logic_fizzbuzz(N):
    file = open('Software1.txt','w')
    for i in values :
        if i % 3 == 0 and i % 5 == 0:
            file.write("fizzbuzz" + '\n')
        elif i % 3 == 0:
            file.write("fizz" + '\n')
        elif i % 5 == 0:
            file.write("buzz" + '\n')
        else:
            file.write(str(i) + '\n')
            
    file.close()
    

# load json and create model
json_file = open('weight.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model1 = tf.keras.models.model_from_json(loaded_model_json,custom_objects = None )
# load weights into new model
model1.load_weights("model.h5")
logic_fizzbuzz(100)



#str(sys.argv[1]

test_x = np.array([binary_encode(x,10) for x in values])
test_y = np.array([fizz_buzz_encode(x) for x in values])

answer = model1.predict(test_x)

#validation = model1.evaluate(test_x,test_y)


def fizz_buzz_ML(N):
    file = open("Software2.txt",'w')
    for i in range(1,N):
        index = np.unravel_index(answer[i-1].argmax(),answer[i-1].shape)
        
        if index == (0,):
            file.write("fizzbuzz" + '\n')
        elif index == (1,):
            file.write("fizz" + '\n')
        elif index == (2,):
            file.write("buzz" + '\n')
        else:
            file.write(str(values[i-1]) + '\n')
    file.close()
            
fizz_buzz_ML(101)

import os
repo_path = os.path.dirname(os.path.abspath(__file__))
out_sw_1_fp = repo_path + "/Software1.txt"
out_sw_2_fp = repo_path + "/Software2.txt"
