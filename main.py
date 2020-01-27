#!/usr/bin/env python
# coding: utf-8

# In[1]:


import model 

def logic_fizzbizz(N):
    file = open('Software1.0.txt','a')
    for i in range(1,N+1):
        if i % 3 == 0 and i % 5 == 0:
            file.write("fizzbuzz" + '\n')
        elif i % 3 == 0:
            file.write("fizz" + '\n')
        elif i % 5 == 0:
            file.write("buzz" + '\n')
        else:
            file.write(str(i) + '\n')
            
    f.close()
    
logic_fizz_buzz(100)

test_input = open('test_input.txt','r').read().strip()
string = test_input.split("\n")
values = [int(x) for x in string]

test_x = np.array([binary_encode(x) for x in values])
test_y = np.array([fizz_buz_enocde(x) for x in values])

answer = model.predict(test_x)
validation = model.evaluate(test_x,test_y)

def fizz_buzz_ML(N):
    file = open("software2.0.txt",'a')
    for i in range(1,N+1):
        index = 
        if index == (0,):
            file.write("fizz_buzz" + '\n')
        elif index == (1,):
            file.write("fizz" + '\n')
        elif index == (2,):
            file.write("buzz" + '\n')
        else:
            file.write(str(i) + '\n')
            
fizz_buzz_ML(100)

            
        


# In[ ]:




