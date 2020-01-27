from model import *
def logic_fizzbuzz(N):
    file = open('Software1.0.txt','w')
    for i in range(1,N+1):
        if i % 3 == 0 and i % 5 == 0:
            file.write("fizzbuzz" + '\n')
        elif i % 3 == 0:
            file.write("fizz" + '\n')
        elif i % 5 == 0:
            file.write("buzz" + '\n')
        else:
            file.write(str(i) + '\n')
            
    file.close()
    

logic_fizzbuzz(100)

test_input = open('test_input.txt','r').read().strip()
string = test_input.split("\n")
values = [int(x) for x in string]

test_x = np.array([binary_encode(x,10) for x in values])
test_y = np.array([fizz_buzz_encode(x) for x in values])

answer = model1.predict(test_x)

validation = model1.evaluate(test_x,test_y)


def fizz_buzz_ML(N):
    file = open("software2.0.txt",'w')
    for i in range(1,N):
        index = np.unravel_index(answer[i-1].argmax(),answer[i-1].shape)
        
        if index == (0,):
            file.write("fizzbuzz" + '\n')
        elif index == (1,):
            file.write("fizz" + '\n')
        elif index == (2,):
            file.write("buzz" + '\n')
        else:
            file.write(str(i) + '\n')
    file.close()
            
fizz_buzz_ML(101)
