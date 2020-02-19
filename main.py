import tensorflow as tf 
import numpy as np 
fashion_mnist = tf.keras.datasets.fashion_mnist 
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()

test_images = test_images / 255.0 



json_file1 = open('parameters1.json', 'r')
loaded_model_json1 = json_file1.read()
json_file1.close()

model1 = tf.keras.models.model_from_json(loaded_model_json1,custom_objects = None )

model1.load_weights("weight_values1.h5")
model1.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),metrics = ['accuracy'])
answer1 = model1.predict(test_images)
test_loss1 , test_acc1 = model1.evaluate(test_images,test_labels)
test_acc1 = round(test_acc1,4)



json_file2 = open('parameters2.json', 'r')
loaded_model_json2 = json_file2.read()
json_file2.close()

model2 = tf.keras.models.model_from_json(loaded_model_json2,custom_objects = None )

model2.load_weights("weight_values2.h5")
model2.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),metrics = ['accuracy'])


test_images = test_images.reshape(10000,28,28,1)
answer2 = model2.predict(test_images)
test_loss2 , test_acc2 = model2.evaluate(test_images,test_labels)
test_acc2 = round(test_acc2,4)


file1 = open("multi-layer-net.txt",'w')
file2 = open("convolution-neural-net.txt",'w')


file1.write("Loss on Test Data : {}\n".format(test_loss1))
file1.write("Accuracy on Test Data : {}\n".format(test_acc1))
file1.write("gt_label,pred_label \n")
for i in range(10000):
    file1.write("{},{}\n".format(test_labels[i],np.argmax(answer1[i])))
    
file1.close()


file2.write("Loss on Test Data : {}\n".format(test_loss2))
file2.write("Accuracy on Test Data : {}\n".format(test_acc2))
file2.write("gt_label,pred_label \n")
for i in range(10000):
    file2.write("{},{}\n".format(test_labels[i],np.argmax(answer2[i])))

file2.close()
