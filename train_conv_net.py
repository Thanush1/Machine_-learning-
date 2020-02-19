import tensorflow as tf
import numpy as np
#from tensorflow.keras.preprocessing.image import ImageDataGenerator 

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images,train_labels) , (test_images,test_labels) = fashion_mnist.load_data()

train_images = train_images.reshape(60000,28,28,1)
test_images = test_images.reshape(10000,28,28,1)

train_images = train_images / 255.0
test_images = test_images / 255.0


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),activation = 'relu',input_shape = [28,28,1]),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.12),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation = 'relu'),
    tf.keras.layers.Dense(10,activation = 'softmax')
])


model.summary()


model.compile(optimizer = 'adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ['accuracy']
             )

'''
generator = ImageDataGenerator(
    rotation_range= 5 ,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    #shear_range = 0.05 ,
    zoom_range = 0.02
    )
generator.fit(train_images)
model.fit_generator(generator.flow(train_images, train_labels, batch_size=32),
                    steps_per_epoch=len(train_images) / 32, epochs= 5)
'''

model.fit(train_images,train_labels,epochs = 10)


model.evaluate(test_images,test_labels)


model_json = model.to_json()

with open("parameters2.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("weight_values2.h5")
