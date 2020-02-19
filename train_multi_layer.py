import tensorflow as tf
import numpy as np
#from tensorflow.keras.preprocessing.image import ImageDataGenerator 

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()

train_images , test_images = train_images / 255.0 , test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = [28,28]),
    tf.keras.layers.Dense(128,activation = 'relu'),
    tf.keras.layers.Dense(64,activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,activation = 'softmax')

])

model.compile(optimizer = 'adam' ,
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ['accuracy'])

'''
generator = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    zoom_range = 0.02
    )
train_images = train_images.reshape(60000,28,28,1)
test_images = test_images.reshape(10000,28,28,1)

generator.fit(train_images)
model.fit_generator(generator.flow(train_images, train_labels, batch_size=32),
                    steps_per_epoch=len(train_images) / 32, epochs= 15)

'''
model.summary()
model.fit(train_images , train_labels , epochs = 50)
model.evaluate(test_images , test_labels)


model_json = model.to_json()
with open("parameters1.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("weight_values1.h5")
