import tensorflow as tf
# import keras
mnist = tf.keras.datasets.mnist
# from keras import backend as K
# sess = tf.Session()
# K.set_session(sess)
# from keras.utils import np_utils


from create_model import create_model

# import tensorflowjs as tfjs

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)
x_train, x_test = x_train / 255.0, x_test / 255.0

# y_train = np_utils.to_categorical(y_train, 10)
# y_test = np_utils.to_categorical(y_test, 10)

model = create_model()

model.fit(x_train, y_train, epochs=10)
loss, acc = model.evaluate(x_test, y_test)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
model.summary()

model.save_weights('./checkpoint/saved_weights', overwrite=True)

# tfjs.converters.save_keras_model(model, "tfjsmode1")