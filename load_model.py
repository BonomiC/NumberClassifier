import numpy as np
import tensorflow as tf
mnist = tf.keras.datasets.mnist

from create_model import create_model

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = create_model()

model.load_weights('./checkpoint/saved_weights')

loss, acc = model.evaluate(x_test, y_test)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
model.summary()

for i in range(10):
	rnd = np.random.randint(0, len(x_test))
	expected = y_test[rnd]
	test = x_test[rnd].reshape(1, 784)
	prediction = model.predict(test)
	predicted = np.argmax(prediction)
	print "Expected: ", expected
	print "Predicted: ", predicted
	print "****************"
	if expected == predicted:
		print "CORRECT"
	else:
		print "INCORRECT"
	print "****************"

# print(type(test))

# for i, p in enumerate(prediction[0]):
	# print i, p