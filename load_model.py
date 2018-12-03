import numpy as np
import tensorflow as tf
from create_model import create_model
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

mnist = tf.keras.datasets.mnist

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

fig = plt.figure()
for i in range(9):
	rnd = np.random.randint(0, len(x_test))
	expected = y_test[rnd]
	test = x_test[rnd].reshape(1, 784)

	prediction = model.predict(test)
	predicted = np.argmax(prediction)
	plt.subplot(3, 3, i+1)
	plt.tight_layout()
	plt.imshow(x_test[rnd], cmap='gray', interpolation='none')
	plt.title("Digit: {}, Prediction: {}".format(y_test[rnd], predicted))
	plt.xticks([])
	plt.yticks([])
plt.show()

# print(type(test))

# for i, p in enumerate(prediction[0]):
# print i, p
