import numpy as np
import tensorflow as tf
from create_model import create_model
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')

import scipy.misc

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

for i in range(1):
	print x_test[i]

	for j in range(28):
		for k in range(28):
			if x_test[i][j][k] > 0.04:
				x_test[i][j][k] = 1
			else:
				x_test[i][j][k] = 0

	print x_test[i]

	scipy.misc.toimage(x_test[i], cmin=0.0, cmax=1.0).save("mod_test_out.jpg")

# fig = plt.figure()
# for i in range(1):
# 	test = x_test[i].reshape(1, 784)

# 	# print test

# 	# for j in range(784):
# 		# test[0][j] = 1

# 	# print test

# 	print x_test[i]

# 	print "*****************"
# 	for j in range(28):
# 		for k in range(28):
# 			x_test[i][j][k] = 0

# 	print x_test[i]

# 	plt.subplot(3, 3, i+1)
# 	plt.tight_layout()
# 	plt.imshow(x_test[i], cmap='gray', interpolation='none')
# 	plt.title("Digit: {}".format(y_test[i]))
# 	plt.xticks([])
# 	plt.yticks([])
# plt.show()

# print(type(test))

# for i, p in enumerate(prediction[0]):
# print i, p
