from mnist import MNIST
import numpy as np

def read_data():
	mndata = MNIST('training_data')
	mndata.gz = True
	images, labels = mndata.load_training()
	
	# print(mndata.display(images[0]))

	# print(images[0])

	np_images = np.zeros((len(images), 784))
	np_labels = np.zeros((len(labels), 10))

	for i, image in enumerate(images):
		for j, num in enumerate(image):
			normalized = np.interp(num, [0, 255], [0, 1])
			images[i][j] = normalized

		# print(i)
		np_images[i] = images[i]
		np_labels[i][labels[i]] = 1

		if i > 500:
			break

	# print(images[0])

	print(labels[0])
	print(np_labels[0])

	return (np_images, np_labels)
