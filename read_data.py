from mnist import MNIST
import numpy as np

def read_data():
	mndata = MNIST('training_data')
	mndata.gz = True
	images, labels = mndata.load_training()
	
	# print(mndata.display(images[0]))

	np_images = np.zeros((len(images), 784))
	np_labels = np.zeros((len(labels), 10))

	for idx, image in enumerate(images):
		np_images[idx] = images[idx]
		np_labels[idx][labels[idx]] = 1

	print(labels[0])
	print(np_labels[0])

read_data()