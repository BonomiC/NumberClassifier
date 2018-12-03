from mnist import MNIST
import random

mndata = MNIST('training_data')
mndata.gz = True
images, labels = mndata.load_training()

index = random.randrange(0, len(images))
print(mndata.display(images[index]))
print(labels[index])

print(images[index])
