# Vi har fått in random kordinater nu får vi in bilder.
# Dom är klassificerade bild = siffra
# Kommer då bli 10 classification areas
# Baserat på vad?!


import random
from mnist import MNIST # This only helps with reading the files weirdidx3-ubyte

mndata = MNIST('samples/')
images, labels = mndata.load_training()
index = random.randrange(0, len(images))  # choose an index ;-)
print(mndata.display(images[index]))
print(images[index])
print(labels[index])
