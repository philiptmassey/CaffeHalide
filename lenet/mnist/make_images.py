import struct
import numpy
from PIL import Image

labels = []
with open("t10k-labels-idx1-ubyte", 'rb') as inputfile:
    data = inputfile.read();
    data = data[8:]

    for byte in data:
        labels.append(ord(byte))

pictures = [[] for x in xrange(10)]
with open("t10k-images-idx3-ubyte", 'rb') as inputfile:
    data = inputfile.read()
    data = data[16:]
    size = 28 * 28

    for i in xrange(0, len(data) / size):
        index = i * size
        picture = data[index : index + size]
        picture = [ord(x) for x in picture]
        picture = numpy.asarray(picture, dtype=numpy.uint8).reshape(28, 28)
        pictures[labels[i]].append(picture)

   

for i in xrange(10):
    for j in xrange(len(pictures[i])):
        Image.fromarray(pictures[i][j]).convert("L").save("%d-%d.png" % (i, j))
