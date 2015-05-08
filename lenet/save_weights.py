import sys
import numpy
from PIL import Image

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print "Usage: python save_weights.py <plain text>"
        exit(0)
    
    text_input = sys.argv[1]

    layer0_weights = []
    layer0_biases = []
    layer1_weights = []
    layer1_biases = []
    layer2_weights = []
    layer2_biases = []
    layer3_weights = []
    layer3_biases = []

    with open(text_input, 'r') as inputfile:
        current = ""
        weight = []
        for line in inputfile:
            line = line.strip()
            if len(line) == 0:
                continue

            if line[0] == '!':
                current = line[1:]
                continue

            line = line.replace('[', '')
            line = line.replace(']', '')
            line = line.split()
            values = [int(127 * float(x)) + 127 for x in line]

            if current == "w0" or current == "w1":
                    # Collect 25 numbers
                    weight.extend(values)

                    if len(weight) == 25:
                        weight = numpy.asarray(weight)
                        weight = weight.reshape(5, 5)
                        if current == "w0":
                            layer0_weights.append(weight)
                        else:
                            layer1_weights.append(weight)
                        weight = []
            else:
                if current == "b0":
                    layer0_biases.extend(values)
                elif current == "b1":
                    layer1_biases.extend(values)
                elif current == "b2":
                    layer2_biases.extend(values)
                elif current == "b3":
                    layer3_biases.extend(values)
                elif current == "w2":
                    layer2_weights.extend(values)
                elif current == "w3":
                    layer3_weights.extend(values)

    layer0_biases = numpy.asarray([layer0_biases], dtype=numpy.uint8)
    layer1_biases = numpy.asarray([layer1_biases], dtype=numpy.uint8)
    layer2_biases = numpy.asarray([layer2_biases], dtype=numpy.uint8)
    layer3_biases = numpy.asarray([layer3_biases], dtype=numpy.uint8)
    layer0_weights = numpy.asarray(layer0_weights, dtype=numpy.uint8)
    layer1_weights = numpy.asarray(layer1_weights, dtype=numpy.uint8)

    layer2_weights_array = numpy.ndarray((96, 4), dtype=numpy.uint8)
    for i in xrange(4 * 96):
        layer2_weights_array[i / 4][i % 4] = layer2_weights[i]
    layer2_weights = layer2_weights_array

    layer3_weights_array = numpy.ndarray((4, 10), dtype=numpy.uint8)
    for i in xrange(4 * 10):
        layer3_weights_array[i / 10][i % 10] = layer3_weights[i]
    layer3_weights = layer3_weights_array

    # Print sizes to check
    print layer0_biases.shape
    print layer1_biases.shape
    print layer2_biases.shape
    print layer3_biases.shape
    print layer0_weights.shape
    print layer1_weights.shape
    print layer2_weights.shape
    print layer3_weights.shape

    print layer3_weights
    print layer3_biases

    for i, image in enumerate(layer0_weights):
        Image.fromarray(image).convert('L').save("res/l0w%d.png" % i)        
    for i, image in enumerate(layer1_weights):
        Image.fromarray(image).convert('L').save("res/l1w%d.png" % i)
    Image.fromarray(layer2_weights).convert('L').save("res/l2w.png")
    Image.fromarray(layer3_weights).convert('L').save("res/l3w.png")
    Image.fromarray(layer0_biases).convert('L').save("res/l0b.png")
    Image.fromarray(layer1_biases).convert('L').save("res/l1b.png")
    Image.fromarray(layer2_biases).convert('L').save("res/l2b.png")
    Image.fromarray(layer3_biases).convert('L').save("res/l3b.png")
