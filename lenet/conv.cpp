#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include <iomanip>

#include <Halide.h>

using Halide::Image;
#include "image_io.h"

// Constants
const int NUM_IMAGES = 5;

const int IMAGE_SIZE = 28;
const int REDUCE_IMAGE_SIZE = 4;

const int FILTER_SIZE = 5;
const int POOL_SIZE = 2;

const int LAYER0_NODES = 1;
const int LAYER1_NODES = 4;
const int LAYER2_NODES = 6;
const int LAYER3_NODES = 4;
const int LAYER4_NODES = 10;

Halide::Func convolution_layer(Halide::Func input, Halide::Func weights,
    Halide::Func bias, int filter_size, int input_layers, int pool_size) {

    // Convolution
    Halide::Func convolution;
    Halide::Var x, y, z, w;
    Halide::RDom r(0, filter_size, 0, filter_size, 0, input_layers);

    convolution(x, y, z, w) = 0.0f;
    convolution(x, y, z, w) += weights(r.x, r.y, r.z, z) * input(x + r.x, y + r.y, r.z, w);

    // Max pool
    Halide::Func subsample;
    Halide::RDom s(0, pool_size, 0, pool_size);
    subsample(x, y, z, w) = convolution(pool_size * x, pool_size * y, z, w);
    subsample(x, y, z, w) = Halide::max(convolution(pool_size * x + s.x,
        pool_size * y + s.y, z, w), subsample(x, y, z, w));

    // Non-linear bias
    Halide::Func biased;
    biased(x, y, z, w) = tanh(subsample(x, y, z, w) + bias(z, 0));

    return biased;
}

Halide::Func flatten(Halide::Func input, int dim3_size, int dim2_size) {
    
    Halide::Func flatten1, flatten2;
    Halide::Var x, y, z;

    flatten1(x, y, z) = input(x, y / dim3_size, y % dim3_size, z);
    
    // Only y = 0 should be used
    flatten2(x, y, z) = flatten1(x / dim2_size, x % dim2_size, z);

    return flatten2;
}

Halide::Func fully_connected_layer(Halide::Func input, Halide::Func weights,
    Halide::Func bias, int size) {

    Halide::Func product;
    Halide::Var x, y, z;
    Halide::RDom r(0, size);

    // Only y = 0 should be used
    product(x, y, z) = 0.0f;
    product(x, y, z) += weights(x, r.x) * input(r.x, y, z);
    product(x, y, z) = tanh(product(x, y, z) + bias(x, 0));

    return product;
}

Halide::Func classification(Halide::Func input, int size) {

    Halide::Func softmax;
    Halide::Var x, y, z;

    softmax(x, y, z) = exp(input(x, y, z)); // Ignore normalization
    
    Halide::Func classification;
    Halide::RDom r(0, size);
    classification(x, y, z) = Halide::argmax(softmax(r.x, 0, z))[0];

    return classification;
}

void classify(Halide::Func layer0, Halide::Func *weights, 
    Halide::Func *bias) {

    // Layer 0 -- Convolution
    Halide::Func layer1 = convolution_layer(layer0, weights[0], bias[0],
        FILTER_SIZE, LAYER0_NODES, POOL_SIZE);

    // Layer 1 -- Convolution
    Halide::Func layer2 = convolution_layer(layer1, weights[1], bias[1],
        FILTER_SIZE, LAYER1_NODES, POOL_SIZE);

    // Flatten many feature maps onto a single level for future layers
    Halide::Func flattened = flatten(layer2, LAYER2_NODES, 
        LAYER2_NODES * REDUCE_IMAGE_SIZE);
    
    // Layer 2 -- Fully connected hidden layer
    Halide::Func layer3 = fully_connected_layer(flattened, weights[2],
        bias[2], LAYER2_NODES * REDUCE_IMAGE_SIZE * REDUCE_IMAGE_SIZE);
    

    // Layer 3 -- Logistic Softmax
    Halide::Func layer4 = fully_connected_layer(layer3, weights[3],
        bias[3], LAYER3_NODES);

    // Layer 4 -- Maximum node / classification
    Halide::Func layer5 = classification(layer4, LAYER4_NODES);
   
    // Realize to perform computation
    Halide::Image<int32_t> output(1, 1, NUM_IMAGES);
    layer5.realize(output);
    std::cout << output(0, 0, 0) << std::endl;
}
    
int main(int argc, char **argv) {
    
    // Load weight images
    // Weights are stored in Image<T> types with dimensions:
    // row value, column value, input layer number, output layer number
    Halide::Image<float> layer0_weights_image(FILTER_SIZE, FILTER_SIZE, 
        LAYER0_NODES, LAYER1_NODES);
    for (int i = 0; i < LAYER0_NODES; i++) {
        for (int j = 0; j < LAYER1_NODES; j++) {
            std::string filename = 
                "res/l0w" + std::to_string(LAYER1_NODES * i + j) + ".png";
            Halide::Image<float> weight = load<float>(filename);
            for (int x = 0; x < FILTER_SIZE; x ++) {
                for (int y = 0; y < FILTER_SIZE; y++) {
                    layer0_weights_image(x, y, i, j) = 
                        (255 * weight(x, y) - 127) / 127.0f;
                }
            }
        }
    }

    Halide::Image<float> layer1_weights_image(FILTER_SIZE, FILTER_SIZE, 
        LAYER1_NODES, LAYER2_NODES);
    for (int i = 0; i < LAYER1_NODES; i++) {
        for (int j = 0; j < LAYER2_NODES; j++) {
            std::string filename = 
                "res/l1w" + std::to_string(LAYER2_NODES * i + j) + ".png";
            Halide::Image<float> weight = load<float>(filename);
            for (int x = 0; x < FILTER_SIZE; x ++) {
                for (int y = 0; y < FILTER_SIZE; y++) {
                    layer1_weights_image(x, y, i, j) = 
                        (255 * weight(x, y) - 127) / 127.0f;
                }
            }
        }
    }

    // Load weight functions
    Halide::Var x, y, z, w;
    Halide::Func layer0_weights;
    Halide::Func layer1_weights;
    layer0_weights(x, y, z, w) = layer0_weights_image(x, y, z, w);
    layer1_weights(x, y, z, w) = layer1_weights_image(x, y, z, w);

    Halide::Image<float> layer2_weight_input = load<float>("res/l2w.png");
    Halide::Func layer2_weights;
    layer2_weights(x, y) = 
        (255 * layer2_weight_input(x, y) - 127) / 127.0f;

    Halide::Image<float> layer3_weight_input = load<float>("res/l3w.png");
    Halide::Func layer3_weights;
    layer3_weights(x, y) = 
        (255 * layer3_weight_input(x, y) - 127) / 127.0f;

    // Load biases
    Halide::Image<float> layer0_bias_input = load<float>("res/l0b.png");
    Halide::Func layer0_bias;
    layer0_bias(x, y) = (255*layer0_bias_input(x, y) - 127) / 127.0f;

    Halide::Image<float> layer1_bias_input = load<float>("res/l1b.png");
    Halide::Func layer1_bias;
    layer1_bias(x, y) = (255*layer1_bias_input(x, y) - 127) / 127.0f;

    Halide::Image<float> layer2_bias_input = load<float>("res/l2b.png");
    Halide::Func layer2_bias;
    layer2_bias(x, y) = (255*layer2_bias_input(x, y) - 127) / 127.0f;

    Halide::Image<float> layer3_bias_input = load<float>("res/l3b.png");
    Halide::Func layer3_bias;
    layer3_bias(x, y) = (255*layer3_bias_input(x, y) - 127) / 127.0f;

    Halide::Func weights[4] = {layer0_weights, layer1_weights,
        layer2_weights, layer3_weights};
    Halide::Func bias[4] = {layer0_bias, layer1_bias, layer2_bias,
        layer3_bias};

    // Load large tiled image for batch classifiation
    Halide::Func layer0;
    Halide::Image<float> input(28, 28, 1, NUM_IMAGES);
    for (int i = 0; i < NUM_IMAGES; i++) {
        std::string filename = "mnist/1-" + std::to_string(i) + ".png";
        Halide::Image<float> image = load<float>(filename);
        for (int x = 0; x < 28; x++) {
            for (int y = 0; y < 28; y++) {
                input(x, y, 0, i) = image(x, y);
            }
        }
    }
    layer0(x, y, z, w) = input(x, y, z, w);
    classify(layer0, weights, bias);
}
