#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include <iomanip>

#include <Halide.h>

using Halide::Image;
#include "image_io.h"

Halide::Func bias3(Halide::Func input, Halide::Func bias) {
    
    Halide::Func biased;
    Halide::Var x, y, z;

    biased(x, y, z) = tanh(input(x, y, z) + bias(z, 0));

    return biased;
}

Halide::Func bias2(Halide::Func input, Halide::Func bias) {
    
    Halide::Func biased;
    Halide::Var x, y;

    biased(x, y) = tanh(input(x, y) + bias(y, 0));

    return biased;
}

Halide::Func maxPool(Halide::Func input, int pool_size) {

    Halide::Func subsample;
    Halide::Var x, y, c;
    
    Halide::RDom r(0, pool_size, 0, pool_size);
    
    subsample(x, y, c) = input(pool_size * x, pool_size * y, c);
    subsample(x, y, c) = Halide::max(input(pool_size*x + r.x, pool_size*y + r.y, c),
        subsample(x,y,c));
    
    return subsample;
}

Halide::Func convolution(Halide::Func input, Halide::Func weights,
    int filter_size, int input_layers) {

    Halide::Func convolution;
    Halide::Var x, y, z;

    Halide::RDom r(0, filter_size, 0, filter_size, 0, input_layers);

    convolution(x, y, z) = 0.0f;
    convolution(x, y, z) += weights(r.x, r.y, r.z, z) * input(x + r.x, y + r.y, r.z);

    return convolution;
}

Halide::Func flatten(Halide::Func input, int dim3_size, int dim2_size) {
    
    Halide::Func flatten1, flatten2;
    Halide::Var x, y;

    flatten2(x, y) = 0.0f;
    flatten1(x, y) = input(x, y / dim3_size, y % dim3_size);
    flatten2(x, 0) = flatten1(x / dim2_size, x % dim2_size);

    return flatten2;
}

Halide::Func dotproduct(Halide::Func input, Halide::Func weights, 
    int size) {

    Halide::Func product;
    Halide::Var x, y;
    Halide::RDom r(0, size);

    product(x, y) = 0.0f;
    product(x, y) += weights(r.x, x) * input(r.x, 0);

    return product;
}

Halide::Func dotproduct_final(Halide::Func input, Halide::Func weights, 
    int size) {

    Halide::Func product;
    Halide::Var x, y;
    Halide::RDom r(0, size);

    product(x, y) = 0.0f;
    product(x, y) += weights(x, r.x) * input(y, r.x);

    return product;
}

Halide::Func softmax(Halide::Func input, int size) {

    Halide::Func softmax;
    Halide::Var x, y;

    softmax(x, y) = exp(input(x, y)); // Ignore normalization

    return softmax;
}

int classify(std::string filename, Halide::Func *weights, 
    Halide::Func *biases) {
    Halide::Image<float> input = load<float>(filename);

    Halide::Var x, y, z;
    Halide::Func originalImage;
    originalImage(x, y, z) = 0.0f;
    originalImage(x, y, 0) = input(x, y);

    // Layer 0 -- Convolution
    Halide::Func layer00 = convolution(originalImage, weights[0], 5, 1);
    Halide::Func layer01 = maxPool(layer00, 2);
    Halide::Func layer02 = bias3(layer01, biases[0]);

    // Layer 1 -- Convolution
    Halide::Func layer10 = convolution(layer02, weights[1], 5, 4);
    Halide::Func layer11 = maxPool(layer10, 2);
    Halide::Func layer12 = bias3(layer11, biases[1]);
    
    // Layer 2 -- Fully connected hidden layer
    Halide::Func layer20 = flatten(layer12, 6, 24);
    Halide::Func layer21 = dotproduct(layer20, weights[2], 96);
    Halide::Func layer22 = bias2(layer21, biases[2]);
    
    // Layer 3 -- Logistic Softmax
    Halide::Func layer30 = dotproduct_final(layer22, weights[3], 4);
    Halide::Func layer31 = bias2(layer30, biases[3]);

    // Layer 4 -- Maximum node / classification
    Halide::Func layer40 = softmax(layer31, 10);
    Halide::Func layer41;
    Halide::RDom r(0, 10);
    layer41(x, y) = Halide::argmax(layer40(r.x, 0))[0];
    
    //Halide::Image<int32_t> output(1, 1);
    //layer41.realize(output);
    //std::cout << output(0, 0) << std::endl;
     
    Halide::Image<float> output(10, 1);
    layer40.realize(output);
    std::cout << output(0, 0) << std::endl;
    std::cout << output(1, 0) << std::endl;
    std::cout << output(2, 0) << std::endl;
    std::cout << output(3, 0) << std::endl;
    std::cout << output(4, 0) << std::endl;
    std::cout << output(5, 0) << std::endl;
    std::cout << output(6, 0) << std::endl;
    std::cout << output(7, 0) << std::endl;
    std::cout << output(8, 0) << std::endl;
    std::cout << output(9, 0) << std::endl;
    
    return output(0, 0);
}
    
int main(int argc, char **argv) {
    
    // Load weight images
    // Weights are stored in Image<T> types with dimensions:
    // row value, column value, input layer number, output layer number
    Halide::Image<float> layer0_weights_image(5, 5, 1, 4);
    for (int i = 0; i < 4; i++) {
        std::string filename = "res/l0w" + std::to_string(i) + ".png";
        Halide::Image<float> weight = load<float>(filename);
        for (int x = 0; x < 5; x ++) {
            for (int y = 0; y < 5; y++) {
                layer0_weights_image(x, y, 0, i) = (255*weight(x, y) - 127) / 127.0f;
            }
        }
    }

    Halide::Image<float> layer1_weights_image(5, 5, 4, 6);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 6; j++) {
            std::string filename = "res/l1w" + std::to_string(6 * i + j) + ".png";
            Halide::Image<float> weight = load<float>(filename);
            for (int x = 0; x < 5; x ++) {
                for (int y = 0; y < 5; y++) {
                    layer1_weights_image(x, y, i, j) = (255*weight(x, y) - 127) / 127.0f;
                }
            }
        }
    }

    Halide::Func layer0_weights;
    Halide::Func layer1_weights;
    Halide::Var a1, a2, a3, a4;
    layer0_weights(a1, a2, a3, a4) = layer0_weights_image(a1,a2,a3,a4);
    layer1_weights(a1, a2, a3, a4) = layer1_weights_image(a1,a2,a3,a4);

    Halide::Image<float> layer2_weight_input = load<float>("res/l2w.png");
    Halide::Var w20, w21;
    Halide::Func layer2_weights;
    layer2_weights(w20, w21) = (255*layer2_weight_input(w20, w21) - 127) / 127.0f;

    Halide::Image<float> layer3_weight_input = load<float>("res/l3w.png");
    Halide::Var w30, w31;
    Halide::Func layer3_weights;
    layer3_weights(w30, w31) = (255*layer3_weight_input(w30, w31) - 127) / 120.0f;

    // Load biases
    Halide::Var bx, by;
    Halide::Image<float> layer0_bias_input = load<float>("res/l0b.png");
    Halide::Func layer0_bias;
    layer0_bias(bx, by) = (255*layer0_bias_input(bx, by) - 127) / 127.0f;

    Halide::Image<float> layer1_bias_input = load<float>("res/l1b.png");
    Halide::Func layer1_bias;
    layer1_bias(bx, by) = (255*layer1_bias_input(bx, by) - 127) / 127.0f;

    Halide::Image<float> layer2_bias_input = load<float>("res/l2b.png");
    Halide::Func layer2_bias;
    layer2_bias(bx, by) = (255*layer2_bias_input(bx, by) - 127) / 127.0f;

    Halide::Image<float> layer3_bias_input = load<float>("res/l3b.png");
    Halide::Func layer3_bias;
    layer3_bias(bx, by) = (255*layer3_bias_input(bx, by) - 127) / 127.0f;

    Halide::Func weights[4] = {layer0_weights, layer1_weights,
        layer2_weights, layer3_weights};
    Halide::Func biases[4] = {layer0_bias, layer1_bias, layer2_bias,
        layer3_bias};

    for (int i = 0; i < 5; i++) {
        std::string filename = "mnist/1-" + std::to_string(i) + ".png";
        classify(filename, weights, biases);
    }
}
