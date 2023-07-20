#include "neural.h"

int main() {

    srand(time(NULL));

    /* Input */
    Matrix* X = initMatrix(4, 2);
    X->data[0] = 0;
    X->data[1] = 0;
    X->data[2] = 0;
    X->data[3] = 1;
    X->data[4] = 1;
    X->data[5] = 0;
    X->data[6] = 1;
    X->data[7] = 1;

    /* Output */
    Matrix* y = initMatrix(4, 1);
    y->data[0] = 0;
    y->data[1] = 1;
    y->data[2] = 1;
    y->data[3] = 0;

    /* Layers */
    Layer* layer1 = initLayer(2, 4);
    Layer* layer2 = initLayer(4, 1);

    /* Variables for storing outputs */
    Matrix* sig1_output = initMatrix(1, 1); 
    Matrix* sig2_output = initMatrix(1, 1);
    Matrix* dvalues = initMatrix(1, 1);

    /* Training epochs */
    for(int i = 0; i < 10000; i++) {

        /* Forward pass */
        layer1->output = dot(X, layer1->weights);
        sig1_output = sigmoid(layer1->output);
        layer2->output = dot(sig1_output, layer2->weights);
        sig2_output = sigmoid(layer2->output);

        printf("Error: %f\n", MSE_Loss(sig2_output, y));

        /* Backward pass */
        dvalues = derivative_MSE_Loss(sig2_output, y);
        dvalues = hadamard_product(dvalues, derivative_sigmoid(layer2->output));
        layer2->dweights = dot(transpose(sig1_output), dvalues);
        dvalues = dot(dvalues, transpose(layer2->weights));
        dvalues = hadamard_product(dvalues, derivative_sigmoid(layer1->output));
        layer1->dweights = dot(transpose(X), dvalues);

        /* Weight updates */
        layer1->weights = update_weights_SGD(layer1->weights, layer1->dweights, 1);
        layer2->weights = update_weights_SGD(layer2->weights, layer2->dweights, 1);

    }

    Matrix* X_test = initMatrix(1, 2);
    X_test->data[0] = 0;
    X_test->data[1] = 1;

    layer1->output = dot(X_test, layer1->weights);
    sig1_output = sigmoid(layer1->output);
    layer2->output = dot(sig1_output, layer2->weights);
    sig2_output = sigmoid(layer2->output);

    printf("Result:\n");
    printMatrix(sig2_output);

}