#include "matrix.h"

int main() {

    srand(time(NULL));

    Matrix* ulaz = initMatrix(4, 2);
    ulaz->data[0] = 0;
    ulaz->data[1] = 0;
    ulaz->data[2] = 0;
    ulaz->data[3] = 1;
    ulaz->data[4] = 1;
    ulaz->data[5] = 0;
    ulaz->data[6] = 1;
    ulaz->data[7] = 1;

    Matrix* izlaz = initMatrix(4, 1);
    izlaz->data[0] = 0;
    izlaz->data[1] = 1;
    izlaz->data[2] = 1;
    izlaz->data[3] = 0;

    Layer* layer1 = initLayer(2, 10);
    Layer* layer2 = initLayer(10, 1);

    Matrix* weights1 = initMatrix(2, 10);
    Matrix* weights2 = initMatrix(10, 1);

    Matrix* layer1_dweights = initMatrix(1, 1); 
    Matrix* layer1_output = initMatrix(1, 1);
    Matrix* sig1_output = initMatrix(1, 1); 
    Matrix* layer2_dweights = initMatrix(1, 1); 
    Matrix* layer2_output = initMatrix(1, 1);
    Matrix* sig2_output = initMatrix(1, 1);
    Matrix* dvalues = initMatrix(1, 1);

    for(int i = 0; i < 5000; i++) {

        /* Forward pass */
        layer1_output = dot(ulaz, weights1);
        sig1_output = sigmoid(layer1_output);
        layer2_output = dot(sig1_output, weights2);
        sig2_output = sigmoid(layer2_output);

        printf("Error: %f\n", MSE_Loss(sig2_output, izlaz));

        /* Backward pass */
        dvalues = derivative_MSE_Loss(sig2_output, izlaz);
        dvalues = hadamard_product(dvalues, derivative_sigmoid(layer2_output));
        layer2_dweights = dot(transpose(sig1_output), dvalues);
        dvalues = dot(dvalues, transpose(weights2));
        dvalues = hadamard_product(dvalues, derivative_sigmoid(layer1_output));
        layer1_dweights = dot(transpose(ulaz), dvalues);

        weights1 = update_weights(weights1, layer1_dweights, 10);
        weights2 = update_weights(weights2, layer2_dweights, 10);

    }

    layer1_output = dot(ulaz, weights1);
    sig1_output = sigmoid(layer1_output);
    layer2_output = dot(sig1_output, weights2);
    sig2_output = sigmoid(layer2_output);

    printf("Rezultat:\n");
    printMatrix(sig2_output);
    printf("Greska:\n");
    printMatrix(MSE_Loss_matrix(sig2_output, izlaz));


}