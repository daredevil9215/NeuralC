#include "matrix.h"
#include "classes.h"

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

    Matrix* sig1_output = initMatrix(1, 1); 
    Matrix* sig2_output = initMatrix(1, 1);
    Matrix* dvalues = initMatrix(1, 1);

    for(int i = 0; i < 10000; i++) {

        /* Forward pass */
        layer1->output = dot(ulaz, layer1->weights);
        sig1_output = sigmoid(layer1->output);
        layer2->output = dot(sig1_output, layer2->weights);
        sig2_output = sigmoid(layer2->output);

        printf("Error: %f\n", MSE_Loss(sig2_output, izlaz));

        /* Backward pass */
        dvalues = derivative_MSE_Loss(sig2_output, izlaz);
        dvalues = hadamard_product(dvalues, derivative_sigmoid(layer2->output));
        layer2->dweights = dot(transpose(sig1_output), dvalues);
        dvalues = dot(dvalues, transpose(layer2->weights));
        dvalues = hadamard_product(dvalues, derivative_sigmoid(layer1->output));
        layer1->dweights = dot(transpose(ulaz), dvalues);

        layer1->weights = update_weights(layer1->weights, layer1->dweights, 10);
        layer2->weights = update_weights(layer2->weights, layer2->dweights, 10);

    }

    layer1->output = dot(ulaz, layer1->weights);
    sig1_output = sigmoid(layer1->output);
    layer2->output = dot(sig1_output, layer2->weights);
    sig2_output = sigmoid(layer2->output);

    printf("Rezultat:\n");
    printMatrix(sig2_output);
    printf("Greska:\n");
    printMatrix(MSE_Loss_matrix(sig2_output, izlaz));


}