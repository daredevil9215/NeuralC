#include "matrix.h"

int main() {
    srand(time(NULL));
    struct Matrix* mat1 = initMatrix(2, 1);
    mat1->data[0] = 0;
    mat1->data[1] = 1;
    struct Matrix* mat2 = initMatrix(2,1);
    mat2 = relu(mat2);
    printMatrix(mat2);
    printf("%f\n", BCE_Loss(mat1, mat2));
}