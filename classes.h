typedef struct Matrix {

    int rows, columns;
    double* data;

} Matrix;

typedef struct Layer {

    Matrix* weights;
    Matrix* output;
    Matrix* dweights;
    Matrix* dinputs;
    
} Layer;