#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "classes.h"

/* Random numbers */

double rand_gen() {
   return ( (double)(rand()) + 1. )/( (double)(RAND_MAX) + 1. );
}

double normalRandom() {
   double v1=rand_gen();
   double v2=rand_gen();
   return cos(2*3.14*v2)*sqrt(-2.*log(v1));
}

/* Matrix operations */

Matrix* initMatrix(int n, int m) {

    Matrix* mat = (Matrix*)malloc(sizeof( Matrix));
    mat->rows = n;
    mat->columns = m;
    mat->data = (double*)malloc(sizeof(double) * n * m);
    for(int i = 0; i < n * m; i++) {
        mat->data[i] = normalRandom();
    }
    return mat;

}

Matrix* dot(Matrix* mptr1, Matrix* mptr2) {
    if (mptr1->columns == mptr2->rows) {
        Matrix* result = (Matrix*)malloc(sizeof(Matrix));
        result->rows = mptr1->rows;
        result->columns = mptr2->columns;
        result->data = (double*)calloc(result->rows * result->columns, sizeof(double));
        for(int i = 0; i < mptr1->rows; i++) {
            for(int j = 0; j < mptr2->columns; j++) {
                for(int k = 0; k < mptr2->rows; k++) {
                    result->data[i * result->columns + j] += mptr1->data[i * mptr1->columns + k] * mptr2->data[k * mptr2->columns + j];
                }
                
            }
        }
        return result;
    }
}

Matrix* transpose( Matrix* mptr) {
    Matrix* result = (Matrix*)malloc(sizeof(Matrix));
    result->rows = mptr->columns;
    result->columns = mptr->rows;
    result->data = malloc(sizeof(double) * result->rows * result->columns);
    for(int i = 0; i < mptr->rows; i++) {
        for(int j = 0; j < mptr->columns; j++) {
            result->data[j * mptr->rows + i] = mptr->data[i * mptr->columns + j];
        }
    }
    return result;
}

Matrix* hadamard_product( Matrix* mptr1,  Matrix* mptr2) {

    if(mptr1->rows == mptr2->rows && mptr2->columns == mptr2->columns) {

        Matrix* result = ( Matrix*)malloc(sizeof( Matrix));
        result->rows = mptr1->rows;
        result->columns = mptr1->columns;
        result->data = (double*)malloc(sizeof(double) * result->rows * result->columns);
        for(int i = 0; i < result->rows * result->columns; i++) {
            result->data[i] = mptr1->data[i] * mptr2->data[i];
        }
        return result;
    }
}

Matrix* clip( Matrix* mptr, double lower, double upper) {

    Matrix* result = ( Matrix*)malloc(sizeof( Matrix));
    result->rows = mptr->rows;
    result->columns = mptr->columns;
    result->data = (double*)malloc(sizeof(double) * result->rows * result->columns);
    for(int i = 0; i < result->rows * result->columns; i++) {

        if(mptr->data[i] < lower) {
            mptr->data[i] = lower;
        }
        else if(mptr->data[i] > upper) {
            mptr->data[i] = upper;
        }

        result->data[i] = mptr->data[i];

    }

    return result;

}

void printMatrix(Matrix* mptr) {

    printf("[");

    for(int i = 0; i < mptr->rows * mptr->columns; i++) {

        
        if(i != mptr->rows * mptr->columns - 1) {
            printf("%f ", mptr->data[i]);
        }

        else {
            printf("%f", mptr->data[i]);
        }

        if((i + 1) % mptr->columns == 0 && i != mptr->rows * mptr->columns - 1) {
            printf("\n");

        }
    }

    printf("]\n");
}

/* Functions */

Matrix* relu(Matrix* mptr) {
     Matrix* result = (Matrix*)malloc(sizeof(Matrix));
    result->rows = mptr->rows;
    result->columns = mptr->columns;
    result->data = (double*)malloc(sizeof(double) * result->rows * result->columns);
    for(int i = 0; i < result->rows * result->columns; i++) {
        if(mptr->data[i] > 0) {
            result->data[i] = mptr->data[i];
        }
        else {
            result->data[i] = 0;
        }
    }
    return result;
}

Matrix* derivative_relu( Matrix* mptr) {
     Matrix* result = ( Matrix*)malloc(sizeof( Matrix));
    result->rows = mptr->rows;
    result->columns = mptr->columns;
    result->data = (double*)malloc(sizeof(double) * result->rows * result->columns);
    for(int i = 0; i < result->rows * result->columns; i++) {
        if(mptr->data[i] > 0) {
            result->data[i] = 1;
        }
        else {
            result->data[i] = 0;
        }
    }
    return result;
}

Matrix* sigmoid(Matrix* mptr) {
    Matrix* result = (Matrix*)malloc(sizeof(Matrix));
    result->rows = mptr->rows;
    result->columns = mptr->columns;
    result->data = (double*)malloc(sizeof(double) * result->rows * result->columns);
    for(int i = 0; i < result->rows * result->columns; i++) {
        result->data[i] = 1 / (1 + exp(- mptr->data[i]));
    }
    return result;
}

Matrix* derivative_sigmoid( Matrix* mptr) {
     Matrix* result = ( Matrix*)malloc(sizeof( Matrix));
    result->rows = mptr->rows;
    result->columns = mptr->columns;
    result->data = (double*)malloc(sizeof(double) * result->rows * result->columns);
    for(int i = 0; i < result->rows * result->columns; i++) {
        result->data[i] = (1 / (1 + exp(- mptr->data[i]))) * (1 - 1 / (1 + exp(- mptr->data[i])));
    }
    return result;
}

Matrix* natural_log(Matrix* mptr) {
    Matrix* result = (Matrix*)malloc(sizeof(Matrix));
    result->rows = mptr->rows;
    result->columns = mptr->columns;
    result->data = (double*)malloc(sizeof(double) * result->rows * result->columns);
    for(int i = 0; i < result->rows * result->columns; i++) {
        result->data[i] = log(mptr->data[i]);
    }
    return result;
}

/* Loss functions */

double BCE_Loss( Matrix* y_true,  Matrix* y_pred) {
    double result = 0;
    y_pred = clip(y_pred, 1e-7, 1.0 - 1e-7);
    if(y_true->rows == y_pred->rows && y_true->columns == y_pred->columns) {
        for(int i = 0; i < y_true->rows * y_true->columns; i++) {
            result += -1.0 * (y_true->data[i] * log(y_pred->data[i]) + (1.0 - y_true->data[i]) * log(1.0 - y_pred->data[i]));
        }
        return result / (y_true->rows * y_true->columns); 
    }

}

double MSE_Loss( Matrix* y_pred,  Matrix* y_true) {
    double result = 0;
    if(y_true->rows == y_pred->rows && y_true->columns == y_pred->columns) {
        for(int i = 0; i < y_true->rows * y_true->columns; i++) {
            result += 0.5 * pow(y_pred->data[i] - y_true->data[i], 2);
        }
        return result / (y_true->rows * y_true->columns); 
    }

}

Matrix* MSE_Loss_matrix( Matrix* y_pred,  Matrix* y_true) {
    Matrix* result = ( Matrix*)malloc(sizeof( Matrix));
    result->rows = y_pred->rows;
    result->columns = y_pred->columns;
    result->data = (double*)malloc(sizeof(double) * result->rows * result->columns);
    for(int i = 0; i < result->rows * result->columns; i++) {
        result->data[i] = pow(y_pred->data[i] - y_true->data[i], 2);
    }
    return result;
}

Matrix* derivative_MSE_Loss( Matrix* y_pred,  Matrix* y_true) {
    if(y_true->rows == y_pred->rows && y_true->columns == y_pred->columns) {
         Matrix* result = ( Matrix*)malloc(sizeof( Matrix));
        result->rows = y_true->rows;
        result->columns = y_true->columns;
        result->data = (double*)malloc(sizeof(double) * result->rows * result->columns);
        for(int i = 0; i < y_true->rows * y_true->columns; i++) {
            result->data[i] = y_pred->data[i] - y_true->data[i];
        }
        return result;
    }

}

/* Layer initializer */

Layer* initLayer(int n, int m) {
    Layer* result = (Layer*)malloc(sizeof(Layer));
    result->weights = initMatrix(n, m);
    return result;
}

/* SGD optimizer */

Matrix* update_weights_SGD( Matrix* weights,  Matrix* dweights, double learning_rate) {
     Matrix* result = ( Matrix*)malloc(sizeof( Matrix));
    result->rows = weights->rows;
    result->columns = weights->columns;
    result->data = (double*)malloc(sizeof(double) * result->rows * result->columns);
    for(int i = 0; i < result->rows * result->columns; i++) {
        result->data[i] = weights->data[i] - learning_rate * dweights->data[i];
    }
    return result;
}