#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

struct Matrix
{
    int rows, columns;
    double* data;
};

double rand_gen() {
   return ( (double)(rand()) + 1. )/( (double)(RAND_MAX) + 1. );
}

double normalRandom() {
   double v1=rand_gen();
   double v2=rand_gen();
   return cos(2*3.14*v2)*sqrt(-2.*log(v1));
}

struct Matrix* initMatrix(int n, int m) {

    struct Matrix* mat = (struct Matrix*)malloc(sizeof(struct Matrix));
    mat->rows = n;
    mat->columns = m;
    mat->data = (double*)malloc(sizeof(double) * n * m);
    for(int i = 0; i < n * m; i++) {
        mat->data[i] = normalRandom();
    }
    return mat;

}

void printMatrix(struct Matrix* mptr) {

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

struct Matrix* dot(struct Matrix* mptr1, struct Matrix* mptr2) {
    if (mptr1->columns == mptr2->rows) {
        struct Matrix* result = (struct Matrix*)malloc(sizeof(struct Matrix));
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

struct Matrix* transpose(struct Matrix* mptr) {
    struct Matrix* result = (struct Matrix*)malloc(sizeof(struct Matrix));
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

struct Matrix* relu(struct Matrix* mptr) {
    struct Matrix* result = (struct Matrix*)malloc(sizeof(struct Matrix));
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

struct Matrix* sigmoid(struct Matrix* mptr) {
    struct Matrix* result = (struct Matrix*)malloc(sizeof(struct Matrix));
    result->rows = mptr->rows;
    result->columns = mptr->columns;
    result->data = (double*)malloc(sizeof(double) * result->rows * result->columns);
    for(int i = 0; i < result->rows * result->columns; i++) {
        result->data[i] = 1 / (1 + exp(- mptr->data[i]));
    }
    return result;
}

struct Matrix* natural_log(struct Matrix* mptr) {
    struct Matrix* result = (struct Matrix*)malloc(sizeof(struct Matrix));
    result->rows = mptr->rows;
    result->columns = mptr->columns;
    result->data = (double*)malloc(sizeof(double) * result->rows * result->columns);
    for(int i = 0; i < result->rows * result->columns; i++) {
        result->data[i] = log(mptr->data[i]);
    }
    return result;
}

struct Matrix* hadamard_product(struct Matrix* mptr1, struct Matrix* mptr2) {

    if(mptr1->rows == mptr2->rows && mptr2->columns == mptr2->columns) {

        struct Matrix* result = (struct Matrix*)malloc(sizeof(struct Matrix));
        result->rows = mptr1->rows;
        result->columns = mptr1->columns;
        result->data = (double*)malloc(sizeof(double) * result->rows * result->columns);
        for(int i = 0; i < result->rows * result->columns; i++) {
            result->data[i] = mptr1->data[i] * mptr2->data[i];
        }
        return result;
    }
}

struct Matrix* clip(struct Matrix* mptr, double lower, double upper) {

    struct Matrix* result = (struct Matrix*)malloc(sizeof(struct Matrix));
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

double BCE_Loss(struct Matrix* y_true, struct Matrix* y_pred) {
    double result = 0;
    y_pred = clip(y_pred, 1e-7, 1.0 - 1e-7);
    if(y_true->rows == y_pred->rows && y_true->columns == y_pred->columns) {
        for(int i = 0; i < y_true->rows * y_true->columns; i++) {
            printf("Prvi i drugi log: %f %f\n", log(y_pred->data[i]), log(1.0 - y_pred->data[i]));
            result += -1.0 * (y_true->data[i] * log(y_pred->data[i]) + (1.0 - y_true->data[i]) * log(1.0 - y_pred->data[i]));
        }
        return result / (y_true->rows * y_true->columns); 
    }

}
