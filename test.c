#include<stdio.h>
#include<string.h>
#include<stdbool.h>

#include "neural.h"

#define MAXCHAR 1000

void read_csv(char* path) {

    int n = 284806;
    int m = 31;

    Matrix* result = (Matrix*)malloc(sizeof(Matrix));
    result->rows = n;
    result->columns = m;
    result->data = (double*)malloc(sizeof(double) * result->rows * result->columns);

    for(int i = 0; i < result->rows; i++) {
        for(int j = 0; j < result->columns; j++) {

        }
    }

    FILE *fp;
    char row[MAXCHAR];
    char* token;

    fp = fopen(path,"r");

    while (feof(fp) != true)
    {

        if(n == 0) {
            n += 1;
            fgets(row, MAXCHAR, fp);
            continue;
        }

        fgets(row, MAXCHAR, fp);

        token = strtok(row, ",");

        while(token != NULL && n == 1)
        {
            token = strtok(NULL, ",");
            m += 1;
        }

        n += 1;

    }

    printf("(%d, %d)", n, m);

}

int main(){

    FILE *fp;
    char row[MAXCHAR];
    char* token;

    fp = fopen("creditcard.csv","r");

    int brojac = 0;

    while (brojac < 10)
    {
        fgets(row, MAXCHAR, fp);
        printf("Row: %s", row);

        token = strtok(row, ",");

        while(token != NULL)
        {
            printf("Token: %s\n", token);
            token = strtok(NULL, ",");
        }
        brojac += 1;
    }

    read_csv("creditcard.csv");
    

    return 0;
}
