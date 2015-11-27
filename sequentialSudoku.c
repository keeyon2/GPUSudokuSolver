#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void extractNumbers(char* fileName, int* board);
void printGrid(int* board);
int** createArray(int m, int n);
void printGrid(int** board);
void extractNumbers(char* fileName, int** board);
void startSeq(char* name);

int main ( int argc, char *argv[] )
{
    // Read in file
    if (argc < 2) {
        printf("Sorry, we need a command line argument with the sudoku puzzle to solve");
        exit(0);
    }

    else {
        char* name = argv[1];
		startSeq(name);
    }
}

//void backTracking(int** grid)

void startSeq(char* name) {
    int** grid = createArray(9, 9);
    extractNumbers(name, grid);
	int** grid2 = createArray(9, 9); 
    printGrid(grid);
    //printGrid(grid2);
	int array[9][9];
	array[0][0] = 4;
    extractNumbers2(name, &array[0][0]);
    printGrid2(&array[0][0]);
    int array2[9][9];
    memcpy(array2, array, sizeof(int) * 81);
    printGrid2(&array2[0][0]);
    array2[0][0] = 15;
    printGrid2(&array2[0][0]);
    printGrid2(&array[0][0]);
}

void extractNumbers2(char* fileName, int* grid) {
    FILE *input;
    input = fopen(fileName, "r");
    char inp;
    for (int row = 0; row < 9; row++) {
        for (int column = 0; column < 9; column++) {
            fscanf(input," %c", &inp);
            int number = inp - '0';
            grid[row * 9 + column] = number;
        }
    }

    fclose(input);
}


void extractNumbers(char* fileName, int** board) {
    FILE *input;
    input = fopen(fileName, "r");
    char inp;
    for (int row = 0; row < 9; row++) {
        for (int column = 0; column < 9; column++) {
            fscanf(input," %c", &inp);
            int number = inp - '0';
            board[row][column] = number;
        }
    }

    fclose(input);
}

void printGrid2(int* board) {
    printf("\n");
    for (int row = 0; row < 9; row++) {
        for (int column = 0; column < 9; column++) {
            printf("%d ", board[row * 9 + column]);
        }
        printf("\n");
    }
	printf("\n");
}

void printGrid(int** board) {
    printf("\n");
    for (int row = 0; row < 9; row++) {
        for (int column = 0; column < 9; column++) {
            printf("%d ", board[row][column]);
        }
        printf("\n");
    }
	printf("\n");
}

int** createArray(int m, int n)
{
    int* values = calloc(m*n, sizeof(int));
    int** rows = malloc(n*sizeof(int*));
    for (int i=0; i<n; ++i)
    {
        rows[i] = values + i*m;
    }
    return rows;
}
