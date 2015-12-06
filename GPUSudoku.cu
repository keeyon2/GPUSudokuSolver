#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <curand.h>
#include <curand_kernel.h>

#define MAX 9
#define MAX_RAND_TRIES 100

int ROWS = 9;
int COLUMNS = 9;
char* FILENAME;

//char* concat(char *s1, char *s2);
void printGridToFile(int* board);
void startSeq(char* name);

int main ( int argc, char *argv[] ) {
    // Read in file
    if (argc < 2) {
        printf("Sorry, we need a command line argument with the sudoku puzzle to solve");
        exit(0);
    }

    else {
        char* name; 
        name = strtok(argv[1], ".");
        FILENAME = name;

		startSeq(FILENAME);
    }
}

void printGrid(int* board) {
    int copyGrid[ROWS][COLUMNS];
    memcpy(copyGrid, board, sizeof(int) * ROWS * COLUMNS);

    printf("\n");
    for (int row = 0; row < ROWS; row++) {
        for (int column = 0; column < COLUMNS; column++) {
            printf("%d ", copyGrid[row][column]);
        }
        printf("\n");
    }
	printf("\n");
}

void extractNumbers(char* fileName, int* grid) {
    FILE *input;
    input = fopen(fileName, "r");
    char inp;
    for (int row = 0; row < ROWS; row++) {
        for (int column = 0; column < COLUMNS; column++) {
            fscanf(input," %c", &inp);
            int number = inp - '0';
            grid[row * COLUMNS + column] = number;
        }
    }

    fclose(input);
}

__device__ void d_swap (int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

__device__ void d_randomize(int nineArray[], curandState_t state) {
	int tryValid = curand(&state) % MAX;
    for (int i = 8; i > 0; i--) {
        int j = curand(&state) % (i+1);
        d_swap(&nineArray[i], &nineArray[j]);
    }
}

__device__ int d_numberPlacementValid(int numberToCheck, int checkingRow, int checkingColumn, 
        int board[MAX][MAX]) {

    // Check if number to check exists in Column
    int boardValue = 0;
    for (int row = 0; row < MAX; row++) {
        boardValue = board[row][checkingColumn];
        if (boardValue == numberToCheck) {
            return 0;
        }
    }

    // Check if number to check exists in Row 
    for (int column = 0; column < MAX; column++) {
        boardValue = board[checkingRow][column];
        if (boardValue == numberToCheck) {
            return 0;
        }
    }

    // Check if exists in 3 x 3 grid
    int rowGrid = checkingRow / 3;
    int columnGrid = checkingColumn / 3;
    for (int rowAdd = 0; rowAdd < 3; rowAdd++) {
        for (int colAdd = 0; colAdd < 3; colAdd++) {
            int rowValue = (rowGrid * 3) + rowAdd;
            int colValue = (columnGrid * 3) + colAdd;
            boardValue = board[rowValue][colValue];
            if (boardValue == numberToCheck) {
                return 0;
            }
        }
    }

    return 1;
}

__global__ void replaceZeros(int* d_sudoku, int* d_sudoku_solution, int timeCalled) {

    __shared__ int shared_sudoku[9][9];
    
	int thread_x = threadIdx.x;
	int thread_y = threadIdx.y;

	int blockId = blockIdx.x + blockIdx.y *gridDim.x;

	int threadId = blockId * (blockDim.x * blockDim.y) +
		(threadIdx.y * blockDim.x) + threadIdx.x;

	shared_sudoku[thread_x][thread_y] = d_sudoku[thread_x+ 9*thread_y];

    // Synch threads to synch shared data
    __syncthreads();


	curandState_t state;
	curand_init(threadId, gridDim.y/2, timeCalled, &state);

    // Create thread individual sudoku board
    int local_sudoku[9][9];
    for (int row = 0; row < 9; row++) {
        for (int col = 0; col < 9; col++) {
            local_sudoku[row][col] = shared_sudoku[row][col];
        }
    }

    // For each element, try to random a value that is valid
    for (int row = 0; row < 9; row++) {
        for (int col = 0; col < 9; col++) {
            if (local_sudoku[row][col] == 0) {
				int insertNum = 0;
                int nineArray[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
                d_randomize(nineArray, state);
                for (int i = 0; i < 9; i++) {
					if (d_numberPlacementValid(nineArray[i], row, col, local_sudoku)) {
						insertNum = nineArray[i];
						break;
					}
                }

				if (insertNum == 0) {
                    return;
				}

                else {
                    local_sudoku[row][col] = insertNum;
                }
            }
        }
    }

    // Only get here with solved sudoku puzzle
    // Placing values in solution
    for (int row = 0; row < 9; row++) {
        for (int col = 0; col < 9; col++) {
            d_sudoku_solution[row * 9 + col] = local_sudoku[row][col];
        }
    }
}

void printGridToFile(int* board) {
    char* extension = ".sol";
    int stringSize = strlen(FILENAME) + 3;
    char outputFileName[stringSize]; 
    strcpy(outputFileName, FILENAME);
    strcpy(outputFileName, extension);


    //char* outputFileName = concat(FILENAME, extension); 
    //printf("Output name is %s", outputFileName);

    FILE *ofp;
    ofp = fopen(outputFileName, "w");

    int copyGrid[ROWS][COLUMNS];
    memcpy(copyGrid, board, sizeof(int) * ROWS * COLUMNS);

    //printf("\n");
    for (int row = 0; row < ROWS; row++) {
        for (int column = 0; column < COLUMNS; column++) {
            //printf("%d ", board[row * COLUMNS + column]);
            fprintf(ofp, "%d ", copyGrid[row][column]);
        }
        fprintf(ofp, "\n");
    }
    fclose(ofp);
}

void startSeq(char* name) {
	int originalGrid[ROWS][COLUMNS];
    FILENAME = name;
    extractNumbers(name, &originalGrid[0][0]);
    //printGrid(&originalGrid[0][0]);

    int sudokuSize = sizeof(int) * 81;

    int *d_sudoku;
    int *sudoku;
    int *d_sudoku_solution;
    int *sudoku_solution;

    cudaHostAlloc((void**)&sudoku, sudokuSize, cudaHostAllocDefault);
    cudaHostAlloc((void**)&sudoku_solution, sudokuSize, cudaHostAllocDefault);
    for (int row = 0; row < ROWS; row++) {
        for (int col = 0; col < COLUMNS; col++) {
            sudoku[row * 9 + col] = originalGrid[row][col];
            sudoku_solution[row * 9 + col] = 0;
        }
    }

    dim3 dimGrid(12, 15);
    dim3 dimBlock(9, 9);

    cudaMalloc((void**)&d_sudoku, sudokuSize);
    cudaMalloc((void**)&d_sudoku_solution, sudokuSize);
    cudaMemcpy(d_sudoku, sudoku, sudokuSize, cudaMemcpyHostToDevice);

    replaceZeros<<<dimGrid,dimBlock>>>(d_sudoku, d_sudoku_solution, 0); 

    cudaMemcpy(sudoku_solution, d_sudoku_solution, sudokuSize, cudaMemcpyDeviceToHost);
    cudaFree(d_sudoku);
    cudaFree(d_sudoku_solution);

    //printf("We found solution");
    //printGrid(sudoku_solution);
    printGridToFile(sudoku_solution);
}
