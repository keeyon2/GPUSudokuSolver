#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int ROWS = 9;
int COLUMNS = 9;
char* FILENAME;

typedef enum { false, true} bool;

void extractNumbers(char* fileName, int* board);
void printGrid(int* board);
void printGridToFile(int* board);
void startSeq(char* name);
bool sudokuSolved(int board[ROWS][COLUMNS]);
bool backTracking(int grid[ROWS][COLUMNS]);
char* concat(char *s1, char *s2);
bool numberPlacementValid(int numberToCheck, int checkingRow, int checkingColumn, 
        int board[ROWS][COLUMNS]);

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

bool findUnassignedLocation(int grid[ROWS][COLUMNS], int *row, int *col) {
    int localRow = *row;
    int localCol = *col;

    for (localRow = 0; localRow < ROWS; localRow++) {
        for (localCol = 0; localCol < COLUMNS; localCol++) {
            if (grid[localRow][localCol] == 0) {
                *row = localRow;
                *col = localCol;
                return true;
            }
        }
    }
    return false;
}

bool backTracking(int grid[ROWS][COLUMNS]) {
    int row;
    int column;

    if (!findUnassignedLocation(grid, &row, &column)) {
        return true;
    }

    for (int insertNumber = 1; insertNumber <= ROWS; insertNumber++) {
        bool isValid = numberPlacementValid(insertNumber, row, column, grid);
        if (isValid) {
            grid[row][column] = insertNumber;

            if (backTracking(grid)) {
                return true;
            }

            grid[row][column] = 0;
        }
    }
    return false;
}

bool sudokuSolved(int board[ROWS][COLUMNS]) {
    int copyGrid[ROWS][COLUMNS];
    memcpy(copyGrid, board, sizeof(int) * ROWS * COLUMNS);
    for (int row = 0; row < ROWS; row++) {
        for (int column = 0; column < COLUMNS; column++) {
            if (copyGrid[row][column] == 0) {
                return false;
            }
        }
    }
    //printGrid(&board[0][0]);
    return true;
}

bool numberPlacementValid(int numberToCheck, int checkingRow, int checkingColumn, 
        int board[ROWS][COLUMNS]) {

    int copyGridForChecking[ROWS][COLUMNS];
    memcpy(copyGridForChecking, board, sizeof(int) * ROWS * COLUMNS);

    // Check if number to check exists in Column
    int boardValue = 0;
    for (int row = 0; row < ROWS; row++) {
        boardValue = copyGridForChecking[row][checkingColumn];
        if (boardValue == numberToCheck) {
            return false;
        }
    }

    // Check if number to check exists in Row 
    for (int column = 0; column < COLUMNS; column++) {
        boardValue = copyGridForChecking[checkingRow][column];
        if (boardValue == numberToCheck) {
            return false;
        }
    }

    // Check if exists in 3 x 3 grid
    int rowGrid = checkingRow / 3;
    int columnGrid = checkingColumn / 3;
    for (int rowAdd = 0; rowAdd < 3; rowAdd++) {
        for (int colAdd = 0; colAdd < 3; colAdd++) {
            int rowValue = (rowGrid * 3) + rowAdd;
            int colValue = (columnGrid * 3) + colAdd;
            boardValue = copyGridForChecking[rowValue][colValue];
            if (boardValue == numberToCheck) {
                return false;
            }
        }
    }

    return true;
}

void startSeq(char* name) {
	int originalGrid[ROWS][COLUMNS];
    FILENAME = name;
    extractNumbers(name, &originalGrid[0][0]);

    bool backtracked = backTracking(originalGrid);

    if (backtracked) {
        printf("We found solution\n");
        //printGrid(&originalGrid[0][0]);
        printGridToFile(&originalGrid[0][0]);
    }

    else {
        printf("Sorry, but we found no solution to to this sudoku puzzle\n");
    }
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

void printGrid(int* board) {
    int copyGrid[ROWS][COLUMNS];
    memcpy(copyGrid, board, sizeof(int) * ROWS * COLUMNS);

    printf("\n");
    for (int row = 0; row < ROWS; row++) {
        for (int column = 0; column < COLUMNS; column++) {
            //printf("%d ", board[row * COLUMNS + column]);
            printf("%d ", copyGrid[row][column]);
        }
        printf("\n");
    }
	printf("\n");
}

void printGridToFile(int* board) {
    char* extension = ".sol";
    char* outputFileName = concat(FILENAME, extension); 
    printf("Output name is %s\n", outputFileName);

    FILE *ofp;
    ofp = fopen(outputFileName, "w");

    int copyGrid[ROWS][COLUMNS];
    memcpy(copyGrid, board, sizeof(int) * ROWS * COLUMNS);

    printf("\n");
    for (int row = 0; row < ROWS; row++) {
        for (int column = 0; column < COLUMNS; column++) {
            //printf("%d ", board[row * COLUMNS + column]);
            fprintf(ofp, "%d ", copyGrid[row][column]);
        }
        fprintf(ofp, "\n");
    }
    fclose(ofp);
}

char* concat(char *s1, char *s2) {
    char *result = malloc(strlen(s1)+strlen(s2)+1);//+1 for the zero-terminator
    //in real code you would check for errors in malloc here
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}
