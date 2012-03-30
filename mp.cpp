/*
 Mikhail Chliakhovski
 9630117
 COMP 428
 Assignment 3

 Optimal matrix-parenthesization (sequential)
 */

#include <stdlib.h>
#include <iostream>
#include <limits>
#include <algorithm>
#include <sys/time.h>

#define N 5 // size of matrix chain

using namespace std;

// clock function
double getClock() {
    timeval tp;
    gettimeofday(&tp, NULL);
    return (tp.tv_sec + (tp.tv_usec / 1000000.0)) * 1000; // return ms
}

// minimum cost function for matrix chain multiplication
void cost(int *r, int C[N+1][N+1], int i, int j) {

	if (i == j) {
		C[i][j] = 0;
		return;
	}

	int k, result = numeric_limits<int>::max(); // largest possible integer value for running minimum

	for(k = i; k < j; k++)
		result = min(C[i][k] + C[k+1][j] + r[i-1]*r[k]*r[j], result);

	C[i][j] = result;
}

void fillWithZeros(int matrix[N+1][N+1]) {

	int i, j;

	for(i = 0; i < N+1; i++)
		for(j = 0; j < N+1; j++)
			matrix[i][j] = 0;

}

void printMatrix(int matrix[N+1][N+1]) {

	int i, j;

	for(i = 0; i < N+1; i++) {
		for(j = 0; j < N+1; j++)
			cout << matrix[i][j] << " ";
		cout << endl;
	}

}

// MAIN
int main(int argc, char *argv[]) {

	double start, end; // time stamps

	int r [] = {3, 4, 5, 2, 3, 4}; // size N + 1 array of matrix dimensions

	// create cost matrix
	int C [N+1][N+1];

	// initialize matrix to all zeros
	fillWithZeros(C);

	int i, j, k;

	// record start time
	start = getClock();

	// apply the cost function one diagonal at a time
	for (k = 1; k <= N; k++) {
		i = 1;
		j = k;
		while (j <= N) {
			cost(r, C, i, j);
			i++;
			j++;
		}
	}

	// record end time
	end = getClock();

	printMatrix(C);

	// output total time
    cout << "Total time: " << end-start << " ms" << endl;

	return 0;
    
}
