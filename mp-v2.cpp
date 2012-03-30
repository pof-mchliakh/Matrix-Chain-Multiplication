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
#define SIZE (N*(N+1))/2 // number of non-empty elements in matrix

using namespace std;

// clock function
double getClock() {
    timeval tp;
    gettimeofday(&tp, NULL);
    return (tp.tv_sec + (tp.tv_usec / 1000000.0)) * 1000; // return ms
}

// function to map matrix coordinates to flattened array index
int map(int i, int j) {
	return ((N-(j-i)-1)*(N-(j-i))/2) + i;
}

// minimum cost function for matrix chain multiplication
void optimalCost(int *r, int *C, int i, int j) {

	int k, result = numeric_limits<int>::max(); // largest possible integer value for running minimum

	for (k = i; k < j; k++)
		result = min(C[map(i,k)] + C[map(k+1,j)] + r[i]*r[k+1]*r[j+1], result);

	C[map(i,j)] = result;
}

// function to print flattened array as matrix
void printAsMatrix(int *arr) {

	int i, j, k;

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			k = map(i, j);
			if (k < SIZE)
				cout << arr[k] << " ";
			else
				cout << 0 << " ";
		}
		cout << endl;
	}

}

// MAIN
int main(int argc, char *argv[]) {

	double start, end; // time stamps

	int r [] = {3, 4, 5, 2, 3, 4}; // size N + 1 array of matrix dimensions

	// create flattened cost matrix and intialize to all zeros
	int C [SIZE] = {0};

	int i, j, k;

	// record start time
	start = getClock();

	// apply the cost function one diagonal at a time
	for (k = 1; k < N; k++) { // start at second diagonal
		i = 0;
		j = k;
		while (j < N) {
			optimalCost(r, C, i, j);
			i++;
			j++;
		}
	}

	// record end time
	end = getClock();

	printAsMatrix(C);

	// output total time
	cout << "Total time: " << end-start << " ms" << endl;

	return 0;
    
}
