/*
 Mikhail Chliakhovski
 9630117
 COMP 428
 Assignment 3

 Optimal matrix-parenthesization (parallel solution 1)
 */

#include "mpi.h"
#include <stdlib.h>
#include <iostream>
#include <limits>
#include <algorithm>
#include <sys/time.h>
#include <math.h>

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

	int P, // number of tasks
		pid; // rank

	double start, end; // time stamps

	MPI_Init(&argc, &argv);

    // get number of tasks and current id
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

	int r [] = {3, 4, 5, 2, 3, 4}; // size N + 1 array of matrix dimensions

	// create cost matrix
	int C [N+1][N+1];

	// initialize matrix to all zeros
	fillWithZeros(C);

	int i, j, k, n, s;

	// record start time
	if (pid == 0) start = getClock();

	// apply the cost function one diagonal at a time
	for (k = 1; k <= N; k++) {
		if (N-k+1 > P) { // if number of tasks exceeds diagonal length
			s = floor(N-k+1/P); // segment size is diagonal length divided by number of tasks
			i = pid * s + 1; // start of segment
			j = i + k - 1;
			if (pid != P-1) // if not last task
				n = pid * s + s; // end of segment
			else
				n = N - k + 1; // end of diagonal
			while (i <= n) {
				cost(r, C, i, j);
				i++;
				j++;
			}
		} else {
			if (pid < N-k+1) { // if there is work
				cost(r, C, pid+1, pid+k);
				int sendbuf [N-k+1], recbuf [N-k+1];
				sendbuf[pid] = C[pid+1][pid+k];
				MPI_Alltoall(sendbuf, 1, MPI_INT, recbuf, 1, MPI_INT, MPI_COMM_WORLD);
				for(i = 1, j = k; j <= N; i++, j++)
					C[i][j] = recbuf[i-1];
			} else
				break;
		}
	}

    if (pid == 0) {
	    // record end time
	    end = getClock();

	    printMatrix(C);

	    // output total time
	    cout << "Total time: " << end-start << " ms" << endl;
	}

    MPI_Finalize();

	return 0;
    
}
