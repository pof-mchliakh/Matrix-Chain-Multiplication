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

#define N 51 // size of matrix chain
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

	int P, // number of tasks
		pid; // rank

	double start, end; // time stamps

	MPI_Init(&argc, &argv);

    // get number of tasks and current id
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

	int r [] = {30, 44, 54, 29, 3, 42, 48, 53, 96, 31, 30, 44, 54, 29, 3, 42, 48, 53, 96, 31, 30, 44, 54, 29, 3, 42, 48, 53, 96, 31, 30, 44, 54, 29, 3, 42, 48, 53, 96, 31, 30, 44, 54, 29, 3, 42, 48, 53, 96, 31, 99}; // size N + 1 array of matrix dimensions

	// create flattened cost matrix and intialize to all zeros
	int C [SIZE] = {0};

	int i, j, k, m, n, s;

	// record start time
	if (pid == 0) start = getClock();

	// apply the cost function one diagonal at a time
	for (k = 1; k < N; k++) { // start at second diagonal
		if (N-k > P) { // if diagonal length exceeds number of tasks
			s = floor((N-k)/P); // segment size is diagonal length divided by number of tasks
			i = pid * s; // start of segment
			j = i + k;
			if (pid != P-1) // if not last task
				n = pid * s + s; // end of segment
			else
				n = N - k; // end of diagonal
			while (i < n) {
				optimalCost(r, C, i, j);
				i++;
				j++;
			}
			i = pid * s;
			j = i + k;
			MPI_Allgather(C+map(i,j), s, MPI_INT, C+map(0,k), s, MPI_INT, MPI_COMM_WORLD);
			m = (N-k) % P; // remainder
			if (m != 0) {
				// last task broadcasts remaining cells
				MPI_Bcast(C + map(P*s, k + P*s), m, MPI_INT, P-1, MPI_COMM_WORLD);
			}
		} else { // there are at least as many tasks as cells in diagonal
			int sendbuf, recbuf [P];
			if (pid < N-k) { // if there is work
				optimalCost(r, C, pid, pid+k);
				sendbuf = C[map(pid,pid+k)];
				MPI_Allgather(&sendbuf, 1, MPI_INT, recbuf, 1, MPI_INT, MPI_COMM_WORLD);
				for(i = 0, j = k; j < N; i++, j++)
					C[map(i,j)] = recbuf[i];
			} else // no need to fill send buffer
				MPI_Allgather(&sendbuf, 1, MPI_INT, recbuf, 1, MPI_INT, MPI_COMM_WORLD);
		}
	}

	// wait until all tasks complete
    MPI_Barrier(MPI_COMM_WORLD);

    if (pid == 0) {
	    // record end time
	    end = getClock();

	    printAsMatrix(C);

	    // output total time
	    cout << "Total time: " << end-start << " ms" << endl;
	}

    MPI_Finalize();

	return 0;
    
}
