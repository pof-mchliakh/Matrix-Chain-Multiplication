/*
 Mikhail Chliakhovski
 9630117
 COMP 428
 Assignment 3

 Optimal matrix-parenthesization (parallel solution 2)
 */

#include "mpi.h"
#include <stdlib.h>
#include <iostream>
#include <limits>
#include <algorithm>
#include <sys/time.h>
#include <math.h>

#define N 10 // size of matrix chain
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
void optimalCost(int P, int pid, int *r, int *C, int i, int j, int *computed, int n, int *minima, MPI_Request *requests, int lowest) {

	if (computed[n] < j-i) { // if there are costs to compute

		int k, l, flag1, flag2;

		for (k = i + computed[n]; k < j; k++) {

			if (pid != 0) {

				// check if both values have been received
				if (i == k) // if value from main diagonal
					flag1 = 1;
				else if (i < N-lowest) { // if value belongs to this task
					if (C[map(i,k)] != numeric_limits<int>::max()) // if not missing
						flag1 = 1;
					else
						flag1 = 0;
				} else // value must belong to another task
					MPI_Test(&requests[map(i-N+lowest,k)], &flag1, MPI_STATUS_IGNORE);	

				if (k+1 == j)
					flag2 = 1;
				else if (k+1 < N-lowest) {
					if (C[map(k+1,j)] != numeric_limits<int>::max())
						flag2 = 1;
					else
						flag2 = 0;
				} else
					MPI_Test(&requests[map(k+1-N+lowest,j)], &flag2, MPI_STATUS_IGNORE);	

			} else
				flag1 = flag2 = 1; // lowest task need not check

			if (flag1 && flag2) {
				C[map(i,j)] = min(C[map(i,k)] + C[map(k+1,j)] + r[i]*r[k+1]*r[j+1], C[map(i,j)]);
				computed[n]++;
				(*minima)--;
			} else
				break;

		}

		// if cost is fully computed, send to all tasks above current
		if (k == j) {
			if (pid != P-1) { // the higest task does not send
				MPI_Request request;
				for (l = pid+1; l < P; l++) {
					MPI_Isend(&C[map(i,j)], 1, MPI_INT, l, map(i,j), MPI_COMM_WORLD, &request);
				}
			}
		}
	}
}

// function for partitioning the matrix (by rows) between tasks
void partition(int P, int pid, int *lowest, int *highest, int *elements, int *below) {

	int average, row;
	int buffer [2];
	MPI_Status status;

	average = floor(((N-1)*N)/(2*P));

	if (pid == 0) {
		*lowest = row = 1;
		*below = 0;
	} else {
		MPI_Recv(buffer, 2, MPI_INT, pid-1, 0, MPI_COMM_WORLD, &status);
		*lowest = row = buffer[0] + 1;
		*below = buffer[1];
	}

	if (pid != P-1) {
		while (((row*(row+1))/2)-*below < average)
			row++;
		if ((row*(row+1))/2-*below-average < abs(((row-1)*row)/2-*below-average))
			buffer[1] = (row*(row+1)) / 2; // total number of elements below this task's region

		else {
			buffer[1] = ((row-1)*row) / 2;
			row--;
		}
		*elements = buffer[1] - *below;
	} else {
		row = N - 1;
		*elements = ((N-1)*N) / 2 - *below;
	}

	*highest = buffer[0] = row;

	if (pid != P-1)
		MPI_Send(buffer, 2, MPI_INT, pid+1, 0, MPI_COMM_WORLD);

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
		pid, // rank
		lowest, // lowest row belonging to this task
		highest, // highest row belonging to this task
		elements, // total number of elements belonging to this task
		below, // total number of elements below this task's region
		minima; // total number of minima to compute

	double start, end; // time stamps

	MPI_Request request;

	MPI_Init(&argc, &argv);

    // get number of tasks and current id
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

	int r [] = {3, 4, 5, 2, 3, 4, 2, 5, 7, 3, 4}; // size N + 1 array of matrix dimensions

	// create flattened cost matrix
	int C [SIZE];

	int i, j, k, n;

	// initialize to largest integer for missing costs
	for (i = 0; i < SIZE-N; i++)
		C[i] = numeric_limits<int>::max();
	// and 0's for the main diagonal
	for (i = SIZE-N; i < SIZE; i++)
		C[i] = 0;

	// record start time
	if (pid == P-1) start = getClock();

	// compute start and end row for this task
	partition(P, pid, &lowest, &highest, &elements, &below);

	MPI_Request *requests = new MPI_Request [lowest + below];

	// all but the lowest task get ready to receive
	if (pid != 0) {
		// only from the rows below current task
		for (k = 1; k < lowest; k++) { // start at second diagonal
			i = N - lowest;
			j = N - lowest + k;
			while (j < N) {
					MPI_Irecv(&C[map(i,j)], 1, MPI_INT, MPI_ANY_SOURCE, map(i,j), MPI_COMM_WORLD, &requests[map(i-N+lowest,j)]);
					i++;
					j++;
			}
		}
	}

	// keep track of how many minima have been computed for each element
	int *computed = (int*)calloc(elements, sizeof(int)); // initializes to all 0's

	// calculate the total number of minima to be computed
	minima = 0;
	//int tester [SIZE] = {0};
	for (k = 1; k < highest+1; k++) { // start at second diagonal
			i = N - highest - 1;
			j = N - highest - 1 + k;
			if (pid != 0) {
				while (i < N - lowest && j < N) {
 					minima += j-i; // number of costs per element
					i++;
					j++;
				}
			} else {
				while (j < N) {
					minima += j-i;
					i++;
					j++;
				}
			}
	}

	// repeat until all costs have been computed
	while (minima > 0) {
		// apply the cost function one diagonal at a time
		for (k = 1, n = 0; k < highest+1; k++) { // start at second diagonal
			i = N - highest - 1;
			j = N - highest - 1 + k;
			if (pid != 0) {
				while (i < N - lowest && j < N) {
					optimalCost(P, pid, r, C, i, j, computed, n, &minima, requests, lowest);
					i++;
					j++;
					n++;
				}
			} else {
				while (j < N) {
					optimalCost(P, pid, r, C, i, j, computed, n, &minima, NULL, 0);
					i++;
					j++;
					n++;
				}
			}
		}
	}

	// wait until all tasks complete
    MPI_Barrier(MPI_COMM_WORLD);

    if (pid == P-1) {
	    // record end time
	    end = getClock();

	    printAsMatrix(C);

	    // output total time
	    cout << "Total time: " << end-start << " ms" << endl;
	}

    MPI_Finalize();

	return 0;
    
}
