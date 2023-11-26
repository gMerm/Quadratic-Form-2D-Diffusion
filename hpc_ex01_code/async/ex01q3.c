// Exercise 1, question 3: initial code

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

void do_work(int i) {
	printf("processing %d\n", i);
	sleep(5);
}

int main(int argc, char** argv) {
	int rank;
	int size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (rank == 0)
		printf("Running with %d MPI processes\n", size);

	int M = 2;	// two tasks per process
	int input;

	if(rank == 0) {
		int N = M*size;
		srand48(time(0));

		for(int i=0; i<N; i++) {
			input = lrand48() % 1000;	// some random value
			MPI_Send(&input, 1, MPI_INT, i%size, 100, MPI_COMM_WORLD);
		}
	}

	for(int i = 0; i < M; i++) {
		MPI_Recv(&input, 1, MPI_INT, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		do_work(input);
	}

	MPI_Finalize();
	return 0;
}
