// Exercise 1, question 3: todo (put your solution here)

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

void do_work(int i,int rank) {
    printf("processing %d, rank = %d\n", i, rank);
    sleep(5);
}

int main(int argc, char** argv) {
    int rank;
    int size;

    MPI_Init(&argc, &argv);
    MPI_Request request;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
        printf("Running with %d MPI processes\n", size);

    int M = 2;    // two tasks per process
    int input;

    if (rank == 0) {
        int N_rec = M * (size - 1); // all different receivers
		int receiver=0;
        srand48(time(0));

        for (int i = 0; i < N_rec; i++) {
            input = lrand48() % 1000;
			receiver = i%(size-1) + 1;

			printf("sending to rank %d\n",receiver);
            MPI_Isend(&input, 1, MPI_INT, receiver, 100, MPI_COMM_WORLD, &request);
        }

        for (int i = 0; i < M; i++) {
            input = lrand48() % 1000;
            do_work(input,rank);
        }

    } 
	else 
	{
		int input_arr[M];
        MPI_Request recv_request[M];

        for (int i = 0; i < M; i++) {
            MPI_Irecv(&input_arr[i], 1, MPI_INT, 0, 100, MPI_COMM_WORLD, &recv_request[i]);
			printf("Receiving, rank = %d\n",rank);
        }
		MPI_Waitall(M, recv_request, MPI_STATUSES_IGNORE);

        for (int i = 0; i < M; i++) {
            do_work(input_arr[i],rank);
        }

		
    }

    MPI_Finalize();
    return 0;
}
