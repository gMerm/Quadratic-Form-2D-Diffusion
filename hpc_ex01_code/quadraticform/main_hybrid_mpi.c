#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>

int main(int argc, char** argv)
{
    int n = 16384;

    if (argc == 2)
        n = atoi(argv[1]);

    int num_procs, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int local_n = n / num_procs;
    int start = rank * local_n;
    int end = start + local_n;
    double time_start, time_end;

    // allocate memory
    double *A = (double *)malloc(n * n * sizeof(double));
    double *v = (double *)malloc(n * sizeof(double));
    double *w = (double *)malloc(n * sizeof(double));

    double result = 0.0;
    double exit_res = 0.0;
    time_start = MPI_Wtime();

    #pragma omp parallel num_threads(omp_get_max_threads())
    {
        #pragma omp for nowait
        for (int i = start; i < end; ++i)
            for (int j = 0; j < n; ++j)
                A[i * n + j] = (i + 2.0 * j) / (n * n);

        #pragma omp single nowait
        for (int i = 0; i < n; ++i)
            v[i] = 1.0 + 2.0 / (i + 0.5);

        #pragma omp single nowait
        for (int i = 0; i < n; ++i)
            w[i] = 1.0 - i / (3.0 * n);
    }

    // compute
    #pragma omp parallel
    {
        double local_res = 0.0;

        #pragma omp for schedule(dynamic, omp_get_max_threads()) nowait
        for (int i = start; i < end; ++i)
            for (int j = 0; j < n; ++j)
                local_res += v[i] * A[i * n + j] * w[j];

        #pragma omp critical
        result += local_res;
    }

    MPI_Reduce(&result, &exit_res, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    time_end = MPI_Wtime();

    if (rank == 0)
    {
        printf("Result = %lf && Time = %lf\n", exit_res, time_end - time_start);
    }

    // free memory
    free(A);
    free(v);
    free(w);
    MPI_Finalize();

    return 0;
}
