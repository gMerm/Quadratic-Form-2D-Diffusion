// sequential code
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>

int main(int argc, char** argv)
{	
	int n = 16384;
	int result=0.;

	//starting the mpi 
	MPI_Init(&argc, &argv);
	int rank;
	int size;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);

	
	if(rank==0){
		printf("I'm rank: %d\n", rank);
		double n_rankMaster=(n/2);

		// allocate memory
		double *A = (double *)malloc(n*n*sizeof(double));
		double *v = (double *)malloc(n*sizeof(double));
		double *w = (double *)malloc(n*sizeof(double));

		//send n partition to rank=1
		double n_rank1=n/2;
		MPI_Send(&n_rank1, sizeof(double), MPI_DOUBLE, 1, 88, MPI_COMM_WORLD);

		//compute A: 0 - n/2
		#pragma omp parallel num_threads(8)
		{
			#pragma omp for
				//startA=omp_get_wtime();
				/// init A_ij = (i + 2*j) / n^2
				for (int i=0; i<n_rankMaster; ++i)
					for (int j=0; j<n_rankMaster; ++j)
						A[i*n_rankMaster+j] = (i + 2.0*j) / (n*n); 
				//endA=omp_get_wtime();
		}

		//compute vector v 
		#pragma omp single
		{
			/// init v_i = 1 + 2 / (i+0.5)
            for (int i=0; i<n_rankMaster; ++i)
                v[i] = 1.0 + 2.0 / (i + 0.5);
		}

		//compute vector w
		#pragma omp single nowait
		{
			/// init w_i = 1 - i / (3.*n)
            for (int i=0; i<n_rankMaster; ++i)
                w[i] = 1.0 - i / (3.0*n);
		}
		

		//compute result (quadratic) for corresponding n-size: 0-n/2
		double local_res=0.;
		double resrank0=0.;
		#pragma omp parallel firstprivate(local_res) shared(resrank0)
        {
            #pragma omp for schedule(dynamic,8) 
            for (int i=0; i<n_rankMaster; ++i)
                for (int j=0; j<n_rankMaster; ++j)
                    local_res += v[i] * A[i*n + j] * w[j];
                    
            #pragma omp critical
                resrank0 += local_res;

        }

		#pragma omp barrier
		double res_from_rank1=0.;

		//receive from rank1 the other half of the result
		MPI_Recv(&res_from_rank1, sizeof(double), MPI_DOUBLE, 1, 98, MPI_COMM_WORLD, &status);


		//compute the whole result
		result = resrank0 + res_from_rank1;
		printf("Result: %lf\n", result);

		/// free memory
		free(A);
		free(v);
		free(w);






	}else{
		printf("I'm rank: %d\n", rank);
		double n_size=0.;

		//receive n partition
		MPI_Recv(&n_size, sizeof(double), MPI_DOUBLE, 0, 88, MPI_COMM_WORLD, &status);

		//allocate memory
		double *A = (double *)malloc(n*n*sizeof(double));
		double *v = (double *)malloc(n*sizeof(double));
		double *w = (double *)malloc(n*sizeof(double));

		//compute A: n/2 - n
		#pragma omp parallel num_threads(8)
		{
			#pragma omp for
				//startA=omp_get_wtime();
				/// init A_ij = (i + 2*j) / n^2
				for (int i=n_size; i<n; ++i)
					for (int j=n_size; j<n; ++j)
						A[i*n_size+j] = (i + 2.0*j) / (n*n); 
				//endA=omp_get_wtime();
		}
		
		//compute vector v 
		#pragma omp single
		{
			/// init v_i = 1 + 2 / (i+0.5)
            for (int i=n_size; i<n; ++i)
                v[i] = 1.0 + 2.0 / (i + 0.5);
		}

		//compute vector w
		#pragma omp single nowait
		{
			/// init w_i = 1 - i / (3.*n)
            for (int i=n_size; i<n; ++i)
                w[i] = 1.0 - i / (3.0*n);
		}

		//compute result (quadratic) for corresponding n-size: n/2-n
		double local_res=0.;
		double resrank1=0.;
		#pragma omp parallel firstprivate(local_res) shared(resrank1)
        {
            #pragma omp for schedule(dynamic,8) 
            for (int i=n_size; i<n; ++i)
                for (int j=n_size; j<n; ++j)
                    local_res += v[i] * A[i*n + j] * w[j];
                    
            #pragma omp critical
                resrank1 += local_res;

        }

		#pragma omp barrier

		//send the resrank1 to master rank to complete the sum
		MPI_Send(&resrank1, sizeof(double), MPI_DOUBLE, 0, 98, MPI_COMM_WORLD);

		/// free memory
		free(A);
		free(v);
		free(w);
	}

	//terminating the mpi_comm_world
	MPI_Finalize();

    return 0;
}

