// sequential code
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char** argv)
{
	int n = 16384;

	if (argc == 2)
		n = atoi(argv[1]);

	// allocate memory
	double *A = (double *)malloc(n*n*sizeof(double));
	double *v = (double *)malloc(n*sizeof(double));
	double *w = (double *)malloc(n*sizeof(double));

	double start,end;
    start = omp_get_wtime(); 

	/// init A_ij = (i + 2*j) / n^2
	for (int i=0; i<n; ++i)
		for (int j=0; j<n; ++j){
			A[i*n+j] = (i + 2.0*j) / (n*n);
			//printf("%lf\n", A[i*n+j]);
		}

	/// init v_i = 1 + 2 / (i+0.5)
	for (int i=0; i<n; ++i){
		v[i] = 1.0 + 2.0 / (i + 0.5);
		//printf("%lf\n", v[i]);
	}

	/// init w_i = 1 - i / (3.*n)
	for (int i=0; i<n; ++i){
		w[i] = 1.0 - i / (3.0*n);
		//printf("%lf\n", w[i]);
	}

	/// compute
	double result = 0.;
	for (int i=0; i<n; ++i)
		for (int j=0; j<n; ++j){
			result += v[i] * A[i*n + j] * w[j];
			//printf("local: %lf\n", result);
			//printf("local v: %lf * local a: %lf * local w: %lf -> %lf\n", v[i],A[i * n + j],w[j],result );
		}

	end = omp_get_wtime();

	printf("Result = %lf && Time = %lf\n", result,end-start);

	/// free memory
	free(A);
	free(v);
	free(w);

	return 0;
}
