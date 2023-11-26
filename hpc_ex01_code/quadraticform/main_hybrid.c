// sequential code
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
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
    double startRes, endRes;
    double result = 0.;
    start = omp_get_wtime(); 

    #pragma omp parallel num_threads(8)
    {
        #pragma omp for
            //startA=omp_get_wtime();
            /// init A_ij = (i + 2*j) / n^2
            for (int i=0; i<n; ++i)
                for (int j=0; j<n; ++j)
                    A[i*n+j] = (i + 2.0*j) / (n*n); 
            //endA=omp_get_wtime();
    }

    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            /// init v_i = 1 + 2 / (i+0.5)
            for (int i=0; i<n; ++i)
                v[i] = 1.0 + 2.0 / (i + 0.5);
        }

        #pragma omp single nowait
        {
            /// init w_i = 1 - i / (3.*n)
            for (int i=0; i<n; ++i)
                w[i] = 1.0 - i / (3.0*n);
        }

        
    }
        /// compute
        double local_res=0.;
        startRes=omp_get_wtime();
        #pragma omp parallel firstprivate(local_res) shared(result)
        {
            #pragma omp for schedule(dynamic,8) 
            for (int i=0; i<n; ++i)
                for (int j=0; j<n; ++j)
                    local_res += v[i] * A[i*n + j] * w[j];
                    
            #pragma omp critical
                result += local_res;

        }
            
        endRes=omp_get_wtime();
    

    end = omp_get_wtime();

    printf("Result = %lf\nTime = %lf\nResult Time = %lf\n", result, end-start, endRes-startRes);

    /// free memory
    free(A);
    free(v);
    free(w);

    return 0;
}

