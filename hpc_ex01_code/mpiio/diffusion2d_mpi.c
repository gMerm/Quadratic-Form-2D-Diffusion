#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>



typedef struct Diagnostics_s
{
    double time;
    double heat;
} Diagnostics;

typedef struct Diffusion2D_s
{
    double D_, L_, T_;
    int N_, Ntot_, real_N_;
    double dr_, dt_, fac_;
    int rank_, procs_;
    int local_N_;
    double *rho_, *rho_tmp_;
    Diagnostics *diag_;
} Diffusion2D;

void initialize_density(Diffusion2D *D2D)
{
    int real_N_ = D2D->real_N_;
    int N_ = D2D->N_;
    int local_N_ = D2D->local_N_;
    double *rho_ = D2D->rho_;
    double dr_ = D2D->dr_;
    double L_ = D2D->L_;
    int rank_ = D2D->rank_;
    int procs_ = D2D->procs_;
    int gi;

    /// Initialize rho(x, y, t=0).
    double bound = 0.25 * L_;

    for (int i = 1; i <= local_N_; ++i) {
        gi = rank_ * (N_ / procs_) + i; // convert local index to global index
        for (int j = 1; j <= N_; ++j) {
            if (fabs((gi - 1)*dr_ - 0.5*L_) < bound && fabs((j-1)*dr_ - 0.5*L_) < bound) {
                rho_[i*real_N_ + j] = 1;
            } else {
                rho_[i*real_N_ + j] = 0;
            }
        }
    }
}


//The init function initializes the Diffusion2D structure, 
//including allocating memory for density arrays and setting up parameters.
void init(Diffusion2D *D2D,
                const double D,
                const double L,
                const int N,
                const int T,
                const double dt,
                const int rank,
                const int procs)
{
    D2D->D_ = D;
    D2D->L_ = L;
    D2D->N_ = N;
    D2D->T_ = T;
    D2D->dt_ = dt;
    D2D->rank_ = rank;
    D2D->procs_ = procs;

    // Real space grid spacing.
    D2D->dr_ = D2D->L_ / (D2D->N_ - 1);

    // Stencil factor.
    D2D->fac_ = D2D->dt_ * D2D->D_ / (D2D->dr_ * D2D->dr_);

    // Number of rows per process.
    D2D->local_N_ = D2D->N_ / D2D->procs_;

    // Small correction for the last process.
    if (D2D->rank_ == D2D->procs_ - 1)
        D2D->local_N_ += D2D->N_ % D2D->procs_;

    // Actual dimension of a row (+2 for the ghost cells).
    D2D->real_N_ = D2D->N_ + 2;

    // Total number of cells.
    D2D->Ntot_ = (D2D->local_N_ + 2) * (D2D->N_ + 2);

    D2D->rho_ = (double *)calloc(D2D->Ntot_, sizeof(double));
    D2D->rho_tmp_ = (double *)calloc(D2D->Ntot_, sizeof(double));
    D2D->diag_ = (Diagnostics *)calloc(D2D->T_, sizeof(Diagnostics));

    // Check that the timestep satisfies the restriction for stability.
    if (D2D->rank_ == 0)
        printf("timestep from stability condition is %e\n", D2D->dr_ * D2D->dr_ / (4.0 * D2D->D_));

    initialize_density(D2D);
}


//The advance function performs the time-stepping of the simulation. 
//It exchanges ghost cells, updates the interior of the local domain using the diffusion equation,
//and swaps pointers to avoid unnecessary copying of data.
void advance(Diffusion2D *D2D)
{
    int N_ = D2D->N_;
    int real_N_ = D2D->real_N_;
    int local_N_ = D2D->local_N_;
    double *rho_ = D2D->rho_;
    double *rho_tmp_ = D2D->rho_tmp_;
    double fac_ = D2D->fac_;
    int rank_ = D2D->rank_;
    int procs_ = D2D->procs_;

    MPI_Status status[2];

    int prev_rank = rank_ - 1;
    int next_rank = rank_ + 1;

    // Exchange ALL necessary ghost cells with neighboring ranks.
    if (prev_rank >= 0) {
        // TODO:MPI
        MPI_Send(&rho_[           1*real_N_+1], N_, MPI_DOUBLE, prev_rank, 100, MPI_COMM_WORLD);
        MPI_Recv(&rho_[           0*real_N_+1], N_, MPI_DOUBLE, prev_rank, 100, MPI_COMM_WORLD, &status[0]);
    }
    else {
        // the purpose of this part will become
        // clear when using asynchronous communication.
    }

    if (next_rank < procs_) {
        // TODO:MPI
        MPI_Recv(&rho_[(local_N_+1)*real_N_+1], N_, MPI_DOUBLE, next_rank, 100, MPI_COMM_WORLD, &status[1]);
        MPI_Send(&rho_[    local_N_*real_N_+1], N_, MPI_DOUBLE, next_rank, 100, MPI_COMM_WORLD);
    }
    else {
        // the purpose of this part will become 
        // clear when using asynchronous communication.
    }

    // Central differences in space, forward Euler in time with Dirichlet
    // boundaries.
    for (int i = 2; i < local_N_; ++i) {
        for (int j = 1; j <= N_; ++j) {
            rho_tmp_[i*real_N_ + j] = rho_[i*real_N_ + j] +
                                     fac_
                                     *
                                     (
                                     + rho_[i*real_N_ + (j+1)]
                                     + rho_[i*real_N_ + (j-1)]
                                     + rho_[(i+1)*real_N_ + j]
                                     + rho_[(i-1)*real_N_ + j]
                                     - 4.*rho_[i*real_N_ + j]
                                     );
        }
    }

    // Note: This exercise is about synchronous communication, but the
    // template code is formulated for asynchronous. In the latter case,
    // when non-blocking send/recv is used, you would need to add an
    // MPI_Wait here to make sure the incoming data arrives before
    // evaluating the boundary cells (first and last row of the local
    // matrix),
    // Namely, network communication takes time and you always want to
    // perform some work while waiting. In this code it means making the
    // diffusion step for the inner part of the grid, which doesn't require
    // any data from other nodes. Afterwards, when the data from
    // neighboring nodes arrives, the first and last row are handled.
    // As this is a synchronous-only exercise, feel free to merge the
    // following for loops into the previous ones.

    // Update the first and the last rows of each rank.
    for (int i = 1; i <= local_N_; i += local_N_- 1) {
        for (int j = 1; j <= N_; ++j) {
            rho_tmp_[i*real_N_ + j] = rho_[i*real_N_ + j] +
                                     fac_
                                     *
                                     (
                                     + rho_[i*real_N_ + (j+1)]
                                     + rho_[i*real_N_ + (j-1)]
                                     + rho_[(i+1)*real_N_ + j]
                                     + rho_[(i-1)*real_N_ + j]
                                     - 4.*rho_[i*real_N_ + j]
                                     );
        }
    }



    // Swap rho_ with rho_tmp_. This is much more efficient,
    // because it does not copy element by element, just replaces storage
    // pointers.
    double *tmp_ = D2D->rho_tmp_;
    D2D->rho_tmp_ = D2D->rho_;
    D2D->rho_ = tmp_;
}


//The code computes diagnostics at each time step, including the total heat in the system. 
//The compute_diagnostics function uses MPI reduction (MPI_Reduce) 
//to gather the total heat across all ranks.
void compute_diagnostics(Diffusion2D *D2D, const int step, const double t)
{
    int N_ = D2D->N_;
    int real_N_ = D2D->real_N_;
    int local_N_ = D2D->local_N_;
    double *rho_ = D2D->rho_;
    double dr_ = D2D->dr_;
    int rank_ = D2D->rank_;

    double heat = 0.0;
    for(int i = 1; i <= local_N_; ++i)
        for(int j = 1; j <= N_; ++j)
            heat += rho_[i*real_N_ + j] * dr_ * dr_;

    // TODO:MPI, reduce heat (sum)
    MPI_Reduce(rank_ == 0? MPI_IN_PLACE: &heat, &heat, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); 


    if (rank_ == 0) {
#if DEBUG
        printf("t = %lf heat = %lf\n", t, heat);
#endif
        D2D->diag_[step].time = t;
        D2D->diag_[step].heat = heat;
    }
}


void write_diagnostics(Diffusion2D *D2D, const char *filename)
{

    FILE *out_file = fopen(filename, "w");
    for (int i = 0; i < D2D->T_; i++)
        fprintf(out_file, "%f\t%f\n", D2D->diag_[i].time, D2D->diag_[i].heat);
    fclose(out_file);
}


//The main function initializes MPI, calls the initialization function, 
//performs the time-stepping loop, computes diagnostics, and writes the results to a file.
int main(int argc, char* argv[])
{
    if (argc < 6) {
        printf("Usage: %s D L T N dt\n", argv[0]);
        return 1;
    }

    int rank, procs;
    //TODO:MPI Initialize MPI, number of ranks (rank) and number of processes (nprocs) involved in the communicator
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    const double D = atof(argv[1]);
    const double L = atoi(argv[2]);
    const int N = atoi(argv[3]);
    const int T = atoi(argv[4]);
    const double dt = atof(argv[5]);

    Diffusion2D system;

    init(&system, D, L, N, T, dt, rank, procs);

    double t0 = MPI_Wtime();
    for (int step = 0; step < T; ++step) {
        advance(&system);
#ifndef _PERF_
        compute_diagnostics(&system, step, dt * step);
#endif
    }
    double t1 = MPI_Wtime();

    if (rank == 0)
        printf("Timing: %d %lf\n", N, t1-t0);

#ifndef _PERF_
    if (rank == 0) {
        char diagnostics_filename[256];
        sprintf(diagnostics_filename, "diagnostics_mpi_%d.dat", procs);
        write_diagnostics(&system, diagnostics_filename);
    }
#endif

    MPI_Finalize();
    return 0;
}
