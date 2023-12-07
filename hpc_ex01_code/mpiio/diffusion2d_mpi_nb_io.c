#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <zlib.h>


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

//GETS CALLED BY init, IT USES D2D POINTER STRUCT TO EXTRACT ITS VALUES AND PUT THEM
//INTO THE LOCAL VARIABLES, COMPUTES bound AND gi, AND WITH A DOUBLE FOR LOOP
//IT FILLS THE rho_ WITH 0 OR 1.
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


//TAKES THE ARGUMENTS THE USER GIVES, PUTS THEM INTO A POINTER STRCUT
//D2D THAT IS USED THW WHOLE TIME, CALLOCS MEMORY FOR rho,rho_tmp_ AND diag_
//COMPUTES AND PRINTS timestep from stability condition AND CALLS initialize_density
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



//CALLED BY MAIN AFTER ALL THE VARIABLES HAVE BEEN INITIALIZED FROM
//init AND initialize_density FUNCTIONS
//EXCHANGE OF THE GHOST CELLS BETWEEN NEIGHBOORING RANKS IS BEING DONE &&
//THEN THE CENTRAL SPACES (NOT THE NEIGHBOORING ONES) ARE BEING CREATED (rho_tmp_)
//UPDATE FIRST AND LAST ROW OF EACH RANK(rho_tmp_)
//AND rho_ IS BEING SWAPPED WITH rho_tmp_ AND THE NEW ONE IS USED
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

    // Non-blocking MPI
    MPI_Request req[4];
    MPI_Status status[4];

    int prev_rank = rank_ - 1;
    int next_rank = rank_ + 1;

    // Exchange ALL necessary ghost cells with neighboring ranks.
    if (prev_rank >= 0) {
        // TODO:MPI
        MPI_Irecv(&rho_[0*real_N_+1], N_, MPI_DOUBLE, prev_rank, 100, MPI_COMM_WORLD, &req[0]);
        MPI_Isend(&rho_[1*real_N_+1], N_, MPI_DOUBLE, prev_rank, 100, MPI_COMM_WORLD, &req[1]);
    }
    else {
        // the purpose of this part will become
        // clear when using asynchronous communication.
        req[0] = MPI_REQUEST_NULL;
        req[1] = MPI_REQUEST_NULL;
    }

    if (next_rank < procs_) {
        // TODO:MPI
        MPI_Irecv(&rho_[(local_N_+1)*real_N_+1], N_, MPI_DOUBLE, next_rank, 100, MPI_COMM_WORLD, &req[2]);
        MPI_Isend(&rho_[local_N_*real_N_+1], N_, MPI_DOUBLE, next_rank, 100, MPI_COMM_WORLD, &req[3]);
    }
    else {
        // the purpose of this part will become 
        // clear when using asynchronous communication.
        req[2] = MPI_REQUEST_NULL;
        req[3] = MPI_REQUEST_NULL;
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

    // ensure boundaries have arrived
    MPI_Waitall(4, req, status);

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


//CALLED AFTER THE advance FUNCTION IS EXECUTED
//EVERY RANK COMPUTES ITS heat AND SENDS IT TO THE rank 0
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


//Writes the coordinates and the density value in a text file.
//It is meaningful only when procs == 1
void write_density_vis(Diffusion2D *D2D, const char *filename)
{
    // int N_ = D2D->N_;
    int real_N_ = D2D->real_N_;
    double *rho_ = D2D->rho_;
    double dr_ = D2D->dr_;
    double L_ = D2D->L_;

    FILE *out_file = fopen(filename, "w");
    for (int i = 0; i < real_N_; ++i) {
        for (int j = 0; j < real_N_; ++j) {
            double x_p = i*dr_ - 0.5*L_;
            double y_p = j*dr_ - 0.5*L_;
            double density_p = rho_[i * real_N_ + j];
            fprintf(out_file, "%f\t%f\t%f\n", x_p, y_p, density_p);
        }
    }
    fclose(out_file);
}


//Writes the density value in a binary file.
//It is meaningful only when procs == 1
void write_density(Diffusion2D *D2D, char *filename)
{
    // int N_ = D2D->N_;
    int real_N_ = D2D->real_N_;
    int local_N_ = D2D->local_N_;
    double *rho_ = D2D->rho_;

    FILE *out_file = fopen(filename, "w");
    for (int i = 1; i <= local_N_; ++i) {
        for (int j = 0; j < real_N_; ++j) {
            double density_p = rho_[i * real_N_ + j];
            fwrite(&density_p, sizeof(double), 1, out_file);
        }
    }
    fclose(out_file);
}


//same with write_density but with MPI I/O
void write_density_mpi(Diffusion2D *D2D, char *filename)
{
    int real_N_ = D2D->real_N_; /*DIASTASI GRAMMIS +2 GHOST CELLS*/
    int local_N_ = D2D->local_N_; /*ROWS per PROC*/
    double *rho_ = D2D->rho_;
    int rank_ = D2D->rank_;
    
    //len for every rank 
    MPI_Offset len = real_N_ * sizeof(double);
    MPI_Offset offset = rank_ * local_N_ * len;
    MPI_Status status;

    //open the file
    MPI_File f;
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &f);

    //write into the file and change offset so it moves
    for (int i=1; i<=local_N_; ++i) {
        MPI_File_write_at_all(f, offset, &rho_[i * real_N_], real_N_, MPI_DOUBLE, &status);
        offset += len;  
    }

    MPI_File_close(&f);
}



void compress_data(const void* src, size_t src_size, void* dest, size_t* dest_size) {

    //create the stream
    z_stream stream;
    stream.zalloc = Z_NULL;
    stream.zfree = Z_NULL;
    stream.opaque = Z_NULL;

    //deflate
    deflateInit(&stream, Z_BEST_COMPRESSION);

    stream.avail_in = src_size;
    stream.next_in = (Bytef*)src;
    stream.avail_out = *dest_size;
    stream.next_out = (Bytef*)dest;

    deflate(&stream, Z_FINISH);

    *dest_size = stream.total_out;

    deflateEnd(&stream);
}


void decompress_and_write(const void* src, size_t src_size, const char* output_filename, MPI_Offset offset,size_t size) {
    
    //create the stream
    z_stream stream;
    stream.zalloc = Z_NULL;
    stream.zfree = Z_NULL;
    stream.opaque = Z_NULL;
    MPI_File out;
    //start decompression
    inflateInit(&stream);
    stream.avail_in = src_size;
    stream.next_in = (Bytef*)src;

    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &out);

    //FILE* output_fp = fopen(output_filename, "ab+");
    if (!out) {
        perror("Error opening output file");
        inflateEnd(&stream);
        exit(EXIT_FAILURE);
    }

    //Move the file pointer to the correct position based on the offset
    //fseek(out, offset, SEEK_SET);
    int counter =0;
    unsigned char buffer[size];
    int ret;
    do {
        //counter++;
        stream.avail_out = sizeof(buffer);
        stream.next_out = buffer;
        offset +=(sizeof(buffer) - stream.avail_out);

        ret = inflate(&stream, Z_FINISH);
        if (ret == Z_STREAM_ERROR) {
            perror("Error during decompression");
            inflateEnd(&stream);
            MPI_File_close(&out);
            exit(EXIT_FAILURE);
        }

        //fwrite(buffer, 1, sizeof(buffer) - stream.avail_out, output_fp);
        MPI_File_write_at(out, offset, buffer, sizeof(buffer) - stream.avail_out, MPI_BYTE, MPI_STATUS_IGNORE);

    } while (stream.avail_out == 0);

    MPI_File_close(&out);
    inflateEnd(&stream);
}

//same with write_density_mpi but with compression
void write_density_mpi_compressed(Diffusion2D *D2D, char *filename){
    int real_N_ = D2D->real_N_;
    int local_N_ = D2D->local_N_;
    double *rho_ = D2D->rho_;
    int rank_ = D2D->rank_;
    int procs_ = D2D->procs_;

    //size for compression
    size_t uncompressed_size = local_N_ * real_N_ * sizeof(double);
    size_t compressed_size = compressBound(uncompressed_size);
    unsigned char *compressed_buffer = (unsigned char*)malloc(compressed_size);

    //compress buffer, same as the praxis done in write_density_mpi
    compress_data(rho_ + real_N_, uncompressed_size, compressed_buffer, &compressed_size);

    
    MPI_Offset len = compressed_size;
    MPI_Offset offset = rank_ * uncompressed_size;
    size_t buffer_size = local_N_ * real_N_ * sizeof(double);
    MPI_File f;
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &f);
    MPI_File_write_at(f, offset, compressed_buffer, len, MPI_BYTE, MPI_STATUS_IGNORE);

    MPI_File_close(&f);
   

    //decompression
    const char* decompressedFile = "density_mpi_decompressed.bin";

    decompress_and_write(compressed_buffer, compressed_size, decompressedFile, offset,buffer_size);

    free(compressed_buffer);
}






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

#ifdef _DUMP_DENSITY_
    write_density_vis(&system, "density_000000.dat");
#endif

    double t0 = MPI_Wtime();
    for (int step = 0; step < T; ++step) {
        advance(&system);

#ifndef _PERF_
        compute_diagnostics(&system, step, dt * step);
#endif

#ifdef _DUMP_DENSITY_
        if ((step > 0) && (step % 1000 == 0)) {
            char filename[256];
            sprintf(filename, "density_%06d.dat", step);
            write_density_vis(&system, filename);
        }
#endif

    }
    double t1 = MPI_Wtime();

    if (rank == 0)
        printf("Timing: %d %lf\n", N, t1-t0);

    if (procs == 1) {
        write_density(&system, (char *)"density_seq_vis.dat");
        write_density(&system, (char *)"density_seq.bin");
    }

    
    if (procs > 1){

        //For MPI I/O 
        char filenamee[256];
        MPI_File f;
        MPI_File_open(MPI_COMM_WORLD, filenamee , MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &f);
        MPI_File_set_size (f, 0);
        MPI_Offset base;
        MPI_File_get_position(f, &base);
        write_density_mpi(&system, (char *)"density_mpi.bin");
        MPI_File_close(&f);

        //For MPI I/O with compressed buffer
        char filenamee_[256];
        MPI_File f_;
        MPI_File_open(MPI_COMM_WORLD, filenamee_ , MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &f_);
        MPI_File_set_size (f_, 0);
        MPI_Offset base_;
        MPI_File_get_position(f_, &base_);
        write_density_mpi_compressed(&system, (char *)"density_mpi_compressed.bin");
        MPI_File_close(&f_);



        


    }
    

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