make clean
make
mpirun -n 1 diffusion2d_mpi_nb_io 1 1 64 50000 0.00001
mpirun -n 4 diffusion2d_mpi_nb_io 1 1 64 50000 0.00001	