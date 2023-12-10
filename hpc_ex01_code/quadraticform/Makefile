CC=mpicc
CFLAGS=-O2 -Wall -I/opt/homebrew/opt/libomp/include
LDFLAGS=-L/opt/homebrew/opt/libomp/lib -Xpreprocessor -fopenmp -lomp


ALL=qf_seq qf_hybrid main_hybrid main_hybrid_mpi

all: $(ALL)

qf_seq: qf_seq.c
	gcc-13 qf_seq.c -o sequential -fopenmp -O2

main_hybrid: main_hybrid.c
	gcc-13 main_hybrid.c -o parallelized -fopenmp -O2 -march=native

main_hybrid_mpi: main_hybrid_mpi.c
	$(CC) $(CFLAGS) -o mpiParallelized main_hybrid_mpi.c  $(LDFLAGS) 

.PHONY: clean
clean:
	-rm -f $(ALL)
	-rm -r parallelized
	-rm -r sequential
	-rm -r mpiParallelized

