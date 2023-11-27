CC=mpicc
CFLAGS=-O2 -Wall -I/opt/homebrew/opt/libomp/include
LDFLAGS=-L/opt/homebrew/opt/libomp/lib -Xpreprocessor -fopenmp -lomp

ALL=qf_seq qf_hybrid main_hybrid

all: $(ALL)

qf_seq: qf_seq.c
	$(CC) $(CFLAGS) -o qf_seq qf_seq.c $(LDFLAGS)

qf_hybrid: qf_hybrid.c
	$(CC) $(CFLAGS) -o qf_hybrid qf_hybrid.c $(LDFLAGS)

main_hybrid: main_hybrid.c
	$(CC) $(CFLAGS) -o main_hybrid main_hybrid.c  $(LDFLAGS) 

.PHONY: clean
clean:
	-rm -f $(ALL)

