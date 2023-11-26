CC=gcc-13
CFLAGS=-O2 -Wall
LDFLAGS=-fopenmp

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

