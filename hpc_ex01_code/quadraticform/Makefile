CC=gcc
CFLAGS=-O2 -Wall
LDFLAGS=

ALL=qf_seq qf_hybrid

all: $(ALL)

qf_seq: qf_seq.c
	$(CC) $(CFLAGS) -o qf_seq qf_seq.c $(LDFLAGS)

qf_hybrid: qf_hybrid.c
	$(CC) $(CFLAGS) -o qf_hybrid qf_hybrid.c $(LDFLAGS)


.PHONY: clean
clean:
	-rm -f $(ALL)

