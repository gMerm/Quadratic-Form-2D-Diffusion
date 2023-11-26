CC=mpicc
CFLAGS=-O2 -Wall
LDFLAGS=

ALL=ex01q3 ex01q3_sol

all: $(ALL)

ex01q3: ex01q3.c
	$(CC) $(CFLAGS) -o ex01q3 ex01q3.c $(LDFLAGS)

ex01q3_sol: ex01q3_sol.c
	$(CC) $(CFLAGS) -o ex01q3_sol ex01q3_sol.c $(LDFLAGS)


.PHONY: clean
clean:
	-rm -f $(ALL)

