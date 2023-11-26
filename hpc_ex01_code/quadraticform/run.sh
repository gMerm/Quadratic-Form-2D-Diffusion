make clean
make

echo "\nExecuting Serial: \n"

time ./qf_seq

echo "\nExecuting Parallel: \n"
./main_hybrid