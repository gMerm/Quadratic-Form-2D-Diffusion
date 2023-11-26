make clean
make
clear
echo "\nExecuting Serial: \n"

time ./qf_seq

echo "\nExecuting Parallel: \n"
time ./main_hybrid