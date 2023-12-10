make clean
make
clear
echo "Executing Serial: \n"
time ./sequential 

echo "\nExecuting Parallel: \n"
time ./parallelized 

echo "\nExecuting MPI: \n"
time mpirun -np 4 ./mpiParallelized

