#!/bin/bash

problem_size=16384
executable="./mpiParallelized"

for config in "1x1" "1x2" "1x4" "2x2" "4x1" "4x8"; do
    case $config in
        "1x1")
            echo "1x1" 
            mpirun -np 1 -x OMP_NUM_THREADS=1 $executable $problem_size ;;
        "1x2")
            echo "1x2" 
            mpirun -np 1 -x OMP_NUM_THREADS=2 $executable $problem_size ;;
        "1x4")
            echo "1x4" 
            mpirun -np 1 -x OMP_NUM_THREADS=4 $executable $problem_size ;;
        "2x2")
            echo "2x2" 
            mpirun -np 2 -x OMP_NUM_THREADS=2 $executable $problem_size ;;
        "4x1")
            echo "4x1" 
            mpirun -np 4 -x OMP_NUM_THREADS=1 $executable $problem_size ;;
        "4x8")
            echo "4x8" 
            mpirun -np 4 -x OMP_NUM_THREADS=8 $executable $problem_size ;;
        *)
            echo "Error: $config" ;;
    esac
done
