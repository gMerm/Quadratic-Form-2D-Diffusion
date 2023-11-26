make clean
make

echo "\nexecuting serial"

time ./qf_seq

echo "\executing parallel"
./main_hybrid