#!/bin/bash

nvcc ./src/dispositivos.cu 
./a.out
echo

nvcc ./src/desempeno.cu
./a.out 0
echo
./a.out 1

nvcc ./src/pdisp.cu
./a.out
echo
echo

nvcc ./src/nucleos.cu
./a.out 0
echo
./a.out 1

rm ./a.out
