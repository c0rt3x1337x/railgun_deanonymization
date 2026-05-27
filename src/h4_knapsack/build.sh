#!/bin/bash
gcc -O3 -march=native -shared -fPIC -o libwitness_kernel.so witness_kernel.c
