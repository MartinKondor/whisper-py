#!/bin/bash
########################################################
#                                                      #
####################!!WARNING!!#########################
# ONLY USE THIS FILE IN THE ROOT FOLDER OF WHISPER.CPP #
#                                                      #
########################################################
make
gcc -O3 -std=c11   -pthread -mavx -mavx2 -mfma -mf16c -fPIC -c ggml.c
g++ -O3 -std=c++11 -pthread --shared -fPIC -static-libstdc++ whisper.cpp ggml.o -o libwhisper.so
