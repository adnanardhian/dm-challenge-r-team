#!/usr/bin/env sh

# /modelState (writable) volume has been mounted
# /preprocessedData (writable) already mounted
# /trainingData (read-only) already mounted
# /metadata (read-only) already mounted

python train.py #insert source
