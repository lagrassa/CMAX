#!/bin/bash

mpirun -np 4 python -m odium.experiment.train --env-name FetchPush-v1
