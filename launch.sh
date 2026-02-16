#!/bin.bash             # use bash as command interpreter
#$ -cwd                 # currentWorkingDirectory
#$ -N testData_LHS        # jobName
#$ -j y                 # merges output and errors
#$ -S /bin/bash         # scripting language
#$ -l h_rt=24:00:00     # jobDuration hh:mm:ss
#$ -q aerotes.q         # queueName
#$ -pe smp 5           # cpuNumber
#$ -l h=node-a-3        # forces to use node-a-3 for SU2

##############################################################

### Specify the executable...

export OMP_NUM_THREADS=1
export OMP_THREAD_LIMIT=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


python3 runLoop.py

echo End Parallel Run

