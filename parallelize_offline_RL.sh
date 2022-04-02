#!/bin/bash

for seed in $(seq 0 9);
do

#qsub QSUBS/SAC.qsub $seed 
#qsub QSUBS/AWAC.qsub $seed 
qsub QSUBS/AWAC_GAE.qsub $seed 
qsub QSUBS/AWAC_Q_lambda_Haru.qsub $seed 
qsub QSUBS/AWAC_Q_lambda_Peng.qsub $seed 
qsub QSUBS/AWAC_TB_lambda.qsub $seed 
#qsub QSUBS/BC.qsub $seed 
#qsub QSUBS/BCQ.qsub $seed 
#qsub QSUBS/BEAR.qsub $seed 
#qsub QSUBS/TD3_BC.qsub $seed 

done 
