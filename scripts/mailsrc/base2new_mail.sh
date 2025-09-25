#!/bin/bash

#cd ../..

# custom config
DATASET=$1
CFG=base2new
TRAINER=MAILSRC_Trainer
SHOTS=16
for SEED in 1 2 3
do
  sh scripts/mailsrc/base2new_train_mail.sh $DATASET $SEED
done


for SEED in 1 2 3
do
  sh scripts/mailsrc/base2new_test_mail.sh $DATASET $SEED
done


echo "Parse Base Results................."
path_base=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/
python computeHM.py $path_base

echo "Parse New Results................."
path_new=output/base2new/test_new/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/

python computeHM.py $path_new

