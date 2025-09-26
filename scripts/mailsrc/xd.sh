for SEED in 1 2 3
do
  sh scripts/mailsrc/xd_train_mail.sh imagenet $SEED

  for DATASET in caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101 imagenetv2 imagenet_sketch imagenet_a imagenet_r
  do
      bash scripts/mailsrc/xd_test_mail.sh  $DATASET $SEED
  done
done