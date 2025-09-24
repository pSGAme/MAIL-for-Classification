#bash scripts/mmrl/cross_datasets_train.sh
#for DATASET in dtd eurosat ucf101 oxford_flowers oxford_pets fgvc_aircraft caltech101 food101 stanford_cars sun397 imagenet_a imagenet_r imagenet_sketch imagenetv2
#do
#    bash scripts/mmrl/cross_datasets_test.sh $DATASET
#done

sh scripts/mailsrc/xd_train_maple.sh imagenet 1

for DATASET in dtd eurosat ucf101 oxford_flowers oxford_pets fgvc_aircraft caltech101 food101 stanford_cars sun397 imagenet_a imagenet_r imagenet_sketch imagenetv2
do
    bash scripts/mailsrc/xd_test_maple.sh  $DATASET 1
done

#
#sh scripts/mailsrc/xd_train_maple.sh imagenet 2
#
#for DATASET in dtd eurosat ucf101 oxford_flowers oxford_pets fgvc_aircraft caltech101 food101 stanford_cars sun397 imagenet_a imagenet_r imagenet_sketch imagenetv2
#do
#    bash scripts/mailsrc/xd_test_maple.sh  $DATASET 2
#done
#
#sh scripts/mailsrc/xd_train_maple.sh imagenet 3
#
#for DATASET in dtd eurosat ucf101 oxford_flowers oxford_pets fgvc_aircraft caltech101 food101 stanford_cars sun397 imagenet_a imagenet_r imagenet_sketch imagenetv2
#do
#    bash scripts/mailsrc/xd_test_maple.sh  $DATASET 3
#done