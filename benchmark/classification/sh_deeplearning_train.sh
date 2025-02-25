DATASET=morph-128

nohup python ./benchmark/classification/deeplearning_train.py --model_name resnet18 --dataset $DATASET --batch_size 64 --epochs 100 --gpu_num 0 --seed 42 > ./benchmark/classification/results/deeplearning/$DATASET/logs/resnet18.log 2>&1 &
sleep 1s

nohup python ./benchmark/classification/deeplearning_train.py --model_name resnet50 --dataset $DATASET --batch_size 64 --epochs 100 --gpu_num 1 --seed 42 > ./benchmark/classification/results/deeplearning/$DATASET/logs/resnet50.log 2>&1 &
sleep 1s

nohup python ./benchmark/classification/deeplearning_train.py --model_name vgg11 --dataset $DATASET --batch_size 64 --epochs 100 --gpu_num 2 --seed 42 > ./benchmark/classification/results/deeplearning/$DATASET/logs/vgg11.log 2>&1 &
sleep 1s

nohup python ./benchmark/classification/deeplearning_train.py --model_name vgg16 --dataset $DATASET --batch_size 64 --epochs 100 --gpu_num 3 --seed 42 > ./benchmark/classification/results/deeplearning/$DATASET/logs/vgg16.log 2>&1 &
sleep 1s

nohup python ./benchmark/classification/deeplearning_train.py --model_name vit_base_patch16_224 --dataset $DATASET --batch_size 64 --epochs 100 --gpu_num 4 --seed 42 > ./benchmark/classification/results/deeplearning/$DATASET/logs/vit_base_patch16_224.log 2>&1 &
sleep 1s

nohup python ./benchmark/classification/deeplearning_train.py --model_name vit_tiny_patch16_224 --dataset $DATASET --batch_size 64 --epochs 100 --gpu_num 5 --seed 42 > ./benchmark/classification/results/deeplearning/$DATASET/logs/vit_tiny_patch16_224.log 2>&1 &
sleep 1s

nohup python ./benchmark/classification/deeplearning_train.py --model_name vit_small_patch16_224 --dataset $DATASET --batch_size 64 --epochs 100 --gpu_num 6 --seed 42 > ./benchmark/classification/results/deeplearning/$DATASET/logs/vit_small_patch16_224.log 2>&1 &
sleep 1s

nohup python ./benchmark/classification/deeplearning_train.py --model_name efficientnet_b0 --dataset $DATASET --batch_size 64 --epochs 100 --gpu_num 7 --seed 42 > ./benchmark/classification/results/deeplearning/$DATASET/logs/efficientnet_b0.log 2>&1 &
sleep 1s