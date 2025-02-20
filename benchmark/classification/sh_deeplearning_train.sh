nohup python ./benchmark/classification/deeplearning_train.py --model_name resnet18 --dataset morph-28 --batch_size 64 --epochs 100 --gpu_num 0 --seed 42 > ./benchmark/classification/results/deeplearning/morph-28/resnet18.log 2>&1 &
sleep 1s

nohup python ./benchmark/classification/deeplearning_train.py --model_name resnet50 --dataset morph-28 --batch_size 64 --epochs 100 --gpu_num 1 --seed 42 > ./benchmark/classification/results/deeplearning/morph-28/resnet50.log 2>&1 &
sleep 1s

nohup python ./benchmark/classification/deeplearning_train.py --model_name vgg11 --dataset morph-28 --batch_size 64 --epochs 100 --gpu_num 2 --seed 42 > ./benchmark/classification/results/deeplearning/morph-28/vgg11.log 2>&1 &
sleep 1s

nohup python ./benchmark/classification/deeplearning_train.py --model_name vgg16 --dataset morph-28 --batch_size 64 --epochs 100 --gpu_num 3 --seed 42 > ./benchmark/classification/results/deeplearning/morph-28/vgg16.log 2>&1 &
sleep 1s

nohup python ./benchmark/classification/deeplearning_train.py --model_name vit_base_patch16_224 --dataset morph-28 --batch_size 64 --epochs 100 --gpu_num 4 --seed 42 > ./benchmark/classification/results/deeplearning/morph-28/vit_base_patch16_224.log 2>&1 &
sleep 1s

nohup python ./benchmark/classification/deeplearning_train.py --model_name vit_tiny_patch16_224 --dataset morph-28 --batch_size 64 --epochs 100 --gpu_num 5 --seed 42 > ./benchmark/classification/results/deeplearning/morph-28/vit_tiny_patch16_224.log 2>&1 &
sleep 1s

nohup python ./benchmark/classification/deeplearning_train.py --model_name vit_small_patch16_224 --dataset morph-28 --batch_size 64 --epochs 100 --gpu_num 6 --seed 42 > ./benchmark/classification/results/deeplearning/morph-28/vit_small_patch16_224.log 2>&1 &
sleep 1s

nohup python ./benchmark/classification/deeplearning_train.py --model_name efficientnet_b0 --dataset morph-28 --batch_size 64 --epochs 100 --gpu_num 7 --seed 42 > ./benchmark/classification/results/deeplearning/morph-28/efficientnet_b0.log 2>&1 &
sleep 1s