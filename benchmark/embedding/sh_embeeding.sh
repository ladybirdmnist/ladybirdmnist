DATASET="morph-28"

nohup python ./benchmark/embedding/embedding.py --model_name resnet18 --dataset $DATASET --gpu_num 0 --batch_size 64 --shuffle False --seed 42 > ./benchmark/embedding/results/$DATASET/logs/resnet18.log 2>&1 &
sleep 1s

nohup python ./benchmark/embedding/embedding.py --model_name resnet50 --dataset $DATASET --gpu_num 1 --batch_size 64 --shuffle False --seed 42 > ./benchmark/embedding/results/$DATASET/logs/resnet50.log 2>&1 &
sleep 1s

nohup python ./benchmark/embedding/embedding.py --model_name vgg11 --dataset $DATASET --gpu_num 2 --batch_size 64 --shuffle False --seed 42 > ./benchmark/embedding/results/$DATASET/logs/vgg11.log 2>&1 &
sleep 1s

nohup python ./benchmark/embedding/embedding.py --model_name vgg16 --dataset $DATASET --gpu_num 3 --batch_size 64 --shuffle False --seed 42 > ./benchmark/embedding/results/$DATASET/logs/vgg16.log 2>&1 &
sleep 1s

nohup python ./benchmark/embedding/embedding.py --model_name vit_base_patch16_224 --dataset $DATASET --gpu_num 4 --batch_size 64 --shuffle False --seed 42 > ./benchmark/embedding/results/$DATASET/logs/vit_base_patch16_224.log 2>&1 &
sleep 1s

nohup python ./benchmark/embedding/embedding.py --model_name vit_tiny_patch16_224 --dataset $DATASET --gpu_num 5 --batch_size 64 --shuffle False --seed 42 > ./benchmark/embedding/results/$DATASET/logs/vit_tiny_patch16_224.log 2>&1 &
sleep 1s

nohup python ./benchmark/embedding/embedding.py --model_name vit_small_patch16_224 --dataset $DATASET --gpu_num 6 --batch_size 64 --shuffle False --seed 42 > ./benchmark/embedding/results/$DATASET/logs/vit_small_patch16_224.log 2>&1 &
sleep 1s

nohup python ./benchmark/embedding/embedding.py --model_name efficientnet_b0 --dataset $DATASET --gpu_num 7 --batch_size 64 --shuffle False --seed 42 > ./benchmark/embedding/results/$DATASET/logs/efficientnet_b0.log 2>&1 &