import timm
import torch
import os
import argparse
import pandas as pd
import numpy as np

import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from ladybirdmnist.datasets import LadybirdMNIST

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='model name')
    parser.add_argument('--dataset', type=str, required=True, help='dataset name')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--shuffle', type=bool, default=False, help='shuffle')
    parser.add_argument('--gpu_num', type=int, default=0, help='gpu number')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    return parser.parse_args()

def main():
    args = parse_args()

    model = timm.create_model(args.model_name, pretrained=True, num_classes=0, in_chans=3)
    config = timm.data.resolve_model_data_config(model)
    if '-28' in args.dataset:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224,224)),
            timm.data.create_transform(**config, is_training=True)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            timm.data.create_transform(**config, is_training=True)
        ])

    train_dataset = LadybirdMNIST(
        root='./data/LadybirdMNIST',
        train=True,
        download=True,
        transform=transform,
        dataset = [args.dataset],
        random_seed = args.seed,
        shuffle = args.shuffle,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=4,
    )

    device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)
    
    embeddings = []
    labels_list = []
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data[0].to(device), labels
            features = model.forward_features(data)
            features = model.forward_head(features)
            embeddings.append(features.detach().cpu().numpy())
            labels_list.append(labels.numpy())
            
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    np.savez(f'./benchmark/embedding/results/{args.dataset}/{args.model_name}.npz', embeddings=embeddings, labels=labels)

if __name__ == '__main__':
    main()