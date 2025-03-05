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

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='model name')
    parser.add_argument('--dataset', type=str, required=True, help='dataset name')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--gpu_num', type=int, default=0, help='gpu number')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    return parser.parse_args()

def main():
    args = parse_args()

    model = timm.create_model(args.model_name, pretrained=True, num_classes=10, in_chans=3)
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
        split='train',
        download=True,
        transform=transform,
        dataset = [args.dataset],
        random_seed = args.seed,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    
    device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')
    metrics_dir = f'./benchmark/classification/results/deeplearning/{args.dataset}/{args.model_name}'
    os.makedirs(metrics_dir, exist_ok=True)

    model.to(device)

    best_acc = 0

    optimizer = optim.Adadelta(model.parameters(), lr=1e-1)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        all_preds = []
        all_labels = []
        correct = 0
        total = 0

        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data[0].to(device), labels.to(device)
            optimizer.zero_grad()

            features = model.forward_features(data)
            classified_features = model.forward_head(features)
            loss = F.cross_entropy(classified_features, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(classified_features, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
        avg_loss = epoch_loss / len(train_loader)
        acc = 100 * correct / total
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        print(f'Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

        metrics_df = pd.DataFrame({
            'epoch': [epoch+1],
            'loss': [avg_loss],
            'accuracy': [acc],
            'precision': [precision],
            'recall': [recall],
            'f1': [f1]
        })

        if not os.path.exists(os.path.join(metrics_dir, 'metrics.csv')):
            metrics_df.to_csv(os.path.join(metrics_dir, 'metrics.csv'), index=False)
        else:
            metrics_df.to_csv(os.path.join(metrics_dir, 'metrics.csv'), mode='a', header=False, index=False)

        if acc > best_acc:
            best_acc = acc

            for file in os.listdir(metrics_dir):
                if file.endswith('.pth'):
                    os.remove(os.path.join(metrics_dir, file))

            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'acc': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }, os.path.join(metrics_dir, f'best_model_{epoch+1}.pth'))

    print(f'Best Accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    main()
