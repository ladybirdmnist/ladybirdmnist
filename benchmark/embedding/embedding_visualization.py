import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
from umap import UMAP

if __name__ == '__main__':
    dataset = 'morph-128'
    dpath = f'./benchmark/embedding/results/{dataset}'
    model_names = ['resnet18', 'resnet50', 'vgg11', 'vgg16', 'vit_base_patch16_224', 'vit_tiny_patch16_224', 'vit_small_patch16_224', 'efficientnet_b0']
    for model_name in model_names:
        data = np.load(f'{dpath}/{model_name}.npz')
        embeddings = data['embeddings']
        labels = data['labels']
        saved_dir =  f'{dpath}/visualization/{model_name}'
        os.makedirs(saved_dir, exist_ok=True)

        class_means = {}
        for label in np.unique(labels):
            class_means[label] = np.mean(embeddings[labels == label], axis=0)

        # mahalanobis distance
        epsilon = 1e-6
        cov_matrix = np.cov(embeddings, rowvar=False)
        inv_cov_matrix = np.linalg.inv(cov_matrix + epsilon * np.eye(cov_matrix.shape[0]))

        distances_matrix = np.zeros((10, 10))
        for i in range(10):
            for j in range(10):
                if i != j:
                    distance = mahalanobis(class_means[i], class_means[j], inv_cov_matrix)
                    distances_matrix[i, j] = distance

        # normalize distances matrix
        distances_matrix = (distances_matrix - np.min(distances_matrix)) / (np.max(distances_matrix) - np.min(distances_matrix))
        
        plt.figure(figsize=(10, 10))
        sns.heatmap(distances_matrix, annot=True, cmap='viridis', fmt='.2f')
        plt.title('Class-wise Mahalanobis Distance Matrix Heatmap')
        plt.savefig(f'{saved_dir}/mahalanobis_distance.png')

        # umap

        print(f'start {model_name} umap')
        umap = UMAP(n_components=2, random_state=42)
        umap_result = umap.fit_transform(embeddings)
        np.save(f'{saved_dir}/umap_result.npy', umap_result)

        plt.figure(figsize=(10, 8))
        plt.scatter(umap_result[:, 0], umap_result[:, 1], c=labels, cmap='viridis', s=5, alpha=0.3)
        plt.colorbar(label='Class Label')
        plt.title('UMAP Visualization of Embeddings')
        plt.savefig(f'{saved_dir}/umap.png')

        # pca
        print(f'start {model_name} pca')
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(embeddings)
        np.save(f'{saved_dir}/pca_result.npy', pca_result)

        plt.figure(figsize=(10, 8))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='viridis', s=5, alpha=0.3)
        plt.colorbar(label='Class Label')
        plt.title('PCA Visualization of Embeddings')
        plt.savefig(f'{saved_dir}/pca.png')

        # t-sne
        print(f'start {model_name} tsne')
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(embeddings)
        np.save(f'{saved_dir}/tsne_result.npy', tsne_result)

        plt.figure(figsize=(10, 8))
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis', s=5, alpha=0.3)
        plt.colorbar(label='Class Label')
        plt.title('t-SNE Visualization of Embeddings')
        plt.savefig(f'{saved_dir}/tsne.png')
