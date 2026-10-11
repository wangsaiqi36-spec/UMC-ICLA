import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

def DPC_MK(X, n_clusters, k_rate=0.1, device='cpu'):
    """
    Density Peaks Clustering with Mixed K-Nearest-Neighbor strategy (DPC-MK)
    """
    X = torch.from_numpy(X).float().to(device)
    N = X.size(0)
    k = max(int(N * k_rate), 1)

    # 密度计算
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
    nbrs.fit(X.cpu().numpy())
    dist, idx = nbrs.kneighbors(X.cpu().numpy())
    idx = torch.from_numpy(idx[:, 1:]).long().to(device)
    dist = torch.from_numpy(dist[:, 1:]).float().to(device)

    knn_mask = torch.zeros(N, N, device=device, dtype=torch.bool)
    knn_mask.scatter_(1, idx, 1)
    RNN = (knn_mask.t() & knn_mask).sum(1).float()
    SNN = (knn_mask.float() @ knn_mask.t().float()).diag()
    dist_contrib = (1.0 / (dist + 1e-5)).sum(1)
    rho = RNN + torch.sqrt(SNN.float()) + dist_contrib

    # 相对距离
    dist_matrix = torch.cdist(X, X, p=2)
    rho_ord = rho.argsort(descending=True)
    delta = torch.zeros(N, device=device)
    delta[rho_ord[0]] = dist_matrix[rho_ord[0], :].max()

    for i in range(1, N):
        higher_mask = rho_ord[:i]
        delta[rho_ord[i]] = dist_matrix[rho_ord[i], higher_mask].min()

    # 中心选择
    gamma = rho * delta
    centers_idx = gamma.topk(n_clusters).indices

    # 标签分配
    labels = torch.full((N,), -1, device=device, dtype=torch.long)
    labels[centers_idx] = torch.arange(n_clusters, device=device)

    rho_ord = rho.argsort(descending=True)
    for i in rho_ord:
        if labels[i] == -1:
            # Find nearest neighbor with higher density
            dist_to_higher_density_points = dist_matrix[i, rho < rho[i]]
            if len(dist_to_higher_density_points) > 0:
                nearest_neighbor_idx = torch.argmin(dist_to_higher_density_points)
                original_indices = torch.where(rho < rho[i])[0]
                labels[i] = labels[original_indices[nearest_neighbor_idx]]
            else:
                # If no points have higher density, it's a noise point or an isolated cluster
                # Assign it to the nearest cluster center
                dists_to_centers = dist_matrix[i, centers_idx]
                labels[i] = labels[centers_idx[torch.argmin(dists_to_centers)]]

    # 计算聚类中心
    centroids = torch.zeros(n_clusters, X.size(1), device=device)
    for i in range(n_clusters):
        centroids[i] = X[labels == i].mean(0)

    return labels.cpu().numpy(), centroids.cpu().numpy()