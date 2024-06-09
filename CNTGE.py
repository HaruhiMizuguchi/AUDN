"""
Input:
    D_UT = {x_ij^t}(i=1,...,n_r, j=1,...,n_t/n_r) : ラベルなしターゲット
    n_r : 1ラウンドでラベル付けする個数
    n_t : ラベルなしターゲットの個数
    β : 転送スコアの閾値

Output:
    D_UT : |D_UT|^t = |D_UT|^(t-1) - 2*n_r
    D_LT : |D_LT|^t = |D_LT|^(t-1) + 2*n_r

Flow:
    1. D_UTをK-meansでクラスタリング
    2. 各クラスタの中心点 {u_i}(i=1,...,K) を計算
    3. 各中心点の転送スコア {w_t(u_i)} を計算
    4.1. w_t(u_i) > β のクラスタ
        1. 最も転送スコアが高いクラスタの全てのインスタンスに疑似ラベル付け
    4.2. w_t(u_i) < β のクラスタ
        1. 各インスタンスについて、勾配ベクトルを計算
        2. 勾配ベクトルの大きさが大きい順に n_r 個のインスタンスにラベル付け
"""
# 未実装：勾配の計算とそれ以降
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from utils import calculate_transfer_score
from torch.nn import functional as F
import torch
import numpy as np
from globals import *

def Kmeans(D_ut_train, feature_extracter, k):
    # 特徴量を抽出する
    features = feature_extracter(torch.tensor(np.array([D_ut_train[i][0].numpy() for i in range(len(D_ut_train))])))
    # print(features.size())
    # K-meansクラスタリング
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(features)

    # クラスタ中心とラベルの取得
    centroids = torch.tensor(kmeans.cluster_centers_.reshape(k, b, c, d))
    labels = torch.tensor(kmeans.labels_)
    print(centroids[0].shape)
    return centroids, labels

def clustering(D_ut_train, k, feature_extracter, mode: str):
    if mode == "Kmeans":
        return Kmeans(D_ut_train, feature_extracter, k)

def devide_by_transferrable(D_ut_train, centroids, cluster_labels, source_classifier, domain_discriminator):
    w_centroids = [calculate_transfer_score(centroid, source_classifier, domain_discriminator) for centroid in centroids]
    # 最大のw_centroidのインデックスを取得
    max_index = np.argmax(w_centroids)
    # 最大ではないかつβを超える要素のインデックスを取得
    above_beta_not_max_indices = [i for i, value in enumerate(w_centroids) if (value > beta and i != max_index)]
    # β以下の要素のインデックスを取得
    below_beta_indices = [i for i, value in enumerate(w_centroids) if value <= beta]
    
    # 抽出されたデータポイントのインデックスを取得
    max_cluster_indices = [i for i, label in enumerate(cluster_labels) if label == max_index]
    above_beta_not_max_cluster_indices = [i for i, label in enumerate(cluster_labels) in above_beta_not_max_indices]
    below_beta_cluster_indices = [i for i, label in enumerate(cluster_labels) if label in below_beta_indices]
    
    # Subsetデータセットを作成
    max_w_cluster_dataset = Subset(D_ut_train, max_cluster_indices)
    transferrable_not_max_w_cluster_dataset = Subset(D_ut_train, above_beta_not_max_cluster_indices)
    nontransferrable_dataset = Subset(D_ut_train, below_beta_cluster_indices)
    
    max_w_cluster_centroid = centroids[max_index]
    
    return max_w_cluster_dataset, max_w_cluster_centroid, transferrable_not_max_w_cluster_dataset, nontransferrable_dataset
    

def calc_gradient(nontransferrable_dataset, feature_extractor, source_classifier):
    gradient_norms = []
    for data, _ in nontransferrable_dataset:
        data = data.to(device)
        features = feature_extractor(data)
        outputs = source_classifier(features)
        pseudo_label = torch.argmax(outputs)
        loss = F.cross_entropy(outputs, torch.tensor([pseudo_label]).to(device))
        
        # 最終層のパラメータの勾配を取得
        source_classifier.zero_grad()
        loss.backward()
        gradient = source_classifier.classifier[-1].weight.grad.cpu().numpy().flatten() # 最終層の重み行列の勾配を取得
        # 勾配のノルムを計算
        gradient_norm = np.linalg.norm(gradient)
        gradient_norms.append(gradient_norm)

    return gradient_norms

def psuedo_labeling(max_w_cluster_dataset, centroid, feature_extractor, source_classifier):
    plt_features = [max_w_cluster_dataset[i][0] for i in range(len(max_w_cluster_dataset))]
    Y_plt = source_classifier(feature_extractor(centroid))
    Y_plt = [Y_plt for _ in range(plt_features)]
    D_plt_new = torch.utils.data.TensorDataset(plt_features, Y_plt)
    return D_plt_new

def AL_labeling(nontransferrable_dataset, feature_extractor, source_classifier, n_r):
    ##############
    # 勾配の計算
    ##############
    gradients_norm = calc_gradient(nontransferrable_dataset, feature_extractor, source_classifier)
    # トップn_r個とそれ以外のもののインデックスを取得
    top_n_r_indices = np.argsort(gradients_norm)[-n_r:]
    other_indices = [i for i in range(len(gradients_norm)) if i not in top_n_r_indices]
    # トップn_r個のデータセットを抽出
    labeled_nontransferrable_dataset = [nontransferrable_dataset[i] for i in top_n_r_indices]
    not_labeled_nontransferrable_dataset = [nontransferrable_dataset[i] for i in other_indices]
    
    return labeled_nontransferrable_dataset, not_labeled_nontransferrable_dataset

def run_CNTGE(D_ut_train, D_lt, D_plt, feature_extractor, source_classifier, domain_discriminator, k, n_r):
    
    centroids, cluster_labels = clustering(D_ut_train, k, feature_extractor, mode="Kmeans")
    max_w_cluster_dataset, max_w_cluster_centroid, transferrable_not_max_w_cluster_dataset, nontransferrable_dataset = devide_by_transferrable(D_ut_train, centroids, cluster_labels, source_classifier, domain_discriminator)
    
    D_plt_new = psuedo_labeling(max_w_cluster_dataset, max_w_cluster_centroid)
    labeled_nontransferrable_dataset, not_labeled_nontransferrable_dataset = AL_labeling(nontransferrable_dataset, n_r)
    
    # ターゲットのラベル付きデータに、ALしたものを追加
    D_lt = ConcatDataset([D_lt, labeled_nontransferrable_dataset])
    # 疑似ラベル付きデータに、PLしたものを追加
    D_plt = ConcatDataset([D_plt, D_plt_new])
    # ターゲットの未ラベルデータを、ALまたはPLしなかったものに更新
    D_ut_train = ConcatDataset([transferrable_not_max_w_cluster_dataset, not_labeled_nontransferrable_dataset])
    # データローダに変換
    D_ut_train_loader = DataLoader(D_ut_train, batch_size=batch_size, shuffle=True)
    D_lt_loader = DataLoader(D_lt, batch_size=batch_size, shuffle=True)
    D_plt_loader = DataLoader(D_plt, batch_size=batch_size, shuffle=True)
    
    return D_ut_train, D_lt, D_plt, D_ut_train_loader, D_lt_loader, D_plt_loader