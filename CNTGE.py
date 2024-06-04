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
import torch
import numpy as np


def Kmeans(D_ut_train, k):
    # 特徴量を抽出する
    features = [D_ut_train[i][0].numpy().flatten() for i in range(len(D_ut_train))]
    # K-meansクラスタリング
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(np.array(features))

    # クラスタ中心とラベルの取得
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    return centroids, labels

def clustering(D_ut_train, k, mode: str):
    if mode == "Kmeans":
        return Kmeans(D_ut_train, k)

def devide_by_transferrable(D_ut_train, centroids, cluster_labels, source_classifier, domain_discriminator, β):
    w_centroids = [calculate_transfer_score(centroid, source_classifier, domain_discriminator) for centroid in centroids]
    # 最大のw_centroidのインデックスを取得
    max_index = np.argmax(w_centroids)
    # 最大ではないかつβを超える要素のインデックスを取得
    above_beta_not_max_indices = [i for i, value in enumerate(w_centroids) if (value > β and i != max_index)]
    # β以下の要素のインデックスを取得
    below_beta_indices = [i for i, value in enumerate(w_centroids) if value <= β]
    
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
    

def calc_gradient()

def psuedo_labeling(max_w_cluster_dataset, centroid, feature_extractor, source_classifier):
    plt_features = [max_w_cluster_dataset[i][0] for i in range(len(max_w_cluster_dataset))]
    Y_plt = source_classifier(feature_extractor(centroid))
    Y_plt = [Y_plt for _ in range(plt_features)]
    D_plt_new = torch.utils.data.TensorDataset(plt_features, Y_plt)
    return D_plt_new

def AL_labeling(nontransferrable_dataset, n_r):
    ##############
    # 勾配の計算
    ##############
    gradients
    # トップn_r個とそれ以外のもののインデックスを取得
    top_n_r_indices = np.argsort(gradients)[-n_r:]
    other_indices = [i for i in range(len(gradients)) if i not in top_n_r_indices]
    # トップn_r個のデータセットを抽出
    labeled_nontransferrable_dataset = [nontransferrable_dataset[i] for i in top_n_r_indices]
    not_labeled_nontransferrable_dataset = [nontransferrable_dataset[i] for i in other_indices]
    
    return labeled_nontransferrable_dataset, not_labeled_nontransferrable_dataset

def run_CNTGE(D_ut_train, D_lt, D_plt, source_classifier, domain_discriminator, k, beta, n_r):
    
    centroids, cluster_labels = clustering(D_ut_train, k, mode="Kmeans")
    max_w_cluster_dataset, max_w_cluster_centroid, transferrable_not_max_w_cluster_dataset, nontransferrable_dataset = devide_by_transferrable(D_ut_train, centroids, cluster_labels, source_classifier, domain_discriminator, beta)
    
    D_plt_new = psuedo_labeling(max_w_cluster_dataset, max_w_cluster_centroid)
    labeled_nontransferrable_dataset, not_labeled_nontransferrable_dataset = AL_labeling(nontransferrable_dataset, n_r)
    
    # ターゲットのラベル付きデータに、ALしたものを追加
    D_lt = ConcatDataset([D_lt, labeled_nontransferrable_dataset])
    # 疑似ラベル付きデータに、PLしたものを追加
    D_plt = ConcatDataset([D_plt, D_plt_new])
    # ターゲットの未ラベルデータを、ALまたはPLしなかったものに更新
    D_ut_train = ConcatDataset([transferrable_not_max_w_cluster_dataset, not_labeled_nontransferrable_dataset])
    
    return D_ut_train, D_lt, D_plt