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
from torch.utils.data import Dataset, TensorDataset, DataLoader, Subset, ConcatDataset
from utils import calculate_transfer_score
from torch.nn import functional as F
import torch
import numpy as np
from globals import *

def process_in_batches(model, data, batch_size):
    model.eval()
    all_features = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size].to(device)
            batch_features = model(batch_data)
            all_features.append(batch_features.cpu())
    return torch.cat(all_features)

def Kmeans(D_ut_train, feature_extracter, k):
    # 特徴量を抽出する
    features = torch.stack([D_ut_train[i][0].to(device) for i in range(len(D_ut_train))])
    features = process_in_batches(feature_extracter, features, batch_size)
    # K-meansクラスタリング
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(features)

    # クラスタ中心とラベルの取得
    centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
    labels = torch.tensor(kmeans.labels_).to(device)
    return centroids, labels

def clustering(D_ut_train, k, feature_extracter, mode: str):
    if mode == "Kmeans":
        return Kmeans(D_ut_train, feature_extracter, k)

def devide_by_transferrable(D_ut_train, centroids, cluster_labels, source_classifier, domain_discriminator):
    w_centroids = [calculate_transfer_score(centroid, source_classifier, domain_discriminator).cpu().detach().numpy() for centroid in centroids]
    
    max_index = None
    # 転送スコアが
    # βを超える要素のインデックスを取得
    above_beta_not_max_indices = [i for i, value in enumerate(w_centroids) if (value > beta)]
    # β以下の要素のインデックスを取得
    below_beta_indices = [i for i, value in enumerate(w_centroids) if (value <= beta)]
    # 最大かつβを超えるインデックスを取得
    if above_beta_not_max_indices != []:
        max_index = np.argmax(w_centroids)
        above_beta_not_max_indices.remove(max_index)
    
    
    # 抽出されたデータポイントのインデックスを取得
    max_cluster_indices = [i for i, label in enumerate(cluster_labels) if label == max_index]
    above_beta_not_max_cluster_indices = [i for i, label in enumerate(cluster_labels) if label in above_beta_not_max_indices]
    below_beta_cluster_indices = [i for i, label in enumerate(cluster_labels) if label in below_beta_indices]
    
    # Subsetデータセットを作成
    """max_w_cluster_dataset = Subset(D_ut_train, max_cluster_indices)
    transferrable_not_max_w_cluster_dataset = Subset(D_ut_train, above_beta_not_max_cluster_indices)
    nontransferrable_dataset = Subset(D_ut_train, below_beta_cluster_indices)
    """

    # 特徴量とラベルのテンソルを使用して新しいデータセットを作成        
    if len(max_cluster_indices) != 0:
        max_w_cluster_features = torch.stack([D_ut_train[i][0] for i in range(len(D_ut_train)) if i in max_cluster_indices])
        max_w_cluster_labels = torch.tensor([D_ut_train[i][1] for i in range(len(D_ut_train)) if i in max_cluster_indices])
    else:
        max_w_cluster_features = torch.empty((0, 224, 224, 3))
        max_w_cluster_labels = torch.empty((0,))
    max_w_cluster_dataset = TensorDataset(max_w_cluster_features, max_w_cluster_labels)

    if len(above_beta_not_max_cluster_indices) != 0:
        transferrable_not_max_w_cluster_features = torch.stack([D_ut_train[i][0] for i in range(len(D_ut_train)) if i in above_beta_not_max_cluster_indices])
        transferrable_not_max_w_cluster_labels = torch.tensor([D_ut_train[i][1] for i in range(len(D_ut_train)) if i in above_beta_not_max_cluster_indices])
    else:
        transferrable_not_max_w_cluster_features = torch.empty((0, 224, 224, 3))
        transferrable_not_max_w_cluster_labels = torch.empty((0,))
    transferrable_not_max_w_cluster_dataset = TensorDataset(transferrable_not_max_w_cluster_features, transferrable_not_max_w_cluster_labels)
    
    if len(below_beta_cluster_indices) != 0:
        nontransferrable_features = torch.stack([D_ut_train[i][0] for i in range(len(D_ut_train)) if i in below_beta_cluster_indices])
        nontransferrable_labels = torch.tensor([D_ut_train[i][1] for i in range(len(D_ut_train)) if i in below_beta_cluster_indices])
    else:
        nontransferrable_features = torch.empty((0, 224, 224, 3))
        nontransferrable_labels = torch.empty((0,))
    nontransferrable_dataset = TensorDataset(nontransferrable_features, nontransferrable_labels)
    
    
    max_w_cluster_centroid = centroids[max_index]
    
    return max_w_cluster_dataset, max_w_cluster_centroid, transferrable_not_max_w_cluster_dataset, nontransferrable_dataset
    

def calc_gradient(nontransferrable_dataset, feature_extractor, source_classifier):
    gradient_norms = []
    for data, _ in nontransferrable_dataset:
        data = data.to(device).unsqueeze(0)
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
    plt_features = torch.stack([max_w_cluster_dataset[i][0] for i in range(len(max_w_cluster_dataset))])
    Y_plt = source_classifier(centroid)
    Y_plt = torch.stack([Y_plt for _ in range(len(plt_features))])
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

def run_CNTGE(D_ut_train, D_lt, D_plt, feature_extractor, source_classifier, domain_discriminator, labels_of_prototypes, k, n_r):
    
    print("start CNTGE")
    centroids, cluster_labels = clustering(D_ut_train, k, feature_extractor, mode="Kmeans")
    max_w_cluster_dataset, max_w_cluster_centroid, transferrable_not_max_w_cluster_dataset, nontransferrable_dataset = devide_by_transferrable(D_ut_train, centroids, cluster_labels, source_classifier, domain_discriminator)
    
    # print(len(max_w_cluster_dataset))
    # print(len(transferrable_not_max_w_cluster_dataset))
    if len(max_w_cluster_dataset) > 0:
        D_plt_new = psuedo_labeling(max_w_cluster_dataset, max_w_cluster_centroid, feature_extractor, source_classifier)
    else:
        D_plt_new_features = torch.empty((0, 224, 224, 3))
        D_plt_new_labels = torch.empty((0,))
        D_plt_new = TensorDataset(D_plt_new_features, D_plt_new_labels)
    labeled_nontransferrable_dataset, not_labeled_nontransferrable_dataset = AL_labeling(nontransferrable_dataset, feature_extractor, source_classifier, n_r)
    
    # ターゲットのラベル付きデータに、ALしたものを追加
    D_lt = ConcatDataset([D_lt, labeled_nontransferrable_dataset])
    # 疑似ラベル付きデータに、PLしたものを追加
    D_plt = ConcatDataset([D_plt, D_plt_new])
    new_plt_labels = list(set([D_plt[i][1] for i in range(len(D_plt))]))
    # ターゲットの未ラベルデータを、ALまたはPLしなかったものに更新
    D_ut_train = ConcatDataset([transferrable_not_max_w_cluster_dataset, not_labeled_nontransferrable_dataset])
    #print("len D_ut_train", len(D_ut_train))
    #print("len D_lt", len(D_lt))
    #print("len D_plt", len(D_plt))
    # データローダに変換
    D_ut_train_loader = DataLoader(D_ut_train, batch_size=batch_size, shuffle=True)
    D_lt_loader = DataLoader(D_lt, batch_size=batch_size, shuffle=True)
    if len(D_plt) != 0:
        D_plt_loader = DataLoader(D_plt, batch_size=batch_size, shuffle=True)
    else:
        D_plt_loader = None
    
    print("finish CNTGE")
    
    return D_ut_train, D_lt, D_plt, D_ut_train_loader, D_lt_loader, D_plt_loader, labels_of_prototypes