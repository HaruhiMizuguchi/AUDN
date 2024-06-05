import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from net import feature_extractor, source_classifier, domain_discriminator, prototype_classifier
from CNTGE import run_CNTGE
from utils import calculate_transfer_score, predict
from globals import *

def get_source_weights(ut_features, s_label, source_classifier, domain_discriminator, ut_preds):
    ws = np.array([calculate_transfer_score(x, source_classifier, domain_discriminator) for x in ut_features])
    V = np.sum(ut_preds[ws >= w_alpha], axis=0) / np.sum(ws >= w_alpha)
    weights = np.array([V[label_idx] for label_idx in s_label])
    return weights

def train_epoch(feature_extractor, source_classifier, domain_discriminator, prototype_classifier,
                D_s_loader, D_ut_train_loader, D_lt_loader, D_plt_loader, optimizer):
    """
    1エポック分の学習を行う関数

    Args:
        feature_extractor (nn.Module): 特徴抽出器
        source_classifier (nn.Module): ソースラベル分類器
        domain_discriminator (nn.Module): ドメイン分類器
        prototype_classifier (nn.Module): プロトタイプ分類器
        D_s_loader (DataLoader): ソースドメインのデータローダー
        D_ut_train_loader (DataLoader): ターゲットドメインのラベルなしデータのデータローダー
        D_lt_loader (DataLoader): ターゲットドメインのラベルつきデータのデータローダー
        D_plt_loader (DataLoader): ターゲットドメインの疑似ラベルつきデータのデータローダー
        optimizer (optim.Optimizer): 最適化アルゴリズム
        alpha (float): 勾配反転層の重み係数
        w_0 (float): 転送スコアの閾値

    Returns:
        float: 1エポック分の平均損失
    """
    feature_extractor.train()
    source_classifier.train()
    domain_discriminator.train()
    prototype_classifier.train()
    
    total_loss = 0
    
    for (s_data, s_label), (ut_data, _), (lt_data, lt_label), (plt_data, plt_label) in tqdm(zip(D_s_loader, D_ut_train_loader, D_lt_loader, D_plt_loader)):
        s_data, s_label, ut_data, lt_data, lt_label, plt_data, plt_label = s_data.to(device), s_label.to(device), ut_data.to(device), lt_data.to(device), lt_label.to(device), plt_data.to(device), plt_label.to(device)
        
        # 特徴抽出
        s_features = feature_extractor(s_data)
        ut_features = feature_extractor(ut_data)
        lt_features = feature_extractor(lt_data)
        plt_features = feature_extractor(plt_data)
        
        # --- ソースラベル分類器の学習 L_C ---
        s_preds = source_classifier(s_features)
        ut_preds = source_classifier(ut_features)
        lt_preds = source_classifier(lt_features)

        # 共通ラベルのインデックスを取得
        common_label_indices = (lt_label < n_source_classes).nonzero().squeeze()
        private_label_indices = (lt_label >= n_source_classes).nonzero().squeeze()

        # 共通ラベルのデータのみで損失を計算
        if common_label_indices.numel() > 0:
            source_classification_loss = F.cross_entropy(s_preds, s_label) + \
                                        F.cross_entropy(lt_preds[common_label_indices], lt_label[common_label_indices])
        else:
            source_classification_loss = F.cross_entropy(s_preds, s_label)
        
        # --- ドメイン分類器の学習 ---
        s_domain_preds = domain_discriminator(s_features)
        ut_domain_preds = domain_discriminator(ut_features)
        lt_common_domain_preds = domain_discriminator(lt_features[common_label_indices])
        
        # --- 転送スコアと重みの計算 ---
        ut_transfer_scores = np.array([calculate_transfer_score(ut_feature, source_classifier, domain_discriminator) for ut_feature in ut_features])
        source_weights = get_source_weights(ut_features, s_label, source_classifier, domain_discriminator, ut_preds)

        # --- 敵対的カリキュラム学習 L_adv ---
        adversarial_curriculum_loss = torch.mean(source_weights * torch.log(1 - s_domain_preds)) + \
                                        torch.mean((ut_transfer_scores >= w_0).float() * torch.log(ut_domain_preds)) + \
                                        torch.mean(torch.log(lt_common_domain_preds))
                                        
        # --- 多様性カリキュラム学習 L_div ---
        diverse_curriculum_loss = - torch.mean((ut_transfer_scores < w_0).float() * (torch.sum(F.softmax(ut_preds, dim=1) *
                                                                                        torch.log(F.softmax(ut_preds, dim=1)), dim=1))) \
                                    - torch.mean(torch.sum(F.softmax(source_classifier(lt_features[private_label_indices]), dim=1) * \
                                        torch.log(F.softmax(source_classifier(lt_features[private_label_indices]), dim=1))))
                                    
        # --- プロトタイプ分類器の学習 ---
        # --- クロスエントロピー L_p ---
        prototype_plt_preds = prototype_classifier(plt_features)
        prototype_lt_preds = prototype_classifier(lt_features)
        protopype_classification_loss = F.cross_entropy(prototype_plt_preds, plt_label) + \
            F.cross_entropy(prototype_lt_preds, lt_label)
        
        # --- 自己教師ありクラスタリング損失 ---
        prototypes = prototype_classifier.prototypes.data
        M = torch.cat([lt_features, prototypes], dim=0)  # lt_features と prototypes を結合

        # ドット積を計算
        similarities = torch.mm(ut_features, M.T) / tau

        # softmax を適用して確率を計算
        p = F.softmax(similarities, dim=1)

        # エントロピーを計算
        selfsupervised_clustering_loss = -torch.mean(torch.sum(p * torch.log(p), dim=1))
        
        # --- 全体の損失 ---
        loss = source_classification_loss - adversarial_curriculum_loss + diverse_curriculum_loss + \
            protopype_classification_loss + selfsupervised_clustering_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return total_loss / len(D_s_loader)

def validate(feature_extractor, source_classifier, domain_discriminator, prototype_classifier, target_test_loader, w_0):
    """
    検証データでモデルの性能を評価する関数

    Args:
        feature_extractor (nn.Module): 特徴抽出器
        source_classifier (nn.Module): ソースラベル分類器
        prototype_classifier (nn.Module): プロトタイプ分類器
        target_loader (DataLoader): ターゲットドメインのデータローダー
        w_0 (float): 転送スコアの閾値

    Returns:
        float: 平均クラス精度
    """

    feature_extractor.eval()
    source_classifier.eval()
    prototype_classifier.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data, label in target_test_loader:
            data, label = data.to(device), label.to(device)
            features = feature_extractor(data)
            preds = predict(features, source_classifier, domain_discriminator, prototype_classifier, w_0)
            correct += (preds == label).sum().item()
            total += label.size(0)

    return correct / total