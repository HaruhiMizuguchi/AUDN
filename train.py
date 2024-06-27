import numpy as np
import torch
import itertools
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from net import feature_extractor, source_classifier, domain_discriminator, prototype_classifier
from CNTGE import run_CNTGE
from utils import *
from globals import *

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
    
    # --- 重みとバイアスの確認 ---
    for name, param in source_classifier.named_parameters():
        print(f"--- {name} ---")
        print("平均:", param.mean().item())
        print("標準偏差:", param.std().item())
        print("最大値:", param.max().item())
        print("最小値:", param.min().item())
    # --- ここまで ---
    
    # 各データローダーのイテレーターを作成
    s_iter = iter(D_s_loader)
    ut_iter = iter(D_ut_train_loader)
    lt_iter = iter(D_lt_loader)
    plt_iter = iter(D_plt_loader) if D_plt_loader is not None else iter([])
    
    with torch.no_grad():
        # Vの計算
        all_ut_features = []
        for data, label in D_ut_train_loader:
            all_ut_features.append(data.to(device))
        all_ut_features = feature_extractor(torch.cat(all_ut_features, dim=0))
        ut_preds = source_classifier(all_ut_features)
        V = calculate_V(all_ut_features, source_classifier, domain_discriminator, ut_preds)
        
        # --- 自己教師ありクラスタリング損失 L_nc のMを計算---
        all_lt_features = []
        for data, label in D_lt_loader:
            all_lt_features.append(data.to(device))
        all_lt_features = feature_extractor(torch.cat(all_lt_features, dim=0))
        prototypes = prototype_classifier.get_prototypes()
        M = torch.cat((all_lt_features, prototypes), dim=0)
    i=0
    # 最長のDataloaderに合わせてイテレーション
    for (s_data, s_label), (ut_data, _), (lt_data, lt_label), (plt_data, plt_label) in tqdm(itertools.zip_longest(s_iter, ut_iter, lt_iter, plt_iter, fillvalue=(None, None))):
        
        source_classification_loss = torch.tensor(0.0, requires_grad=True)
        adversarial_curriculum_loss = torch.tensor(0.0, requires_grad=True)
        diverse_curriculum_loss = torch.tensor(0.0, requires_grad=True)
        prototype_classification_loss = torch.tensor(0.0, requires_grad=True)
        selfsupervised_clustering_loss = torch.tensor(0.0, requires_grad=True)
        
        config.t += 1
        if config.t <= config.total_ite:
            config.w_alpha = w_0 - (1 - config.t/config.total_ite) * alpha
        
        if s_data is not None:
            #print("S,", end=' ')
            s_data, s_label = s_data.to(device), s_label.to(device)
            s_features = feature_extractor(s_data)
            s_preds = source_classifier(s_features)
            s_domain_preds = domain_discriminator(s_features)
            source_weights = get_source_weights(s_label, V)
            # --- ソースラベル分類器の学習 L_C ---
            source_classification_loss = source_classification_loss + F.cross_entropy(s_preds, s_label)
            # --- 敵対的カリキュラム学習 L_adv ---
            adversarial_curriculum_loss = adversarial_curriculum_loss + torch.mean(source_weights * torch.log(torch.clamp(1 - s_domain_preds, min=eps)).flatten())
        
        if ut_data is not None:
            #print("ut,", end=' ')
            ut_data = ut_data.to(device)
            ut_features = feature_extractor(ut_data)
            ut_preds = source_classifier(ut_features)
            ut_domain_preds = domain_discriminator(ut_features)
            ut_transfer_scores = torch.stack([calculate_transfer_score(ut_feature, source_classifier, domain_discriminator) for ut_feature in ut_features])
            # ut_ts_above_w_alpha_indices = torch.nonzero(ut_transfer_scores >= Config.w_alpha).squeeze()
            # --- 敵対的カリキュラム学習 L_adv ---
            adversarial_curriculum_loss = adversarial_curriculum_loss + torch.mean((ut_transfer_scores >= config.w_alpha).float().flatten() * torch.log(torch.clamp(ut_domain_preds, min=eps)).flatten())
            # --- 多様性カリキュラム学習 L_div ---
            diverse_curriculum_loss = diverse_curriculum_loss - torch.mean((ut_transfer_scores < config.w_alpha).float() * (torch.sum(ut_preds * torch.log(torch.clamp(ut_preds, min=eps)), dim=1)))
            if i == 0:
                print("div ut 1:",(ut_transfer_scores < config.w_alpha).float())
                print("div ut 2:",(torch.sum(ut_preds * torch.log(torch.clamp(ut_preds, min=eps)), dim=1)))
                print("div ut:",torch.mean((ut_transfer_scores < config.w_alpha).float() * (torch.sum(ut_preds * torch.log(torch.clamp(ut_preds, min=eps)), dim=1))))
            # --- 自己教師ありクラスタリング損失 L_nc ---
            # ドット積を計算
            similarities = torch.mm(ut_features, M.T) / tau
            # softmax を適用して確率を計算
            p = F.softmax(similarities, dim=1)
            # エントロピーを計算
            selfsupervised_clustering_loss = selfsupervised_clustering_loss - torch.mean(torch.sum(p * torch.log(torch.clamp(p, min=eps)), dim=1))
            
        if lt_data is not None:
            #print("lt,", end=' ')
            lt_data, lt_label = lt_data.to(device), lt_label.to(device)
            lt_features = feature_extractor(lt_data)
            lt_preds = source_classifier(lt_features)
            prototype_lt_preds = prototype_classifier(lt_features)
            common_label_indices = (lt_label < n_source_classes).nonzero().squeeze()
            private_label_indices = (lt_label >= n_source_classes).nonzero().squeeze()
            lt_common_domain_preds = domain_discriminator(lt_features[common_label_indices])
            # --- ソースラベル分類器の学習 L_C ---
            if common_label_indices.numel() > 0:
                source_classification_loss = source_classification_loss + F.cross_entropy(lt_preds[common_label_indices], lt_label[common_label_indices])
            # --- 敵対的カリキュラム学習 L_adv ---
            if len(lt_common_domain_preds) >= 0:
                adversarial_curriculum_loss = adversarial_curriculum_loss + torch.mean(torch.log(torch.clamp(lt_common_domain_preds, min=eps)))
            # --- 多様性カリキュラム学習 L_div ---
            diverse_curriculum_loss = diverse_curriculum_loss - (torch.mean(torch.sum(source_classifier(lt_features[private_label_indices]) \
                * torch.log(torch.clamp(source_classifier(lt_features[private_label_indices]), min=eps)))))
            if i == 0:
                print("lt div 1:",source_classifier(lt_features[private_label_indices]))
                print("lt div 2:",torch.log(torch.clamp(source_classifier(lt_features[private_label_indices]), min=eps)))
            # --- プロトタイプ分類器の学習 L_p ---
            prototype_classification_loss = prototype_classification_loss + F.cross_entropy(prototype_lt_preds, lt_label)
            
        if plt_data is not None:
            #print("plt ", end=' ')
            plt_data, plt_label = plt_data.to(device), plt_label.to(device)
            plt_features = feature_extractor(plt_data)
            prototype_plt_preds = prototype_classifier(plt_features)
            # --- プロトタイプ分類器の学習 L_p ---
            prototype_classification_loss = prototype_classification_loss + F.cross_entropy(prototype_plt_preds, plt_label)
        
        #print("is not None")
        
        # --- 全体の損失 ---
        loss = source_classification_loss - adversarial_curriculum_loss + diverse_curriculum_loss + \
            prototype_classification_loss + selfsupervised_clustering_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i+=1
        if i == 1:
            print("train")
            print("source_classification_loss:", source_classification_loss.item())
            print("adversarial_curriculum_loss:", adversarial_curriculum_loss.item())
            print("diverse_curriculum_loss:", diverse_curriculum_loss.item())
            print("prototype_classification_loss:", prototype_classification_loss.item())
            print("selfsupervised_clustering_loss:", selfsupervised_clustering_loss.item())
    
    return loss