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
from utils import *
from globals import *

def train_wo_plt_epoch(feature_extractor, source_classifier, domain_discriminator, prototype_classifier,
                D_s_loader, D_ut_train_loader, D_lt_loader, optimizer):
    """
    疑似ラベル付きデータセットが空の場合に、1エポック分の学習を行う関数

    Args:
        feature_extractor (nn.Module): 特徴抽出器
        source_classifier (nn.Module): ソースラベル分類器
        domain_discriminator (nn.Module): ドメイン分類器
        prototype_classifier (nn.Module): プロトタイプ分類器
        D_s_loader (DataLoader): ソースドメインのデータローダー
        D_ut_train_loader (DataLoader): ターゲットドメインのラベルなしデータのデータローダー
        D_lt_loader (DataLoader): ターゲットドメインのラベルつきデータのデータローダー
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
    
    for (s_data, s_label), (ut_data, _), (lt_data, lt_label) in tqdm(zip(D_s_loader, D_ut_train_loader, D_lt_loader)):
        
        config.t += 1
        if config.t <= config.total_ite:
            config.w_alpha = w_0 - (1 - config.t/config.total_ite) * alpha
        
        s_data, s_label = s_data.to(device), s_label.to(device)
        ut_data = ut_data.to(device)
        lt_data, lt_label = lt_data.to(device), lt_label.to(device)
        
        # 特徴抽出
        s_features = feature_extractor(s_data)
        ut_features = feature_extractor(ut_data)
        lt_features = feature_extractor(lt_data)
        
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
            print("souce_classification_loss,s:",s_preds.size(), s_label.size())
            print("souce_classification_loss,lt:",lt_preds.size(), lt_label.size())
        else:
            source_classification_loss = F.cross_entropy(s_preds, s_label)
            print("souce_classification_loss,s:",s_preds.size(), s_label.size())
        
        # --- ドメイン分類器の学習 ---
        s_domain_preds = domain_discriminator(s_features)
        ut_domain_preds = domain_discriminator(ut_features)
        lt_common_domain_preds = domain_discriminator(lt_features[common_label_indices])
        
        # --- 転送スコアと重みの計算 ---
        ut_transfer_scores = torch.stack([calculate_transfer_score(ut_feature, source_classifier, domain_discriminator) for ut_feature in ut_features])
        source_weights = get_source_weights(ut_features, s_label, source_classifier, domain_discriminator, ut_preds)

        # --- 敵対的カリキュラム学習 L_adv ---
        if len(lt_common_domain_preds) == 0:
            adversarial_curriculum_loss = torch.mean(source_weights * torch.log(torch.clamp(1 - s_domain_preds, min=eps)).flatten()) + \
                                        torch.mean((ut_transfer_scores >= config.w_alpha).float().flatten() * torch.log(torch.clamp(ut_domain_preds, min=eps)).flatten())
            print("adversarial_curriculum_loss,source_weights:",source_weights.size(),torch.log(torch.clamp(1 - s_domain_preds, min=eps)).size(),(source_weights * torch.log(torch.clamp(1 - s_domain_preds, min=eps))).size())
            print("adversarial_curriculum_loss,source_weights:",(ut_transfer_scores >= config.w_alpha).float().size(),torch.log(torch.clamp(ut_domain_preds, min=eps)).size())
        else:
            adversarial_curriculum_loss = torch.mean(source_weights * torch.log(torch.clamp(1 - s_domain_preds, min=eps)).flatten()) + \
                                            torch.mean((ut_transfer_scores >= config.w_alpha).float().flatten() * torch.log(torch.clamp(ut_domain_preds, min=eps)).flatten()) + \
                                            torch.mean(torch.log(torch.clamp(lt_common_domain_preds, min=eps)).flatten())
            print("adversarial_curriculum_loss,source_weights:",source_weights.size(),torch.log(torch.clamp(1 - s_domain_preds, min=eps)).flatten().size(),(source_weights * torch.log(torch.clamp(1 - s_domain_preds, min=eps)).flatten()).size())
            print("adversarial_curriculum_loss,source_weights:",(ut_transfer_scores >= config.w_alpha).float().flatten().size(),torch.log(torch.clamp(ut_domain_preds, min=eps)).flatten().size())
            print("adversarial_curriculum_loss,source_weights:",torch.log(torch.clamp(lt_common_domain_preds, min=eps)).size())
        
        # --- 多様性カリキュラム学習 L_div ---
        diverse_curriculum_loss = - torch.mean((ut_transfer_scores < config.w_alpha).float() * (torch.sum(ut_preds * torch.log(torch.clamp(ut_preds, min=eps)), dim=1))) \
                                    - torch.mean(torch.sum(source_classifier(lt_features[private_label_indices]) * \
                                        torch.log(torch.clamp(source_classifier(lt_features[private_label_indices]), min=eps))))
        print("diverse:",(ut_transfer_scores < config.w_alpha).float().size(),ut_preds.size(),torch.log(torch.clamp(ut_preds, min=eps)).size(),torch.sum(ut_preds * torch.log(torch.clamp(ut_preds, min=eps)), dim=1).size(),((ut_transfer_scores < config.w_alpha).float() * (torch.sum(ut_preds * torch.log(torch.clamp(ut_preds, min=eps)), dim=1))).size())
        print("diverse:",source_classifier(lt_features[private_label_indices]).size(),torch.log(torch.clamp(source_classifier(lt_features[private_label_indices]), min=eps)).size()
              , (source_classifier(lt_features[private_label_indices]) * torch.log(torch.clamp(source_classifier(lt_features[private_label_indices]), min=eps))).size()
              , torch.sum(source_classifier(lt_features[private_label_indices]) * torch.log(torch.clamp(source_classifier(lt_features[private_label_indices]), min=eps))).size())
                                    
        # --- プロトタイプ分類器の学習 ---
        # --- クロスエントロピー L_p ---
        prototype_lt_preds = prototype_classifier(lt_features)
        prototype_classification_loss = F.cross_entropy(prototype_lt_preds, lt_label)
        
        # --- 自己教師ありクラスタリング損失 ---
        prototypes = prototype_classifier.get_prototypes()
        M = torch.cat((lt_features, prototypes), dim=0)  # lt_features と prototypes を結合

        # ドット積を計算
        similarities = torch.mm(ut_features, M.T) / tau

        # softmax を適用して確率を計算
        p = F.softmax(similarities, dim=1)

        # エントロピーを計算
        selfsupervised_clustering_loss = -torch.mean(torch.sum(p * torch.log(torch.clamp(p, min=eps)), dim=1))
        
        # --- 全体の損失 ---
        loss = source_classification_loss - adversarial_curriculum_loss + diverse_curriculum_loss + \
            prototype_classification_loss + selfsupervised_clustering_loss
        """
        print("train_wo_plt")
        print("source_classification_loss:", source_classification_loss.item())
        print("adversarial_curriculum_loss:", adversarial_curriculum_loss.item())
        print("diverse_curriculum_loss:", diverse_curriculum_loss.item())
        print("prototype_classification_loss:", prototype_classification_loss.item())
        print("selfsupervised_clustering_loss:", selfsupervised_clustering_loss.item())
        """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return loss