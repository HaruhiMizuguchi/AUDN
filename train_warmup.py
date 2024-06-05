import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from net import feature_extractor, source_classifier, domain_discriminator
from CNTGE import run_CNTGE
from utils import calculate_transfer_score, predict
from globals import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_source_weights(ut_features, s_label, source_classifier, domain_discriminator, ut_preds):
    ws = np.array([calculate_transfer_score(x, source_classifier, domain_discriminator) for x in ut_features])
    V = np.sum(ut_preds[ws >= w_alpha], axis=0) / np.sum(ws >= w_alpha)
    weights = np.array([V[label_idx] for label_idx in s_label])
    return weights

def train_warmup_epoch(feature_extractor, source_classifier, domain_discriminator,
                D_s_loader, D_ut_train_loader, optimizer):
    """
    1エポック分の学習を行う関数

    Args:
        feature_extractor (nn.Module): 特徴抽出器
        source_classifier (nn.Module): ソースラベル分類器
        domain_discriminator (nn.Module): ドメイン分類器
        D_s_loader (DataLoader): ソースドメインのデータローダー
        D_ut_train_loader (DataLoader): ターゲットドメインのラベルなしデータのデータローダー
        optimizer (optim.Optimizer): 最適化アルゴリズム
        alpha (float): 勾配反転層の重み係数
        w_0 (float): 転送スコアの閾値

    Returns:
        float: 1エポック分の平均損失
    """
    feature_extractor.train()
    source_classifier.train()
    domain_discriminator.train()
    
    total_loss = 0
    
    for (s_data, s_label), (ut_data, _) in tqdm(zip(D_s_loader, D_ut_train_loader)):
        s_data, s_label, ut_data = s_data.to(device), s_label.to(device), ut_data.to(device)
        
        # 特徴抽出
        s_features = feature_extractor(s_data)
        ut_features = feature_extractor(ut_data)
        
        # --- ソースラベル分類器の学習 L_C ---
        s_preds = source_classifier(s_features)
        ut_preds = source_classifier(ut_features)

        # 共通ラベルのデータのみで損失を計算
        source_classification_loss = F.cross_entropy(s_preds, s_label)
        
        # --- ドメイン分類器の学習 ---
        s_domain_preds = domain_discriminator(s_features)
        ut_domain_preds = domain_discriminator(ut_features)
        
        # --- 転送スコアと重みの計算 ---
        ut_transfer_scores = np.array([calculate_transfer_score(ut_feature, source_classifier, domain_discriminator) for ut_feature in ut_features])
        source_weights = get_source_weights(ut_features, s_label, source_classifier, domain_discriminator, ut_preds)
        
        # --- 敵対的カリキュラム学習 L_adv ---
        adversarial_curriculum_loss = torch.mean(source_weights * torch.log(1 - s_domain_preds)) + \
                                        torch.mean((ut_transfer_scores >= w_alpha).float() * torch.log(ut_domain_preds))
                                        
        # --- 多様性カリキュラム学習 L_div ---
        diverse_curriculum_loss = - torch.mean((ut_transfer_scores < w_alpha).float() * (torch.sum(F.softmax(ut_preds, dim=1) *
                                                                                        torch.log(F.softmax(ut_preds, dim=1)), dim=1)))
        
        # --- 全体の損失 ---
        loss = source_classification_loss - adversarial_curriculum_loss + diverse_curriculum_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return total_loss / len(D_s_loader)