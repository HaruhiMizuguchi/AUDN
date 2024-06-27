import math
import numpy as np
import torch
import itertools
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from net import feature_extractor, source_classifier, domain_discriminator
from CNTGE import run_CNTGE
from utils import *
from globals import *

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
    
    print("total_ite:",config.total_ite)
    print("t:",config.t)
    print("w_alpha:",config.w_alpha)
    
    feature_extractor.train()
    source_classifier.train()
    domain_discriminator.train()
    
    # 各データローダーのイテレーターを作成
    s_iter = iter(D_s_loader)
    ut_iter = iter(D_ut_train_loader)
    
    total_loss = 0
    
    # Vの計算
    all_ut_features = []
    """for data, label in D_ut_train_loader:
        with torch.no_grad():
            features = feature_extractor(data.to(device))
        all_ut_features.append(features)"""
    with torch.no_grad():
        for data, label in D_ut_train_loader:
            all_ut_features.append(feature_extractor(data.to(device)))
        all_ut_features = torch.cat(all_ut_features, dim=0)
        ut_preds = source_classifier(all_ut_features)
    V = calculate_V(all_ut_features, source_classifier, domain_discriminator, ut_preds)
    
    for (s_data, s_label), (ut_data, _) in tqdm(itertools.zip_longest(s_iter, ut_iter, fillvalue=(None, None))):
        
        source_classification_loss = torch.tensor(0.0, requires_grad=True)
        adversarial_curriculum_loss = torch.tensor(0.0, requires_grad=True)
        diverse_curriculum_loss = torch.tensor(0.0, requires_grad=True)
        
        config.t += 1
        if config.t <= config.total_ite:
            config.w_alpha = w_0 - ((1 - config.t/config.total_ite) * alpha)
        
        if s_data is not None:
            s_data, s_label = s_data.to(device), s_label.to(device)
            s_features = feature_extractor(s_data)
            s_preds = source_classifier(s_features)
            s_domain_preds = domain_discriminator(s_features)
            source_weights = get_source_weights(s_label, V)
            # --- ソースラベル分類器の学習 L_C ---
            source_classification_loss = source_classification_loss + F.cross_entropy(s_preds, s_label)
            # --- 敵対的カリキュラム学習 L_adv ---
            adversarial_curriculum_loss = adversarial_curriculum_loss + torch.mean(source_weights * torch.log(torch.clamp(1 - s_domain_preds, min=eps)))
            
        if ut_data is not None:
            ut_data = ut_data.to(device)
            ut_features = feature_extractor(ut_data)
            ut_preds = source_classifier(ut_features)
            ut_domain_preds = domain_discriminator(ut_features)
            ut_transfer_scores = torch.stack([calculate_transfer_score(ut_feature, source_classifier, domain_discriminator) for ut_feature in ut_features])
            # --- 敵対的カリキュラム学習 L_adv ---
            adversarial_curriculum_loss = adversarial_curriculum_loss + torch.mean((ut_transfer_scores >= config.w_alpha).float() * torch.log(torch.clamp(ut_domain_preds, min=eps)))
            # --- 多様性カリキュラム学習 L_div ---
            diverse_curriculum_loss = diverse_curriculum_loss - torch.mean((ut_transfer_scores < config.w_alpha).float() * (torch.sum(ut_preds * torch.log(ut_preds), dim=1)))
        
        # --- 全体の損失 ---
        #loss = source_classification_loss - adversarial_curriculum_loss + diverse_curriculum_loss
        loss = source_classification_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return loss