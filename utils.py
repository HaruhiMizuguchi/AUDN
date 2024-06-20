import numpy as np
import torch
from torch.nn import functional as F
from globals import *

def calculate_transfer_score(x, source_classifier, domain_discriminator):
    return source_classifier(x).max() + domain_discriminator(x)

def calculate_transfer_score_debug(x, source_classifier, domain_discriminator):
    return source_classifier(x).max() + domain_discriminator(x), source_classifier(x).max(), domain_discriminator(x)

def get_source_weights(ut_features, s_label, source_classifier, domain_discriminator, ut_preds):
    ws = torch.stack([calculate_transfer_score(x, source_classifier, domain_discriminator) for x in ut_features])
    if torch.isnan(ws).any():
        print("NaN found in ws in get source weights")
        
    # Debugging for w_alpha
    valid_ws = ws.reshape(-1, 1).flatten() >= config.w_alpha
    print(f"sum(valid_ws): {torch.sum(valid_ws)}")
    
    if torch.sum(valid_ws) == 0:
        # Set V to all ones if no valid ws is found
        V = torch.ones_like(ut_preds[0])
    else:
        V = torch.sum(ut_preds[valid_ws], axis=0) / torch.sum(valid_ws).item()

    weights = torch.stack([V[label_idx] for label_idx in s_label])

    return weights

def predict(feature, source_classifier, domain_discriminator, prototype_classifier, w_0):
    w = calculate_transfer_score(feature, source_classifier, domain_discriminator)
    if w > w_0:
        return torch.argmax(source_classifier(feature))
    else:
        return torch.argmax(prototype_classifier(feature,inference_mode=True))
    
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
    transferrable = 0
    ws = []
    with torch.no_grad():
        for data, label in target_test_loader:
            data, label = data.to(device), label.to(device)
            feature = feature_extractor(data)
            preds = predict(feature, source_classifier, domain_discriminator, prototype_classifier, w_0)
            if preds.item() == label.item():
                correct += 1
            total += 1
            if total < 5:
                w, s, d = calculate_transfer_score_debug(feature, source_classifier, domain_discriminator)
                ws.append([w.item(),s.item(),d.item()])
            if calculate_transfer_score(feature, source_classifier, domain_discriminator) > w_0:
                transferrable += 1
    
    print("num transferrable = ",transferrable)
    print("ws = ",ws)
    return correct / total