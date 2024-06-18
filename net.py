import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from globals import *

class GradientReverseLayer(nn.Module):
    """
    勾配反転層 (GRL) を実装するクラス
    """
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return x * -self.alpha  # 勾配を反転させる処理

class feature_extractor(nn.Module):
    """
    特徴抽出器 (Gf) を実装するクラス
    """
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        #self.resnet = resnet50()
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # 最終層を除去

    def forward(self, x):
        # x = self.resnet(x)
        for layer in self.resnet:
            x = layer(x)
        x = x.view(x.size(0), -1)
        return x

class source_classifier(nn.Module):
    """
    ソースラベル分類器 (Gc) を実装するクラス
    """
    def __init__(self, num_source_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(2048, 256),  # ボトルネック層
            nn.ReLU(),
            nn.Linear(256, num_source_classes)  # 全結合層
        )

    def forward(self, x):
        logits = self.classifier(x)
        if logits.dim() == 1:
            # Single data point (1D tensor)
            return torch.softmax(logits, dim=0)
        else:
            # Batch of data points (2D tensor)
            return torch.softmax(logits, dim=1)

class domain_discriminator(nn.Module):
    """
    ドメイン分類器 (Gd) を実装するクラス
    """
    def __init__(self, alpha):
        super().__init__()
        self.grl = GradientReverseLayer(alpha)  # GRLをインスタンス化
        self.discriminator = nn.Sequential(
            nn.Linear(2048, 256),  # ボトルネック層
            nn.ReLU(),
            self.grl,  # GRLのインスタンスを使用
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminator(x)


class prototype_classifier(nn.Module):
    """
    プロトタイプ分類器 (Gp) を実装するクラス
    """
    def __init__(self, tau=0.05):
        super().__init__()
        self.bottleneck = nn.Linear(2048, 256)  # ボトルネック層
        self.prototypes = nn.ParameterList()  # 動的にプロトタイプを追加するためのリスト
        self.prototype_labels = []  # プロトタイプに対応するクラスラベル
        self.tau = tau  # 温度パラメータτ

    def forward(self, x, inference_mode=False):
        x = F.relu(self.bottleneck(x))
        if inference_mode:
            # 推論モードではn_source_classes以上のラベルを持つプロトタイプのみ使用
            active_prototypes = torch.stack([proto for proto, label in zip(self.prototypes, self.prototype_labels) if label >= n_source_classes])
            active_labels = [label for label in self.prototype_labels if label >= n_source_classes]
        else:
            active_prototypes = torch.stack(self.prototypes)
            active_labels = self.prototype_labels

        similarities = F.cosine_similarity(x.unsqueeze(1), active_prototypes.unsqueeze(0), dim=2)  # コサイン類似度計算
        outputs = F.softmax(similarities / self.tau, dim=1)
        
        # 全てのクラス数分のベクトルを出力（無効なクラスには0を設定）
        full_outputs = torch.zeros(x.size(0), self.n_total_classes, device=x.device)
        for i, label in enumerate(active_labels):
            full_outputs[:, label] = outputs[:, i]

        return full_outputs

    def add_prototype(self, new_prototype, class_index):
        """
        新しいプロトタイプを追加するメソッド
        """
        new_prototype = F.relu(self.bottleneck(new_prototype.unsqueeze(0))).squeeze(0)
        self.prototypes.append(nn.Parameter(new_prototype))
        self.prototype_labels.append(class_index)