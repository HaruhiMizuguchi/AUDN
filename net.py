import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

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
        x = self.resnet(x)
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
        return self.classifier(x)

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
        self.prototypes = nn.Parameter(torch.Tensor(0, 256))  # プロトタイプ (初期状態ではユニット数0)
        self.tau = tau  # 温度パラメータτ

    def forward(self, x):
        x = F.relu(self.bottleneck(x))
        similarities = F.cosine_similarity(x.unsqueeze(1), self.prototypes, dim=2)  # コサイン類似度計算
        outputs = F.softmax(similarities / self.tau, dim=1)  # softmax
        return outputs

    def add_prototype(self, new_prototype):
        """
        新しいプロトタイプを追加するメソッド
        """
        self.prototypes = nn.Parameter(torch.cat([self.prototypes, new_prototype.unsqueeze(0)], dim=0))
