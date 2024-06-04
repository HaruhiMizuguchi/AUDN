import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from net import FeatureExtractor, SourceClassifier, DomainDiscriminator, PrototypeClassifier
from CNTGE import run_CNTGE
from utils import calculate_transfer_score, get_source_weights, add_new_prototypes, predict

# ハイパーパラメータ (config.yaml から読み込む)
# 例:
# learning_rate = config['learning_rate']
# batch_size = config['batch_size']
# epochs = config['epochs']
# ...

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(feature_extractor, source_classifier, domain_discriminator, prototype_classifier,
                source_loader, target_loader, optimizer, alpha, w_0, criterion):
    """
    1エポック分の学習を行う関数

    Args:
        feature_extractor (nn.Module): 特徴抽出器
        source_classifier (nn.Module): ソースラベル分類器
        domain_discriminator (nn.Module): ドメイン分類器
        prototype_classifier (nn.Module): プロトタイプ分類器
        source_loader (DataLoader): ソースドメインのデータローダー
        target_loader (DataLoader): ターゲットドメインのデータローダー
        optimizer (optim.Optimizer): 最適化アルゴリズム
        alpha (float): 勾配反転層の重み係数
        w_0 (float): 転送スコアの閾値
        criterion (nn.Module): 損失関数 (クロスエントロピーなど)

    Returns:
        float: 1エポック分の平均損失
    """

    feature_extractor.train()
    source_classifier.train()
    domain_discriminator.train()
    prototype_classifier.train()

    total_loss = 0

    for (source_data, source_label), (target_data, _) in tqdm(zip(source_loader, target_loader), total=len(source_loader)):
        source_data, source_label, target_data = source_data.to(device), source_label.to(device), target_data.to(device)

        # 特徴抽出
        source_features = feature_extractor(source_data)
        target_features = feature_extractor(target_data)

        # --- ソースラベル分類器の学習 ---
        source_preds = source_classifier(source_features)
        source_classification_loss = criterion(source_preds, source_label)

        # --- ドメイン分類器の学習 ---
        source_domain_preds = domain_discriminator(source_features)
        target_domain_preds = domain_discriminator(target_features)
        domain_adversarial_loss = - torch.mean(torch.log(1 - source_domain_preds)) - torch.mean(torch.log(target_domain_preds))

        # --- 転送スコアと重みの計算 ---
        transfer_scores = calculate_transfer_score(target_features, source_classifier, domain_discriminator)
        source_weights = get_source_weights(source_features, source_classifier, w_0)

        # --- 敵対的カリキュラム学習 ---
        adversarial_curriculum_loss = torch.mean(source_weights * torch.log(1 - source_domain_preds)) + \
                                     torch.mean((transfer_scores >= w_0).float() * torch.log(target_domain_preds))

        # --- 多様性カリキュラム学習 ---
        diverse_curriculum_loss = - torch.mean((transfer_scores < w_0).float() * torch.sum(F.softmax(source_preds, dim=1) *
                                                                                        torch.log(F.softmax(source_preds, dim=1)),
                                                                                        dim=1))

        # --- プロトタイプ分類器の学習 ---
        prototype_preds = prototype_classifier(target_features)
        prototype_classification_loss = criterion(prototype_preds, target_label) # target_labelは疑似ラベルかアクティブラーニングで取得したラベル

        # --- 全体の損失 ---
        loss = source_classification_loss + domain_adversarial_loss + adversarial_curriculum_loss + \
               diverse_curriculum_loss + prototype_classification_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(source_loader)

def validate(feature_extractor, source_classifier, prototype_classifier, target_loader, w_0):
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
        for data, label in target_loader:
            data, label = data.to(device), label.to(device)
            features = feature_extractor(data)
            transfer_scores = calculate_transfer_score(features, source_classifier, domain_discriminator)
            preds = predict(features, source_classifier, prototype_classifier, transfer_scores, w_0)
            correct += (preds == label).sum().item()
            total += label.size(0)

    return correct / total

def main():
    # --- データローダーの準備 ---
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
    target_loader_train = DataLoader(target_dataset_train, batch_size=batch_size, shuffle=True)
    target_loader_val = DataLoader(target_dataset_val, batch_size=batch_size, shuffle=False)

    # --- モデルのインスタンス化 ---
    feature_extractor = FeatureExtractor().to(device)
    source_classifier = SourceClassifier(num_source_classes).to(device)
    domain_discriminator = DomainDiscriminator(alpha).to(device)
    prototype_classifier = PrototypeClassifier().to(device)

    # --- 最適化アルゴリズムの定義 ---
    optimizer = optim.SGD(list(feature_extractor.parameters()) +
                          list(source_classifier.parameters()) +
                          list(domain_discriminator.parameters()) +
                          list(prototype_classifier.parameters()),
                          lr=learning_rate, momentum=0.9)

    # --- 損失関数の定義 ---
    criterion = nn.CrossEntropyLoss()

    # --- 学習ループ ---
    best_acc = 0
    for epoch in range(epochs):
        # --- 学習 ---
        train_loss = train_epoch(feature_extractor, source_classifier, domain_discriminator, prototype_classifier,
                                source_loader, target_loader_train, optimizer, alpha, w_0, criterion)

        # --- 検証 ---
        val_acc = validate(feature_extractor, source_classifier, prototype_classifier, target_loader_val, w_0)

        # --- 結果の出力 ---
        print(f"Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")

        # --- ベストモデルの保存 ---
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(feature_extractor.state_dict(), "feature_extractor_best.pth")
            torch.save(source_classifier.state_dict(), "source_classifier_best.pth")
            torch.save(domain_discriminator.state_dict(), "domain_discriminator_best.pth")
            torch.save(prototype_classifier.state_dict(), "prototype_classifier_best.pth")

    print(f"Best Val Acc: {best_acc:.4f}")

if __name__ == "__main__":
    main()