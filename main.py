import torch.optim as optim
import CNTGE, train, train_wo_plt, train_warmup
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from data import create_dataset_dataloader
from utils import validate
from globals import *
from net import feature_extractor, source_classifier, domain_discriminator, prototype_classifier

# モデルのインスタンス化
feature_extractor = feature_extractor().to(device)
source_classifier = source_classifier(n_source_classes).to(device)
domain_discriminator = domain_discriminator(grl_alpha).to(device)
prototype_classifier = prototype_classifier(tau).to(device)

# 最適化アルゴリズム
optimizer = optim.SGD([
    {'params': feature_extractor.parameters()},
    {'params': source_classifier.parameters()},
    {'params': domain_discriminator.parameters()},
    {'params': prototype_classifier.parameters()}
], lr=lr, momentum=momentum, weight_decay=weight_decay)

# データセットとデータローダー
D_s, D_ut_train, D_t_test, D_s_loader, D_ut_train_loader, D_t_test_loader = \
    create_dataset_dataloader(dataset_name, source_domain, target_domain, batch_size, n_source_private, n_share, n_target_private)
D_lt = TensorDataset(torch.Tensor([]), torch.LongTensor([])) # ラベル付きターゲットデータ (初期状態は空)
D_plt = TensorDataset(torch.Tensor([]), torch.LongTensor([])) # 疑似ラベル付きターゲットデータ (初期状態は空)
labels_of_prototypes = []

# --- ソースデータのクラスバイアスを確認 ---
from collections import Counter
source_labels = [label for _, label in D_s]
source_label_counts = Counter(source_labels)
print("ソースデータのクラス分布:", source_label_counts)
# --- ここまで ---

D_s_label = [D_s[i][1] for i in range(len(D_s))]
print("D_s_label:",set(D_s_label))
D_t_test_label = [D_t_test[i][1] for i in range(len(D_t_test))]
print("D_t_test_label:",set(D_t_test_label))
D_ut_label = [D_ut_train[i][1] for i in range(len(D_ut_train))]
print("D_ut_label:",set(D_ut_label))

config.total_ite = len(D_s) / batch_size * (AL_round + 1)
print("total_ite:",config.total_ite)
# 学習
# --- Warm Up ---
loss = train_warmup.train_warmup_epoch(feature_extractor, source_classifier, domain_discriminator,
                D_s_loader, D_ut_train_loader, optimizer)
    
# --- 学習ループ ---

for round in range(AL_round):
#for round in range(1):
    print(f"Round {round+1}/{AL_round}")
    
    # --- CNTGE ---
    D_ut_train, D_lt, D_plt, D_ut_train_loader, D_lt_loader, D_plt_loader, labels_of_prototypes = \
        CNTGE.run_CNTGE(D_ut_train, D_lt, D_plt, feature_extractor, source_classifier, domain_discriminator, prototype_classifier, labels_of_prototypes, n_source_classes, k=n_r, n_r=n_r)
    print(set([D_lt[i][1].item() for i in range(len(D_lt))]))
    # --- 学習 ---
    loss = train.train_epoch(feature_extractor, source_classifier, domain_discriminator, prototype_classifier,
                        D_s_loader, D_ut_train_loader, D_lt_loader, D_plt_loader, optimizer)
    print(f"Loss: {loss:.4f}")

    # --- 検証 ---
    accuracy = validate(feature_extractor, source_classifier, domain_discriminator, prototype_classifier, D_t_test_loader, w_0)
    print(f"Accuracy: {accuracy:.4f}")

print(len(D_t_test_loader))
plt_labels = [D_plt[i][1].item() for i in range(len(D_plt))]
print(set(plt_labels))