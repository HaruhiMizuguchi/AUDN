import torch.optim as optim
import CNTGE, train, train_warmup
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from data import create_dataset_dataloader
from globals import *
from net import feature_extractor, source_classifier, domain_discriminator, prototype_classifier

# データセットとデータローダー
D_s, D_ut_train, D_t_test, D_s_loader, D_ut_train_loader, D_t_test_loader = \
    create_dataset_dataloader(dataset_name, source_domain, target_domain, batch_size, n_source_private, n_share, n_target_private)
    
# モデルのインスタンス化
feature_extractor = feature_extractor().to(device)
source_classifier = source_classifier(n_source_classes).to(device)
domain_discriminator = domain_discriminator(alpha).to(device)
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

steps_per_epoch = len(D_s_loader)
T = AL_round * steps_per_epoch

# 学習
# --- Warm Up ---
#loss = train_warmup.train_warmup_epoch(feature_extractor, source_classifier, domain_discriminator,
#                D_s_loader, D_ut_train_loader, optimizer)

# --- 学習ループ ---
for round in range(AL_round):
    print(f"Round {round+1}/{AL_round}")
    
    # --- CNTGE ---
    D_ut_train, D_lt, D_plt, D_ut_train_loader, D_lt_loader, D_plt_loader = \
        CNTGE.run_CNTGE(D_ut_train, D_lt, D_plt, feature_extractor, source_classifier, domain_discriminator, n_r, n_r)
    
    # --- 敵対的・多様性カリキュラム学習とプロトタイプ分類器学習 ---
    """for epoch in range(steps_per_epoch):
        loss = train.train_epoch(feature_extractor, source_classifier, domain_discriminator, prototype_classifier,
                        D_s_loader, D_ut_train_loader, D_lt_loader, D_plt_loader, optimizer)
        print(f"Epoch {epoch+1}/{steps_per_epoch}, Loss: {loss:.4f}")
        t += 1
        w_alpha = w_0 + (1 - max(t, min_step)/AL_round*steps_per_epoch) * alpha"""

    # --- 検証 ---
    accuracy = train.validate(feature_extractor, source_classifier, domain_discriminator, prototype_classifier, D_t_test_loader, w_0)
    print(f"Accuracy: {accuracy:.4f}")