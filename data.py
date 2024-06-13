import yaml
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os

def check_nan(dataloader):
    nan_found = False
    for images, labels in dataloader:
        if torch.isnan(images).any():
            nan_found = True
            print("NaN found in images:")
            nan_indices = torch.isnan(images).any(dim=[1, 2, 3])
            print(f"Images: {images[nan_indices]}")
            print(f"Labels: {labels[nan_indices]}")
    if not nan_found:
        print("No NaN found in images")

class DomainAdaptationDataset(Dataset):
    def __init__(self, data_file, domain, label_range):
        self.data = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        with open(data_file, 'r') as f:
            for line in f:
                path, label = line.strip().split('\t')
                label = int(label)
                if domain in path and label in label_range:
                    self.data.append((path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        return image, label

# YAMLファイルを読み込む関数
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_dataset_dataloader(dataset_name, source_domain, target_domain, batch_size, n_source_private, n_share, n_target_private):
    # どのデータセットを使うか決めるyamlを読み込む
    dataset_to_use_path = 'dataset_to_use.yaml'
    dataset_name = load_yaml(dataset_to_use_path)['dataset']
    
    # YAMLファイルを読み込む
    config_path = f'config/{dataset_name}.yaml'
    config = load_yaml(config_path)

    # ソースドメインとターゲットドメインを取得
    source_domain = config['data']['dataset']['souce']
    target_domain = config['data']['dataset']['target']
    batch_size = config['data']['dataloader']['batch_size']

    # ラベルセットを定義
    n_source_private = config['data']['dataset']['n_source_private']
    n_share = config['data']['dataset']['n_share']
    n_target_private = config['data']['dataset']['n_target_private']

    source_private_labels = set(range(n_source_private))
    shared_labels = set(range(n_source_private, n_source_private + n_share))
    target_private_labels = set(range(n_source_private + n_share, n_source_private + n_share + n_target_private))

    # データセットを作成
    D_s = DomainAdaptationDataset(f'data/{dataset_name}/images_and_labels.txt', source_domain, source_private_labels.union(shared_labels))
    print("len D_s", len(D_s))
    D_t = DomainAdaptationDataset(f'data/{dataset_name}/images_and_labels.txt', target_domain, target_private_labels.union(shared_labels))

    # データを8:2の割合で訓練データとテストデータに分割
    target_train_size = int(0.8 * len(D_t))
    target_test_size = len(D_t) - target_train_size
    D_ut_train, D_t_test = random_split(D_t, [target_train_size, target_test_size])
    print("len D_ut_train", len(D_ut_train))
    print("len D_t_test", len(D_t_test))

    # データローダを作成
    D_s_loader = DataLoader(D_s, batch_size=batch_size, shuffle=True)
    D_ut_train_loader = DataLoader(D_ut_train, batch_size=batch_size, shuffle=True)
    D_t_test_loader = DataLoader(D_t_test, batch_size=batch_size, shuffle=False)
    
    # NaNのチェック
    """print("Checking source domain data:")
    check_nan(D_s_loader)

    print("Checking target domain training data:")
    check_nan(D_ut_train_loader)

    print("Checking target domain test data:")
    check_nan(D_t_test_loader)
    """
    return D_s, D_ut_train, D_t_test, D_s_loader, D_ut_train_loader, D_t_test_loader