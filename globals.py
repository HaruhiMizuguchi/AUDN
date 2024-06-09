import yaml
import torch

# YAMLファイルを読み込む関数
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

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
n_source_classes = n_source_private + n_share
n_r = config['data']['dataset']['n_r']

# パラメータを取得
alpha = config['data']['train']['alpha']
beta = config['data']['train']['beta']
tau = config['data']['train']['tau']
w_0 = config['data']['train']['w_0']
lr = config['data']['train']['lr']
weight_decay = config['data']['train']['weight_decay']
momentum = config['data']['train']['momentum']
AL_round = config['data']['train']['AL_round']
w_alpha = w_0
t = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")