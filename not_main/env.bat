conda create --name AUDN
conda activate AUDN
conda install cudatoolkit=11.8 -y
conda install git -y
conda install pip -y
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tqdm
pip install scikit-learn
pip install pyyaml