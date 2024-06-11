import os

def count_files_in_directory(root_dir):
    total_files = 0
    for root, dirs, files in os.walk(root_dir):
        total_files += len(files)
    return total_files

# 使用例
root_directory = "data/office/domain_adaptation_images/webcam"  # 調査するフォルダのパスに置き換えてください
file_count = count_files_in_directory(root_directory)
print(f"Total number of files: {file_count}")