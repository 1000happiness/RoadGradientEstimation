import json
from os import path
import sys
import random
import os

def create_svo_split(file_root_path, file_path):
    json_path = os.path.join(file_root_path, file_path, "config.json")
    with open(json_path) as f:
        camera_config = json.load(f)

    frame_num = camera_config["frame_num"]

    index_list = list(range(frame_num))
    # random.shuffle(index_list)
    train_list = index_list[0: frame_num // 10 * 9][1:]
    test_list = index_list[frame_num // 10 * 9:]

    train_list = [file_path + " " + str(i) + " l" for i in train_list] + [file_path + " " + str(i) + " r" for i in train_list]
    test_list = [file_path + " " + str(i) + " l" for i in test_list] + [file_path + " " + str(i) + " r" for i in test_list]

    random.shuffle(train_list)
    random.shuffle(test_list)

    train_content = '\n'.join(train_list)
    test_content = '\n'.join(test_list)

    train_path = os.path.join(file_root_path, file_path, "train_files.txt")
    test_path = os.path.join(file_root_path, file_path, "val_files.txt")

    with open(train_path, 'w') as f:
        f.write(train_content)

    with open(test_path, 'w') as f:
        f.write(test_content)

if __name__ == "__main__":
    file_root_path = sys.argv[1]
    file_path = sys.argv[2]
    create_svo_split(file_root_path, file_path)