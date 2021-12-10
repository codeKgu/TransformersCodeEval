import json
import os
import pathlib

def create_split(split="train", name="train"):
    paths = []
    roots = sorted(os.listdir(split))
    for folder in roots:
        root_path = os.path.join(split, folder)
        paths.append(root_path)


    with open(name+".json", "w") as f:
        json.dump(paths, f)
    
    return paths

# insert path to train and test
# path should be relative to root directory or absolute paths
def main(args):
    paths_to_probs = [args.train_path, args.test_path]
    names = ["train", "test"]

    all_paths = []
    for index in range(len(paths_to_probs)):
        all_paths.extend(create_split(split=paths_to_probs[index], name=names[index]))

    with open("train_and_test.json", "w") as f:
        print(f"Writing all paths. Length = {len(all_paths)}")
        json.dump(all_paths, f)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="get absolute paths to apps train and test files")
    parser.add_argument('--train_path', default='../dataset/train')
    parser.add_argument('--test_path', default='../dataset/test')
    parser.add_argument('--save_folder', default='.')
    args = parser.parse_args()
import json
import os
import pathlib

def create_split(split="train", name="train"):
    paths = []
    roots = sorted(os.listdir(split))
    for folder in roots:
        root_path = os.path.join(split, folder)
        paths.append(root_path)


    with open(name+".json", "w") as f:
        json.dump(paths, f)
    
    return paths

# insert path to train and test
# path should be relative to root directory or absolute paths
def main(args):
    paths_to_probs = [args.train_path, args.test_path]
    names = ["train", "test"]

    all_paths = []
    for index in range(len(paths_to_probs)):
        all_paths.extend(create_split(split=paths_to_probs[index], name=names[index]))

    with open("train_and_test.json", "w") as f:
        print(f"Writing all paths. Length = {len(all_paths)}")
        json.dump(all_paths, f)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="get absolute paths to apps train and test files")
    parser.add_argument('--train_path', default='../dataset/train')
    parser.add_argument('--test_path', default='../dataset/test')
    parser.add_argument('--save_folder', default='.')
    args = parser.parse_args()
