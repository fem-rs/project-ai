import os
import glob

from pathlib import Path

def load_data(data_directory):
    human_labels, labels, data_subsets, data_files = [], [], [], []
    
    for child in data_directory.iterdir():
        data_subsets.append(f'{os.getcwd()!s}\\{str(child)}')
        human_labels.append(str(child).removeprefix('Data\\'))

    for file_path in data_subsets:
        for file in glob.glob(f'{file_path!s}\\*.jpg'):
            data_files.append(file)
            labels.append(human_labels.index(str(file_path).removeprefix(f'{os.getcwd()!s}\\Data\\')))
    
    return (human_labels, labels, data_subsets, data_files) 

def train_ai():
    data_dir = Path(f'{os.getcwd()!s}\\Data')
    
    human_labels, labels, data_subsets, data_files = load_data(data_dir)
    
    print(human_labels)    