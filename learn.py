import cv2
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset

class GeometricTasksDataset(Dataset):

    def __init__(self, path):
        inputs_path = f"{path}/inputs"
        targets_path = f"{path}/targets"
        input_paths = [str(p) for p in sorted(Path(inputs_path).glob("*.png"))]
        target_paths = [str(p) for p in sorted(Path(targets_path).glob("*.png"))]
        examples = list(zip(input_paths, target_paths))
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        input_path, target_path = self.examples[index]
        input_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        target_image = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        input_tensor = (torch.from_numpy(input_image) // 255).float().unsqueeze(0)
        target_tensor = torch.from_numpy(target_image) // 255
        print(input_tensor)
        print(target_tensor)

dataset = GeometricTasksDataset("datasets/dataset1/val")

dataset.__getitem__(random.randint(0, 9))
