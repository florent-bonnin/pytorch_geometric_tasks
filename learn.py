import cv2
import math
from pathlib import Path
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

BATCH_SIZE = 64
DATASET_PATH = "datasets/dataset1"
IMAGE_SIZE = 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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
        return input_tensor, target_tensor

dataset = GeometricTasksDataset("datasets/dataset1/val")

train_dataset = GeometricTasksDataset(f"{DATASET_PATH}/train")
val_dataset = GeometricTasksDataset(f"{DATASET_PATH}/val")
test_dataset = GeometricTasksDataset(f"{DATASET_PATH}/test")

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class Unet(nn.Module):

    def __init__(self, image_size):
        super().__init__()
        self.convolutions = nn.ModuleList()
        in_channels = 1
        while image_size > 1:
            out_channels = 16 if in_channels == 1 else in_channels * 2
            convolution = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
            self.convolutions.append(convolution)
            image_size = math.ceil(image_size / 2)
            in_channels = out_channels
        self.transposed_convolutions = nn.ModuleList()
        while in_channels > 1:
            out_channels = 1 if in_channels == 16 else in_channels // 2
            if len(self.transposed_convolutions) != 0:
                # because of skip connection
                in_channels *= 2
            transposed_convolution = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1)
            self.transposed_convolutions.append(transposed_convolution)
            in_channels = out_channels

    def forward(self, x):
        intermediate_results = []
        for convolution in self.convolutions:
            x = convolution(x)
            x = nn.functional.relu(x)
            intermediate_results.append(x)
        intermediate_results = intermediate_results[:-1]
        for transposed_convolution, intermediate_result in zip(self.transposed_convolutions[:-1], reversed(intermediate_results)):
            x = transposed_convolution(x)
            x = nn.functional.relu(x)
            x = torch.cat((x, intermediate_result), dim=1)
        x = self.transposed_convolutions[-1](x)
        return x

unet = Unet(IMAGE_SIZE)
print(unet)
unet.to(device)

inputs, targets = next(iter(train_dataloader))
inputs = inputs.to(device)
targets = targets.to(device)

output = unet(inputs)
print(output.shape)

print(output[0])
