import cv2
import math
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms.functional import to_pil_image

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
        target_tensor = (torch.from_numpy(target_image) // 255).float().unsqueeze(0)
        return input_tensor, target_tensor

class Unet(nn.Module):

    def __init__(self, image_size, dropout_p=0):
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
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        intermediate_results = []
        for convolution in self.convolutions:
            x = convolution(x)
            x = nn.functional.relu(x)
            x = self.dropout(x)
            intermediate_results.append(x)
        intermediate_results = intermediate_results[:-1]
        for transposed_convolution, intermediate_result in zip(self.transposed_convolutions[:-1], reversed(intermediate_results)):
            x = transposed_convolution(x)
            x = nn.functional.relu(x)
            x = self.dropout(x)
            x = torch.cat((x, intermediate_result), dim=1)
        x = self.transposed_convolutions[-1](x)
        return x

def logits_to_colors(outputs):
    return (outputs > 0).float()

def get_nb_correct_pixels_in_batch(targets, outputs):
    outputs = logits_to_colors(outputs)
    targets = targets.int()
    outputs = outputs.int()
    return (targets == outputs).sum().item()

def get_nb_pixels_in_dataset(dataloader):
    dataset = dataloader.dataset
    nb_images = len(dataset)
    input_tensor, target_tensor = dataset[0]
    nb_pixels_per_image = input_tensor.numel()
    return nb_images * nb_pixels_per_image

def get_batch_intersection_tensor(targets, outputs):
    outputs = logits_to_colors(outputs)
    targets = targets.bool()
    outputs = outputs.bool()
    return (targets & outputs).float()

def get_batch_union_tensor(targets, outputs):
    outputs = logits_to_colors(outputs)
    targets = targets.bool()
    outputs = outputs.bool()
    return (targets | outputs).float()

def get_batch_xor_tensor(targets, outputs):
    outputs = logits_to_colors(outputs)
    targets = targets.bool()
    outputs = outputs.bool()
    return (targets ^ outputs).float()

def get_batch_intersection(targets, outputs):
    return get_batch_intersection_tensor(targets, outputs).sum().item()

def get_batch_union(targets, outputs):
    return get_batch_union_tensor(targets, outputs).sum().item()

def train_the_model(dataloader, model, loss_function, optimizer, device):
    print("training")
    model.train()
    total_loss = 0
    nb_correct_pixels = 0
    total_intersection = 0
    total_union = 0
    for i, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        nb_correct_pixels += get_nb_correct_pixels_in_batch(targets, outputs)
        total_intersection += get_batch_intersection(targets, outputs)
        total_union += get_batch_union(targets, outputs)
        print(".", end="", flush=True)
    average_loss = total_loss / len(dataloader)
    accuracy = nb_correct_pixels / get_nb_pixels_in_dataset(dataloader)
    IoU = total_intersection / total_union
    print()
    return average_loss, accuracy, IoU

def evaluate_the_model(dataloader, model, loss_function, device):
    print("evaluating")
    model.eval()
    total_loss = 0
    nb_correct_pixels = 0
    total_intersection = 0
    total_union = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            total_loss += loss.item()
            nb_correct_pixels += get_nb_correct_pixels_in_batch(targets, outputs)
            total_intersection += get_batch_intersection(targets, outputs)
            total_union += get_batch_union(targets, outputs)
            print(".", end="", flush=True)
    average_loss = total_loss / len(dataloader)
    accuracy = nb_correct_pixels / get_nb_pixels_in_dataset(dataloader)
    IoU = total_intersection / total_union
    print()
    return average_loss, accuracy, IoU

def display(inputs, targets, outputs, file_name):
    outputs = outputs.detach()
    intersection = get_batch_intersection_tensor(targets, outputs)
    xor = get_batch_xor_tensor(targets, outputs)
    green = torch.cat([torch.zeros_like(inputs), intersection, torch.zeros_like(inputs)], dim=1)
    red = torch.cat([xor, torch.zeros_like(inputs), torch.zeros_like(inputs)], dim=1)
    overlay = green + red
    overlay = torch.where((overlay == 0).all(dim=1, keepdim=True), 1, overlay)
    outputs = logits_to_colors(outputs)
    inputs = inputs.expand(-1, 3, -1, -1)
    targets = targets.expand(-1, 3, -1, -1)
    outputs = outputs.expand(-1, 3, -1, -1)
    inputs = 1 - inputs
    targets = 1 - targets
    outputs = 1 - outputs
    stack = torch.stack([inputs, targets, outputs, overlay], dim=1)
    merged = stack.flatten(start_dim=0, end_dim=1)
    grid = torchvision.utils.make_grid(merged, nrow=4)
    pil_image = to_pil_image(grid)
    pil_image.save(file_name)
