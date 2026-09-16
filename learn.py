from learning import display
from learning import evaluate_the_model
from learning import GeometricTasksDataset
from learning import logits_to_colors
from learning import train_the_model
from learning import Unet
import os
from pathlib import Path
import random
import torch
from torch import nn
from torch.utils.data import DataLoader

BATCH_SIZE = 64
DATASET_PATH = "datasets/dataset1"
IMAGE_SIZE = 512
NB_EPOCHS = 10
RESULT_PATH = "results"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

dataset = GeometricTasksDataset("datasets/dataset1/val")

train_dataset = GeometricTasksDataset(f"{DATASET_PATH}/train")
val_dataset = GeometricTasksDataset(f"{DATASET_PATH}/val")
test_dataset = GeometricTasksDataset(f"{DATASET_PATH}/test")

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

unet = Unet(IMAGE_SIZE)
print(unet)
print(f"number of parameters: {sum(p.numel() for p in unet.parameters() if p.requires_grad)}")
unet.to(device)

inputs, targets = next(iter(train_dataloader))
inputs = inputs.to(device)
targets = targets.to(device)

outputs = unet(inputs)
print(outputs.shape)

print("outputs (logits)")
print(outputs[0])

display(inputs, targets, outputs, "display.png")

loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(unet.parameters(), lr=0.001)

#targets_for_loss = targets.float()
#outputs_for_loss = outputs.squeeze(1)
#test_loss = loss_function(outputs_for_loss, targets_for_loss)
#print(test_loss)

if not os.path.exists(RESULT_PATH):
    Path(RESULT_PATH).mkdir()

for i in range(NB_EPOCHS):
    print(f"EPOCH {i + 1}")
    training_loss = train_the_model(train_dataloader, unet, loss_function, optimizer, device)
    validation_loss = evaluate_the_model(val_dataloader, unet, loss_function, device)
    print(f"TRAINING LOSS = {training_loss}")
    print(f"VALIDATION LOSS = {validation_loss}")
    inputs, targets = next(iter(val_dataloader))
    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = unet(inputs)
    display(inputs, targets, outputs, f"{RESULT_PATH}/{i + 1}.png")
    print()
