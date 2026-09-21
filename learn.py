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
DATASET_PATH = "datasets/draw_delaunay_triangulation"
DROPOUT = 0
IMAGE_SIZE = 512
NB_EPOCHS = 200
RESULT_PATH = "display"
WEIGHT_DECAY = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device}\n")

dataset = GeometricTasksDataset("datasets/dataset1/val")

train_dataset = GeometricTasksDataset(f"{DATASET_PATH}/train")
val_dataset = GeometricTasksDataset(f"{DATASET_PATH}/val")
test_dataset = GeometricTasksDataset(f"{DATASET_PATH}/test")

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

unet = Unet(IMAGE_SIZE, DROPOUT)
print(f"{unet}\n")
print(f"number of parameters: {sum(p.numel() for p in unet.parameters() if p.requires_grad)}\n")
unet.to(device)

loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(unet.parameters(), lr=0.001, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

if not os.path.exists(RESULT_PATH):
    Path(RESULT_PATH).mkdir()

best_epoch = 0
best_validation_IoU = 0

for i in range(NB_EPOCHS):
    print(f"epoch {i + 1}")
    print(f"learning rate = {optimizer.param_groups[0]["lr"]}")
    training_loss, training_accuracy, training_IoU = train_the_model(train_dataloader, unet, loss_function, optimizer, device)
    validation_loss, validation_accuracy, validation_IoU = evaluate_the_model(val_dataloader, unet, loss_function, device)
    scheduler.step(validation_loss)
    print(f"training loss = {training_loss}")
    print(f"validation loss = {validation_loss}")
    print(f"training accuracy = {training_accuracy}")
    print(f"validation accuracy = {validation_accuracy}")
    print(f"training IoU = {training_IoU}")
    print(f"validation IoU = {validation_IoU}")
    if validation_IoU > best_validation_IoU:
        best_epoch = i + 1
        best_validation_IoU = validation_IoU
    print(f"best epoch = {best_epoch}")
    print(f"best validation IoU = {best_validation_IoU}")
    print()
    inputs, targets = next(iter(val_dataloader))
    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = unet(inputs)
    display(inputs, targets, outputs, f"{RESULT_PATH}/{i + 1}.png")
