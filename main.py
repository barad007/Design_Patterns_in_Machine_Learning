import numpy as np
from train import Trainer
from utils import save_results
from transform import ValidationTransforms, TrainTransforms

from DatasetFactory import get_dataset, FolderDatasetCreator, CustomImageDatasetCreator, InferenceDatasetCreator
from Memento import Originator, Caretaker
from adapter import use_adapter, Adapter, ImageFileReader

import torch
from torch import nn
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
import torchvision

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == "__main__":

    train_path = "./data/train/"
    validation_data = "./data/file_list.json"
    path_test_files = "data/test/"
    test_results = "test_results.json"

    number_of_classes = 2
    lr = 0.01
    epochs = 1
    batch_size = 32
    num_workers = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = TrainTransforms()
    valid_transform = ValidationTransforms()

    # Create a dataset using Factory Method (Creational Design Patterns)
    train_dataset = get_dataset(FolderDatasetCreator(data=train_path, transform=train_transform))
    print(f"train_dataset: {len(train_dataset)} images.")

    # Create a dataset using Factory Method (Creational Design Patterns)
    validation_dataset = get_dataset(CustomImageDatasetCreator(data=validation_data, transform=valid_transform))
    print(f"validation_dataset: {len(validation_dataset)} images.")

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(validation_dataset)

    data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    data_loader_val = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = torchvision.models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V1", progress=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, number_of_classes)
    nn.init.xavier_uniform_(model.fc.weight)
    model.to(device)

    # Memento (Behavioral Design Pattern)
    originator = Originator(model)
    caretaker = Caretaker(originator)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(params, lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)
    trainer = Trainer(model)

    best_f1 = 0
    for epoch in (range(epochs)):
        train_results = trainer.train_epoch(dataloader=data_loader, criterion=criterion, optimizer=optimizer,
                                            device=device, epoch=epoch)
        scheduler.step()

        validation_results = trainer.validate(dataloader=data_loader, criterion=criterion, device=device)
        val_loss, val_acc, f1, f1_macro, report, report_p, cm = validation_results

        # save results
        if f1 > best_f1:
            # saving the best model using Memento (Behavioral Design Pattern)
            caretaker.save_state()
            print(f"Improvement at epoch {epoch}, f1 was improved from {best_f1} to {f1}.\n")
            best_f1 = f1

    # restoring the best model using Memento (Behavioral Design Pattern)
    caretaker.restore_state()

    model.eval().to(device)

    # Adapter (Structural Design Pattern)
    adapter = Adapter(ImageFileReader(path_test_files))
    json_samples = use_adapter(adapter)

    inference_dataset = get_dataset(InferenceDatasetCreator(data=json_samples, transform=valid_transform))
    print(f"inference_dataset: {len(inference_dataset)} images.")

    inference_dataloader = DataLoader(
        inference_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    id_list, path_list, pvals, pcls = [], [], [], []
    with torch.no_grad():
        for i, sample in enumerate(inference_dataloader):
            img = sample['image'].to(device)
            idx = sample['idx']
            path = sample['img_path']

            pred_logits = model(img)
            output = torch.nn.functional.softmax(input=pred_logits, dim=1)
            softmax_vals, predicted_cls = output.max(1)

            p_cls = predicted_cls.detach().cpu().numpy()
            p_vals = softmax_vals.detach().cpu().numpy()
            idx = idx.detach().cpu().numpy()

            id_list.append(idx)
            path_list.append(path)
            pcls.append(p_cls)
            pvals.append(p_vals)

    id_list = np.concatenate(np.asarray(id_list, dtype=object)).tolist()
    path_list = np.concatenate(np.asarray(path_list, dtype=object)).tolist()
    pvals = np.concatenate(np.asarray(pvals, dtype=object)).tolist()
    pvals = [round(i, 3) if i is not None else None for i in pvals]
    pcls = np.concatenate(np.asarray(pcls, dtype=object)).tolist()


    save_results(test_results, id_list, path_list, pvals, pcls)
