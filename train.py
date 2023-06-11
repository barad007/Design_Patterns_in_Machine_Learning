import torch
from typing import Tuple
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from decorator import decorator_classification_report, decorator_training_results


cls = dict(cat=0, dog=1)


class Trainer:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    @decorator_training_results
    def train_epoch(self, dataloader: torch.utils.data.DataLoader,
                    criterion: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int) -> Tuple[float, float]:
        self.model.train()

        loss_monitor = AverageMeter()
        accuracy_monitor = AverageMeter()

        for i, sample in enumerate(dataloader):
            X = sample['image'].to(device)
            y = sample['label'].type(torch.LongTensor)
            y = y.to(device)

            outputs = self.model(X)
            loss = criterion(outputs, y)

            cur_loss = loss.item()
            _, predicted = outputs.max(1)
            correct_num = predicted.eq(y).sum().item()

            sample_num = X.size(0)
            loss_monitor.update(cur_loss / sample_num)
            accuracy_monitor.update(correct_num / sample_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return epoch, loss_monitor.mean(), accuracy_monitor.mean()

    @decorator_classification_report
    @torch.no_grad()
    def validate(self, dataloader: torch.utils.data.DataLoader,
                 criterion: torch.nn.Module,
                 device: torch.device):
        self.model.eval()

        loss_monitor = AverageMeter()
        accuracy_monitor = AverageMeter()
        preds, gt = [], []

        # with torch.inference_mode():
        # with torch.no_grad():
        for i, sample in enumerate(dataloader):
            X = sample['image'].to(device)
            y = sample['label'].type(torch.LongTensor)
            y = y.to(device)

            test_pred_logits = self.model(X)

            outputs = torch.nn.functional.softmax(input=test_pred_logits, dim=1)
            softmax_vals, indices = outputs.max(1)

            loss = criterion(test_pred_logits, y)
            cur_loss = loss.item()

            _, predicted = outputs.max(1)
            correct_num = predicted.eq(y).sum().item()

            sample_num = X.size(0)
            loss_monitor.update(cur_loss / sample_num)
            accuracy_monitor.update(correct_num / sample_num)

            y = y.detach().cpu().numpy()
            pred = indices.detach().cpu().numpy()

            if len(preds) == 0:
                preds.append(pred)
                gt.append(y)
            else:
                preds[0] = np.append(preds[0], pred, axis=0)
                gt[0] = np.append(gt[0], y, axis=0)

        preds = np.concatenate(np.array(preds), axis=0)
        gt = np.concatenate(np.array(gt), axis=0)
        f1_weighted = round(f1_score(gt, preds, average="weighted", zero_division=0), 3)
        f1_macro = round(f1_score(gt, preds, average="macro", zero_division=0), 3)

        cm = np.around(confusion_matrix(gt, preds, normalize='true'), decimals=3)

        report = classification_report(gt, preds, target_names=list(cls.keys()), output_dict=True)
        report_p = classification_report(gt, preds, target_names=list(cls.keys()))
        return loss_monitor.mean(), accuracy_monitor.mean(), f1_weighted, f1_macro, report, report_p, cm


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def mean(self):
        return self.avg
