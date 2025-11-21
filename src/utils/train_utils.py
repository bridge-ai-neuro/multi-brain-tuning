import torch
import torch.nn as nn
from models import Wav2VecLoRA, HuBERTLoRA
import numpy as np 
import config
test_stories = ["wheretheressmoke"]
val_stories = ['itsabox']
increase_epochs = 10  # Epochs to reach the peak learning rate
total_epochs = 30    # Total number of training epochs


def get_model(model_ckpt, out_dim, lora_rank, bottleneck_dim, device):
    if 'wav2vec' in model_ckpt:
        model = Wav2VecLoRA(out_dim=out_dim,
                            bottleneck_dim=bottleneck_dim, lora_rank=lora_rank,
                            wav2vec_model_name=model_ckpt).to(device)
    elif 'hubert' in model_ckpt:
        model = HuBERTLoRA(out_dim=out_dim,
                           bottleneck_dim=bottleneck_dim, lora_rank=lora_rank,
                           hubert_model_name=model_ckpt).to(device)
    return model

def get_loss_function(loss_name):
    if loss_name == 'cosl2':
        return cosl2_loss_function
    elif loss_name == 'corr':
        return SpatialCorrelationLoss()
    elif loss_name == 'l2':
        return nn.MSELoss()

def get_train_params(model, has_bottleneck=False):
    model_trainable_params = list(filter(lambda p: p.requires_grad, model.lora_model.parameters()))    
    if has_bottleneck:
        linear_trainable_params = list(filter(lambda p: p.requires_grad, model.bottleneck.parameters()))
    else:
        linear_trainable_params = list(filter(lambda p: p.requires_grad, model.linear.parameters()))
    return {'model': model_trainable_params, 'linear': linear_trainable_params}

def evaluate(model, dataloader, loss_function, device):
    # model.eval()
    total_loss = 0
    elosses = []
    with torch.no_grad():
        for input_wav_tensor, output_signal in dataloader:
            input_wav_tensor = input_wav_tensor.to(device)
            output_signal = output_signal.to(device)
            predictions = model(input_wav_tensor)
            loss = loss_function(predictions, output_signal)
            total_loss += loss.item()
            elosses.append(loss.item()) 
    return np.mean(elosses)

def schedule_group_0(epoch):
    if epoch == 0:
        return 0.05
    if epoch < increase_epochs:
        return epoch / increase_epochs
    else:
        return 1 - (epoch - increase_epochs) / (total_epochs - increase_epochs)

# Schedule for the second parameter group (exponential decay)
def schedule_group_1(epoch):
    return max(0.8 ** epoch, 0.1)

# Combining the schedules
def combined_schedule(epoch):
    return [schedule_group_0(epoch), schedule_group_1(epoch)]

class SpatialCorrelationLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps  # Small value to prevent numerical issues

    def forward(self, pred, target):
        """
        Compute negative spatial correlation loss (over voxels).
        :param pred: Predicted voxel representations (batch_size, num_voxels)
        :param target: Ground-truth voxel representations (batch_size, num_voxels)
        :return: Negative correlation loss (scalar)
        """
        # Compute mean per batch sample (mean over voxels)
        pred_mean = pred.mean(dim=1, keepdim=True)
        target_mean = target.mean(dim=1, keepdim=True)

        # Normalize to zero mean
        pred_centered = pred - pred_mean
        target_centered = target - target_mean

        # Compute Pearson correlation per sample
        numerator = (pred_centered * target_centered).sum(dim=1)
        denominator = torch.sqrt((pred_centered ** 2).sum(dim=1) * (target_centered ** 2).sum(dim=1) + self.eps)

        correlation = numerator / denominator  # Shape: (batch,)

        # Loss: negative mean correlation across batch
        loss = 1 - correlation.mean()

        return loss


class SparseRobustLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(SparseRobustLoss, self).__init__()
        self.delta = delta

    def forward(self, predictions, targets):
        error = predictions - targets
        abs_error = torch.abs(error)

        quadratic = torch.minimum(abs_error, torch.tensor(self.delta))
        linear = abs_error - quadratic

        loss = 0.5 * quadratic ** 2 + self.delta * linear
        return loss.mean()

import torch.nn.functional as F
class CosineSimilarityLoss(nn.Module):
    """
    Cosine Similarity Loss: loss = 1 - cosine_similarity
    Supports inputs of shape [B, D] or [B, T, D]
    """
    def __init__(self, dim=-1, reduction='mean'):
        super().__init__()
        self.dim = dim
        self.reduction = reduction

    def forward(self, x, y):
        """
        Args:
            x: predicted tensor (e.g., [B, T, D] or [B, D])
            y: target tensor (same shape as x)
        Returns:
            loss: scalar tensor
        """
        cos_sim = F.cosine_similarity(x, y, dim=self.dim)

        # loss = 1 - similarity
        loss = 1 - cos_sim

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # no reduction

def cosl2_loss_function(predictions, targets):
    # Compute the loss between predictions and targets
    mse = SparseRobustLoss(delta=0.6)
    corr = CosineSimilarityLoss()
    loss = mse(predictions, targets) + corr(predictions, targets)
    return loss

