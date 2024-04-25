import torch
import numpy as np

class EarlyStopping:
    """如果一段耐心限度后的验证损失无法改善，提前停止训练"""
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): 上次验证损失后最长等待多长时间改善.
                            Default: 7
            verbose (bool): 如果为true，打印每一个验证损失改进的消息.
                            Default: False
            delta (float): 受监控数量的最小变化，有资格作为改进.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0 # 计数器，记录验证损失没有改善的次数
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, dataset_name):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, dataset_name)
        elif score < self.best_score + self.delta:
            # 如果验证损失没有改善——验证损失小于最佳验证损失+delta
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, dataset_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, dataset_name):
        '''
        验证损失减少时保存模型.
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if dataset_name in ['handwritten0.mat', 'BRCA.mat', 'ROSMAP.mat']:
            torch.save(model.state_dict(), f'checkpoint_{dataset_name.split(".")[0]}.pt')     # 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss