import random
import numpy as np
import torch
import torch.optim as optim

def set_seed(seed):
    """
    设置所有关键库的随机种子，确保实验可重现性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"✅ Random seed set to {seed}")


class NoamScheduler(optim.lr_scheduler._LRScheduler):
    """
    Transformer 论文中使用的 Noam 学习率调度器。
    lr = d_model^{-0.5} * min(step_num^{-0.5}, step_num * warmup_steps^{-1.5})
    """
    def __init__(self, optimizer, d_model, warmup_steps, last_epoch=-1, verbose=False):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        self.step_num += 1
        
        # 暖启动阶段： step_num * warmup_steps^{-1.5}
        # 衰减阶段： step_num^{-0.5}
        scale = (self.d_model ** -0.5) * min(self.step_num ** -0.5, self.step_num * (self.warmup_steps ** -1.5))
        
        # 返回更新后的学习率
        return [base_lr * scale for base_lr in self.base_lrs]

    def state_dict(self):
        """返回调度器的状态字典，包含 step_num"""
        state_dict = super().state_dict()
        state_dict['step_num'] = self.step_num
        return state_dict

    def load_state_dict(self, state_dict):
        """加载调度器的状态字典"""
        self.step_num = state_dict['step_num']
        super().load_state_dict(state_dict)
