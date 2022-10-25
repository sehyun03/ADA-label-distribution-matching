from torch.optim import lr_scheduler

def get_inv_lr_scheduler(optimizer, gamma=0.0001, power=0.75, last_epoch=-1):
    r"""Get learning rate scheduler decayinng by a power of factor according to step
        (confirmed) same learning rate decaying as in MME """
    lr_lambda = lambda epoch: (1 + gamma * epoch) ** (-power)
    return lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)