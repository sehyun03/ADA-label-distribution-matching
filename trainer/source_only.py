from __future__ import print_function
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
import torch
from torch.utils.data import WeightedRandomSampler
from trainer import base
from loaders.return_dataset import get_loader

class Trainer(base.Trainer):
    def __init__(self, args):
        super().__init__(args)

        ### Class-balanced source sampling
        source_labels = torch.from_numpy(self.source_dataset.labels)
        nbins = len(torch.unique(source_labels))
        source_sample_weights = torch.zeros_like(source_labels).float()
        for i in range(nbins):
            source_sample_weights[source_labels == i] = (1 / nbins) / (source_labels == i).sum()
        source_sampler = WeightedRandomSampler(source_sample_weights.tolist(), len(self.source_dataset), replacement=True)
        self.source_loader = get_loader(args, self.source_dataset, shuffle=False, dl=True, sampler=source_sampler)

    def train_one_epoch(self, epoch):
        args = self.args
        trg_iter = iter(self.target_loader)
        src_iter = iter(self.source_loader)
        pbar = tqdm(trg_iter)
        self.model.train()
        
        ### Train one epoch
        for step, t_data in enumerate(pbar):
            try:
                s_data = next(src_iter)
            except(StopIteration):
                src_iter = iter(self.source_loader)
                s_data = next(src_iter)            
            s_img, s_gtlbl, s_idx = [i.cuda() for i in s_data]

            ### Feed forward with source
            s_logit, s_embs = self.model(s_img, getemb=True)
            loss_lbl = self.criterion(s_logit, s_gtlbl).mean()

            ### Optimize network
            global_step = epoch * (len(self.target_dataset) // args.bs) + step
            loss = loss_lbl
            loss.backward()
            self.am.add({'train-sloss': loss_lbl.detach().cpu().item()})
            self.am.add({'train-loss': loss.detach().cpu().item()})
            source_acc = torch.max(s_logit, dim=1)[1].eq(s_gtlbl).float().mean()
            self.am.add({'train-sacc': source_acc.detach().cpu().item()})
            self.optim.step()
            self.optim.zero_grad()

            ### Lr scheduling
            self.scheduler.step()

            ### Logging
            if step % args.log_interval == 0:
                pbar.set_description('[{} ep{}] step{} Loss_S {:.4f} Method {}'.format(
                        args.session,
                        epoch,
                        step,
                        self.am.get('train-sloss'),
                        args.method,
                        args.session))
                lr_f = self.optim.param_groups[1]['lr']
                wlog_train = {'learning-rate classifier': lr_f}
                wlog_train.update({k:self.am.pop(k) for k,v in self.am.get_whole_data().items()})
                self.args.wandb.log(wlog_train, step=global_step)