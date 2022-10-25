from __future__ import print_function
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
import torch
from torch import optim
import numpy as np
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
import math
from copy import deepcopy
from pathlib import Path

from trainer import base
from model.domain_discriminator import DomainDiscriminator
from loaders.return_dataset import get_loader, get_data_transforms
from utils.data import ForeverDataIterator
from utils.loss import DomainAdversarialLoss
from utils.mmd import rbf_kernel, select_prototypes_wo_plbl
from utils.layer_utils import load_weights

class Trainer(base.Trainer):
    def __init__(self, args):
        super().__init__(args)

        ''' domain adv loss '''
        self.adv_loss = DomainAdversarialLoss(self.domain_discri).cuda()

        ''' resume pretrained src model '''
        assert(os.path.exists(args.resume)), "resume path not exist"
        weight_path = Path(args.resume)
        checkpoint = torch.load(weight_path)
        self.model = load_weights(self.model, checkpoint['model'])
        print("Checkpoint {} loaded!".format(weight_path))

        ''' selected inidices place holder '''
        self.selected_indices = torch.tensor([], dtype=torch.int64)

    def resume_model(self, args):
        print("last model loading")

    def define_model(self):
        ''' domain discreminator '''
        domain_discri = DomainDiscriminator(in_feature=512, hidden_size=1024)
        self.domain_discri = domain_discri.cuda()

        return super().define_model()

    def define_optim(self, args):
        param_groups = self.model.trainable_parameters()
        bblr = args.lr * 0.1
        param_list = [{'params': param_groups[0], 'lr': bblr, 'initial_lr': bblr}, # small lr for backbone
                      {'params': param_groups[1] + list(self.domain_discri.parameters()), 'initial_lr': args.lr}]
        self.optim = optim.SGD(param_list, lr=args.lr, momentum=0.9, nesterov=True,
                                weight_decay=args.weight_decay)

    def update_proto(self, epoch):
        args = self.args
        num_anno = self.num_annos[self.sampling_epochs.index(epoch)]
        round = self.sampling_epochs.index(epoch) + 1
        plbl_cratio = 2 * round

        ''' get target feature vectors '''
        self.target_stand_dataset = deepcopy(self.target_dataset) ### all of the target data
        self.target_stand_dataset.transform = get_data_transforms(args, transform='test')
        target_stand_loader = get_loader(args, self.target_stand_dataset, False, bs=args.bs, dl=False)
        output_all = torch.zeros((len(self.target_stand_dataset), args.ncls))
        feat_all = torch.zeros((len(self.target_stand_dataset), self.inc))
        pbar = tqdm(target_stand_loader)

        self.model = self.model.cuda()
        self.model.eval()
        with torch.no_grad():
            for batch_idx, data_t in enumerate(pbar):
                im_data_t = data_t[0].cuda()
                output1, feat = self.model(im_data_t, getfeat=True)
                if batch_idx == (len(target_stand_loader) - 1):
                    output_all[batch_idx * args.bs:] = output1.cpu()
                    feat_all[batch_idx * args.bs:] = feat.cpu()
                else:
                    output_all[batch_idx * args.bs: (batch_idx + 1) * args.bs] = output1.cpu()        
                    feat_all[batch_idx * args.bs: (batch_idx + 1) * args.bs] = feat.cpu()
        feat_rbfk = rbf_kernel(feat_all)

        ''' get pseudo label indices '''
        pred_prob = torch.softmax(output_all, dim=1)
        pred_all = torch.argmax(pred_prob, dim=1)
        confu, predu = torch.topk(pred_prob, k=2, dim=1)
        confu_gap = confu[:, 0] - confu[:, 1]
        isvalid = confu_gap > 0.8
        predu_valid = -1 * torch.ones_like(predu[:, 0])
        predu_valid[isvalid] = predu[:, 0][isvalid]
        plbl_idxs = isvalid.nonzero()

        ''' add pervious selected indices into plbl_idxs '''
        plbl_idxs = torch.cat([isvalid.nonzero(), self.selected_indices[:,None].long()])

        ''' Get confidence probability '''
        pconfidx = -1 * torch.ones_like(predu_valid)
        pconfidx[predu_valid != -1] = predu_valid[predu_valid != -1]
        pconfu = torch.zeros_like(pconfidx).float()
        pconfu[predu_valid != -1] = pred_prob[torch.arange(pconfidx.shape[0]).long()[predu_valid != -1], pconfidx[predu_valid != -1]]
        pconfu[torch.tensor(self.selected_indices).long()] = 0 # already selected indices

        ''' target labels for active sampling '''
        self.target_label = torch.from_numpy(self.target_dataset.labels)
        nbins = len(torch.unique(self.target_label))
        self.nbins = nbins

        ''' get estimated target label distribution '''
        proto_indices, plbl_indices = select_prototypes_wo_plbl(feat_rbfk, num_anno, plbl_idxs, plbl_cratio)
        self.plbl_indices = plbl_indices
        self.selected_indices = torch.cat([self.selected_indices, proto_indices])

        plbl_labels = pred_all[plbl_indices].numpy()
        plbl_hist = np.histogram(plbl_labels, bins=self.nbins, weights=pconfu[plbl_indices], range=(0, self.nbins))[0]

        sampled_labels = self.target_label[torch.tensor(self.selected_indices)].numpy()
        samp_hist = np.histogram(sampled_labels, bins=self.nbins, range=(0, self.nbins))[0]

        sampled_hist = torch.from_numpy(plbl_hist + samp_hist)
        sampled_dist = F.normalize(sampled_hist + 1, p=1, dim=0).float()
        self.sampled_dist = sampled_dist

        ''' get source sampling weights  '''
        source_labels = torch.from_numpy(self.source_dataset.labels)
        source_sample_weights = torch.zeros_like(source_labels).float()
        for i in range(self.nbins):
            source_sample_weights[source_labels == i] = sampled_dist[i] / (source_labels == i).sum()
        source_sampler = WeightedRandomSampler(source_sample_weights.tolist(), len(self.source_dataset), replacement=True)
        self.source_loader = get_loader(args, self.source_dataset, shuffle=False, dl=True, sampler=source_sampler)

        ''' generate labeled target dataset (repeat dataset into the size of target_dataset) '''
        args = self.args
        n_tgt = len(self.target_dataset)
        n_lbl = self.selected_indices.shape[0]
        if (n_lbl - (n_lbl % args.bs)) == 0: # exception when n_lbl is smaller than minibatch size
            n_lbl_ep = args.bs
        else:
            n_lbl_ep = (n_lbl - (n_lbl % args.bs)) # number of labeled data per epoch with drop_last = True
        n_repeat = math.ceil(n_tgt/n_lbl_ep) # size difference ratio between target & lbl target

        self.labeled_target_dataset = deepcopy(self.target_dataset)
        imgs = self.target_dataset.imgs[self.selected_indices]
        labels = self.target_dataset.labels[self.selected_indices]

        imgs_all = []
        labels_all = []
        for i in tqdm(range(n_repeat)):
            if n_lbl < n_lbl_ep:
                minibatch_indices_1 = torch.randperm(n_lbl)
                minibatch_indices_2 = torch.randperm(n_lbl)[:(n_lbl_ep - n_lbl)]
                minibatch_indices = torch.cat([minibatch_indices_1, minibatch_indices_2])
            else:
                minibatch_indices = torch.randperm(n_lbl)[:n_lbl_ep]
            imgs_all.append(imgs[minibatch_indices].copy())
            labels_all.append(labels[minibatch_indices].copy())
        imgs = np.concatenate(imgs_all)
        labels = np.concatenate(labels_all)

        self.labeled_target_dataset.imgs = imgs
        self.labeled_target_dataset.labels = labels
        self.labeled_target_loader = get_loader(args, self.labeled_target_dataset, shuffle=False, dl=False, bs=args.bs, nw=args.num_workers)        
      
    def train(self):
        args = self.args
        self.optim.zero_grad()
        self.eval_and_log(epoch = 0)
        self.save_model(epoch = 0)
        for epoch in range(args.max_epoch):
            if epoch in self.sampling_epochs:
                self.update_proto(epoch)
            self.train_one_epoch(epoch)
            self.eval_and_log(epoch + 1)
            if epoch % args.save_interval == 0 and epoch != 0:
                self.save_model(epoch + 1)
        self.save_model(args.max_epoch)
        self.log_final_result(mval_acc = self.acc_at_best_val)

    def train_one_epoch(self, epoch):
        args = self.args
        src_iter = ForeverDataIterator(self.source_loader)
        trg_iter = iter(self.target_loader)
        ltrg_iter = iter(self.labeled_target_loader)
        pbar = tqdm(trg_iter)
        self.model.train()
        self.adv_loss.train()

        ''' Train one epoch '''
        for step, t_data in enumerate(pbar):

            ''' load S '''
            s_data = next(src_iter)
            s_img, s_gtlbl, s_idx = [i.cuda() for i in s_data]

            ''' load T '''
            t_img, _, t_idx = [i.cuda() for i in t_data]

            ''' load labeled T '''
            lt_data = next(ltrg_iter)
            lt_img, lt_gtlbl, lt_idx = [i.cuda() for i in lt_data]

            ''' Feedforward S,T '''
            logit, embs = self.model(torch.cat([s_img, t_img]), getemb=True, normemb=False)
            logit_s, logit_t = logit.chunk(2, dim=0)
            embs_s, embs_t = embs.chunk(2, dim=0)
            s_loss_lbl = self.criterion(logit_s, s_gtlbl).mean()
            transfer_loss = self.adv_loss(embs_s, embs_t)
            domain_acc = self.adv_loss.domain_discriminator_accuracy

            ''' Feedforward labeled T '''
            lt_logit, lt_embs = self.model(lt_img, getemb=True, normemb=False)
            lt_loss_lbl = self.criterion(lt_logit, lt_gtlbl).mean()

            ''' Optimize network & log '''
            global_step = epoch * (len(self.target_dataset) // args.bs) + step
            loss = s_loss_lbl + lt_loss_lbl + transfer_loss
            loss.backward()
            self.am.add({'train-sloss': s_loss_lbl.detach().cpu().item()})
            self.am.add({'train-adv': transfer_loss.detach().cpu().item()})
            self.am.add({'train-tloss': lt_loss_lbl.detach().cpu().item()})
            self.am.add({'train-loss': loss.detach().cpu().item()})
            source_acc = torch.max(logit_s, dim=1)[1].eq(s_gtlbl).float().mean()
            self.am.add({'train-sacc': source_acc.detach().cpu().item()})
            self.am.add({'train-domainacc': domain_acc.detach().cpu().item()})
            self.optim.step()
            self.optim.zero_grad()

            ''' Schedule lr '''
            self.scheduler.step()

            ''' Print current training process '''
            if step % args.log_interval == 0:
                pbar.set_description('[{} ep{}] step{} Loss_S {:.4f} Loss_T {:.4f} Method {}'.format(
                        args.session,
                        epoch,
                        step,
                        self.am.get('train-sloss'),
                        self.am.get('train-tloss'),
                        args.method,
                        args.session))
                lr_f = self.optim.param_groups[1]['lr']
                wlog_train = {'learning-rate classifier': lr_f}
                wlog_train.update({k:self.am.pop(k) for k,v in self.am.get_whole_data().items()})
                self.args.wandb.log(wlog_train, step=global_step)