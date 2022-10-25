from __future__ import print_function
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import *
from utils.log_utils import AverageMeter
from utils.loss import *
from utils.layer_utils import load_weights
from loaders import dataset

import numpy as np
from tqdm import tqdm
from pathlib import Path
import time
from datetime import date
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.optim as optim

from loaders.return_dataset import get_dataset, get_loader
from model.resnet import ResNet
from utils.scheduler_utils import get_inv_lr_scheduler

class Trainer(object):
    def __init__(self, args):
        print('Dataset {} Source {} Target {} Network {}'.format(
               args.dataset, args.source, args.target, args.net))
        self.args = args

        ''' Get dataset & loaders '''
        self.nclass = args.ncls
        self.define_dataset(args)
        self.define_loader(args)

        ''' Define model '''
        self.model = self.define_model()
        
        ''' Resume model '''
        self.start_step = 0
        if(args.resume is not None):
            self.resume_model(args)
        else:
            args.start_step = -1
    
        ''' Optimizer '''
        self.define_optim(args)
        self.scheduler = get_inv_lr_scheduler(self.optim, last_epoch=self.start_step)
        self.total_step = args.max_epoch * (len(self.target_dataset) // args.bs)
        self.model.cuda()

        ''' Criterion '''
        self.criterion = nn.CrossEntropyLoss(reduction='none').cuda()
        self.criterion_test = nn.CrossEntropyLoss(reduction='none').cuda()
        
        ''' Define Logger '''
        self.start_date = date.today().strftime("%m/%d/%y")
        self.start_time = time.time()
        self.temp_logf = 'record/recent_run.txt'
        session_core = '_'.join(args.session.split('_')[:-1])
        session_conf = args.session.split('_')[-2] + args.session.split('_')[-1][:-3]
        log_dir = 'record/{}'.format(session_core)
        os.makedirs(log_dir, exist_ok=True)
        self.record_file = os.path.join(log_dir, '{}.txt'.format(session_conf))
        self.am = AverageMeter()
        self.am_test = AverageMeter()
        self.best_val_acc = 0
        self.acc_at_best_val = 0

        ''' Define checkpoint '''
        self.checkpath = 'checkpoint/{}'.format(args.session)
        os.makedirs(self.checkpath, exist_ok=True)

        ''' Define per-round number of annotations & sampling epochs '''
        total_budgets = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]
        self.budget = args.budget
        total_num_annos = [round(self.budget / total_budgets.index(self.budget) * len(self.target_dataset)) for i in range(total_budgets.index(self.budget))]
        total_sampling_epochs = [0, 5, 10, 15, 20]
        
        self.num_annos = total_num_annos[:total_budgets.index(self.budget)]
        self.sampling_epochs = total_sampling_epochs[:total_budgets.index(self.budget)]

    def define_dataset(self, args):
        self.source_dataset = get_dataset(args, split='source_train', transform='train')
        self.target_dataset = get_dataset(args, split='target_train', transform='train')
        self.val_dataset = get_dataset(args, split='target_val', transform='test')
        self.test_dataset = get_dataset(args, split='target_test', transform='test')

        ''' Remove validation set within target_dataset (when searching epoch with val set) '''
        isinter = np.logical_not(np.isin(self.target_dataset.imgs, self.val_dataset.imgs))
        assert(np.all(isinter))
        self.target_dataset.imgs = self.target_dataset.imgs[isinter]
        self.target_dataset.labels = self.target_dataset.labels[isinter]

    def define_loader(self, args):
        self.source_loader = get_loader(args, self.source_dataset, shuffle=True, dl=True,
                                        bs=args.bs, nw=args.num_workers)
        self.target_loader = get_loader(args, self.target_dataset, shuffle=True, dl=True,
                                        bs=args.bs, nw=args.num_workers)
        self.val_loader = get_loader(args, self.val_dataset, shuffle=False, dl=False,
                                     bs=args.bs, nw=2)
        self.test_loader = get_loader(args, self.test_dataset, shuffle=False, dl=False,
                                      bs=args.bs, nw=args.num_workers)

    def define_model(self):
        args = self.args
        if args.net == 'resnet50':
            model = ResNet(args=args, net=args.net, pretrained=not args.scratch)
            self.inc = 2048
        else:
            raise NotImplementedError

        return model

    def resume_model(self, args):
        assert(os.path.exists(args.resume)), "resume path not exist"
        weight_path = Path(args.resume)
        checkpoint = torch.load(weight_path)
        if 'model' in checkpoint:
            self.model = load_weights(self.model, checkpoint['model'])
        else:
            try:
                self.model = load_weights(self.model, checkpoint)
            except RuntimeError as e:
                print("[Loading error exception]:\n{}".format(e))
                self.model = load_weights(self.model, checkpoint, strict=False)

        print("Checkpoint {} loaded!".format(weight_path))

    def define_optim(self, args):
        param_groups = self.model.trainable_parameters()
        bblr = args.lr * 0.1
        param_list = [{'params': param_groups[0], 'lr': bblr, 'initial_lr': bblr},
                      {'params': param_groups[1], 'initial_lr': args.lr}] # small lr for backbone
            
        self.optim = optim.SGD(param_list, lr=args.lr, momentum=0.9, nesterov=True,
                                weight_decay=args.weight_decay)

    def train(self):
        args = self.args
        self.optim.zero_grad()
        self.eval_and_log(epoch = 0)
        self.save_model(epoch = 0)
        for epoch in range(args.max_epoch):
            self.train_one_epoch(epoch)
            self.eval_and_log(epoch + 1)
            if epoch % args.save_interval == 0 and epoch != 0:
                self.save_model(epoch + 1)
        self.save_model(args.max_epoch)
        self.log_final_result(mval_acc = self.acc_at_best_val)

    def eval_and_log(self, epoch):
        loss_test, acc_test = self.test(self.test_loader, prefix='test')
        loss_val, acc_val = self.test(self.val_loader, prefix='val')
        if acc_val >= self.best_val_acc:
            self.best_val_acc = acc_val
            self.acc_at_best_val = acc_test
            self.save_best_model(epoch + 1)
        wlog_test = {'test-tloss': loss_test.detach().cpu().item(), 'test-acc':acc_test,
                        'val-tloss': loss_val.detach().cpu().item(), 'val-acc':acc_val,
                        'epoch': epoch, 'test-acc-at-best-val': self.acc_at_best_val}
        wlog_test.update({k:self.am_test.pop(k) for k,v in self.am_test.get_whole_data().items()})
        global_step = epoch * (len(self.target_dataset) // self.args.bs)
        self.args.wandb.log(wlog_test, step=global_step)
        print('*****Best test acc at best val {:.1f}*****\n'.format(self.acc_at_best_val))

    def save_best_model(self, epoch):
        args = self.args
        saving_path = os.path.join(self.checkpath, "{}_ep{}.pth.tar".
                                    format(args.session, "_best"))
        save_checkpoint = {
            'session_name': args.session,
            'step': epoch * (len(self.target_dataset) // args.bs),
            'epcoh': epoch,
            'model': self.model.state_dict(),
        }
        torch.save(save_checkpoint, saving_path)

    def save_model(self, epoch):
        args = self.args
        saving_path = os.path.join(self.checkpath, "{}_ep{}.pth.tar".
                                    format(args.session, epoch))
        save_checkpoint = {
            'session_name': args.session,
            'step': epoch * (len(self.target_dataset) // args.bs),
            'epcoh': epoch,
            'model': self.model.state_dict(),
        }
        torch.save(save_checkpoint, saving_path)

    def train_one_epoch(self, epoch):
        """ override this funciton: generate training loop for each method """
        raise NotImplementedError

    def test(self, loader, prefix):
        self.model.eval()
        test_loss = 0
        correct = 0
        size = 0
        num_class = self.nclass
        output_all = np.zeros((0, num_class))
        pbar = tqdm(loader)
        with torch.no_grad():
            for batch_idx, data_t in enumerate(pbar):
                im_data_t = data_t[0].cuda()
                gt_labels_t = data_t[1].cuda()
                output1 = self.model(im_data_t)
                output_all = np.r_[output_all, output1.cpu().numpy()]
                size += im_data_t.size(0)
                pred1 = output1.max(1)[1]
                correct += pred1.eq(gt_labels_t.data).cpu().sum()
                test_loss += self.criterion_test(output1, gt_labels_t).mean() / len(loader)
                pbar.set_description('[{} set] Loss: {:.4f}, Correct: {}/{} Acc: {:.1f}%'.
                    format(prefix,
                           test_loss.detach().cpu().item(),
                           correct,
                           size,
                           100. * correct / size))
            
        return test_loss.data, 100. * float(correct) / size

    def log_final_result(self, mval_acc):
        ''' recent run '''
        run_time = (time.time() - self.start_time) / 3600.0
        with open(self.temp_logf, 'a') as f:
            f.write('[{}]   {:.2f}   {:.2f}   {}\n'.format(self.start_date, mval_acc, run_time, self.args.session))

        ''' configure-wise run '''
        scenario_list = dataset.get_scenario(self.args.dataset)
        scidx = scenario_list.index(self.args.session.split('_')[-1][-3:])
        if not os.path.exists(self.record_file):
            score_list = ["-" for i in scenario_list]
        else: # read 
            with open(self.record_file, "r") as f:
                prev_list = f.read().splitlines()
            score_list = [i.strip() for i in prev_list[-1].split('|')]
        score_list[scidx] = "{:.2f}".format(mval_acc)
        with open(self.record_file, 'w+') as f:
            f.write(tabulate([score_list], scenario_list, tablefmt="presto"))

    def load_data(self, data_iter, data_loader):
        try:
            data = next(data_iter)
        except(StopIteration):
            data_iter = iter(data_loader)
            data = next(data_iter)

        return data