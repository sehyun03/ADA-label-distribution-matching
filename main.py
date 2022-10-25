from __future__ import print_function
from utils.utils import *

import torch
import numpy as np
import cv2
import wandb
import os
import importlib
import random
import time
import torch
from options import Options

if __name__ == "__main__":
    args = Options().parse()
    args = Options().modify_command_options(args)

    ''' Fix seed '''
    random.seed(args.seed)
    cv2.setRNGSeed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic=True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    ''' Wandb log '''
    if args.dontlog:
        print("skip logging...")
        os.environ['WANDB_SILENT'] = 'true'
        os.environ['WANDB_MODE'] = 'dryrun'
    else:
        os.environ['WANDB_SILENT'] = 'false'
        os.environ['WANDB_MODE'] = 'run'
        os.environ['WANDB_NOTES'] = args.desc

    ''' Wandb sweep argument '''
    wandb.init(name=args.session, project='LAMDA')
    wandb.config.update(args)
    args.wandb = wandb

    ''' Define Trainer '''
    Trainer = importlib.import_module("trainer.{}".format(args.method.lower()))
    trainer = Trainer.Trainer(args)

    ''' Train & test '''
    full_training_start_time = time.time()
    if(args.testonly):
        trainer.test(trainer.test_loader, prefix='test')
    else:
        trainer.train()
    print("Training finished with {}".format((time.time() - full_training_start_time)//3600))
