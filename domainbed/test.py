import torchvision
import torch

import os
import sys
import torch
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

sys.path.append('/home/mila/o/omar.salemohamed/DomainBed/')

import domainbed.algorithms as algorithms
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed import datasets
from types import SimpleNamespace
from domainbed.lib import misc
import argparse

def load_model(model_dir, n_domains):
    dump = torch.load(os.path.join(model_dir, 'model.pkl'))
    algorithm_class = algorithms.get_algorithm_class(dump["args"]["algorithm"])
    algorithm = algorithm_class(
        dump["model_input_shape"],
        dump["model_num_classes"],
        n_domains,
        dump["model_hparams"])
    algorithm.load_state_dict(dump["model_dict"])
    return algorithm, dump['args'], dump['model_hparams']

def load_dataset(args, hparams):
    args = SimpleNamespace(**args)
    dataset = vars(datasets)[args.dataset](args.data_dir,
                args.test_envs, hparams)
    in_splits = []
    out_splits = []
    uda_splits = []
    val_envs = []
    ## Fill defaults to play nicely with wandb sweeps
    if args.dataset == 'PACS':
        if args.test_envs == None:
            args.test_envs = [0]
        if args.use_densenet == None:
            args.use_densenet = False
    elif args.dataset == 'WILDSCamelyon':
        if args.test_envs == None:
            args.test_envs = [2]
        if args.use_densenet == None:
            args.use_densenet = True
        ## For consistency with wilds benchmark, env 1 should be only used as OOD val dataset
        val_envs.append(1)
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    eval_loaders = [torch.utils.data.DataLoader(
        dataset=env,
        shuffle=False,
        batch_size=64,
        num_workers=4)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]
    loader_dict = {}
    for i in range(len(eval_loaders)):
        loader_dict[eval_loader_names[i]] = eval_loaders[i]
    return loader_dict

model, model_args, hparams = load_model(f'/network/scratch/o/omar.salemohamed/domainbed/output/d29bfe0a-d014-41c4-8fb3-6a355a0ce84e', n_domains=3)
model.eval()

loaders = load_dataset(model_args, hparams)
l = loaders['env0_in']
# dn = torchvision.models.densenet121()
model = model.to('cuda')
# x = torch.randn(64, 3, 224, 224, device='cuda')
for i in range(10):
    x = torch.randn(64, 3, 224, 224, device='cuda')
    x = x.to('cuda')
    model.featurizer(x)

# for i, (x, y) in enumerate(l):
#     x = x.to('cuda')
#     model.featurizer(x)
#     if i > 9:
#         break