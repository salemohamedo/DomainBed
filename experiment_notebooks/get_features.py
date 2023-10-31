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

def gather_activations(dataloader, model, feature_dim, N, K):
    """
    Collect N activations for each class from the model.

    :param dataloader: PyTorch dataloader
    :param model: Your trained model
    :param feature_dim: dimension of the feature/activation from the model
    :param N: number of samples you want for each class
    :param K: number of classes
    :return: Tensor of size [K, N, feature_dim]
    """

    # Store the activations here
    activations_tensor = torch.zeros([K, N, feature_dim])

    # Counters to keep track of how many samples we've seen for each label
    counters = {k: 0 for k in range(K)}

    # Ensure the model is in eval mode
    model.eval()

    with torch.no_grad():
        for x, y in dataloader:
            # Get activations
            activations = model.featurizer(x)

            # For each label in the batch
            for i in range(y.size(0)):
                label = y[i].item()

                # Check if we've already seen enough samples for this label
                if counters[label] < N:
                    activations_tensor[label, counters[label]] = activations[i]
                    counters[label] += 1

            # Check if we've collected enough samples for all labels
            if all([count >= N for count in counters.values()]):
                break

    return activations_tensor

argparser = argparse.ArgumentParser()
argparser.add_argument('--model_path', type=str, required=True)
argparser.add_argument('--n_domains', type=int, required=True)
args = argparser.parse_args()

model, model_args, hparams = load_model(f'/network/scratch/o/omar.salemohamed/domainbed/output/{args.model_path}', n_domains=args.n_domains)
model.eval()

loaders = load_dataset(model_args, hparams)

N = 1000
d = 1024
data_shape = (3, 224, 224)
all_features = torch.zeros(5, 2, N, d)
n_pc_components = 10
for env_id in range(5):
    loader = loaders[f'env{env_id}_out']
    all_features[env_id] = gather_activations(loader, model, d, N, 2)

pca = PCA()
pca_feats = pca.fit_transform(all_features.flatten(end_dim=-2))
pca_feats = pca_feats.reshape(5, 2, N, pca_feats.shape[-1])
# pca_feats = pca_feats[:, :, :, :2]
# print(pca_feats.shape)
df_data = []
for env_id in range(5):
    for label in [0, 1]:
        for n in range(N):
            df_data.append([env_id, label, n] + list(pca_feats[env_id, label, n, :n_pc_components]))
df = pd.DataFrame(df_data, columns = ['env', 'label', 'n'] + [f'pc_{i + 1}' for i in range(n_pc_components)])
for k, v in model_args.items():
    if isinstance(v, list) and len(v) > 1:
        raise Exception(k, v)
    elif isinstance(v, list):
        v = v[0]
    df[k] = v

df.to_pickle(f'dfs/feats_{model_args["run_id"]}.pk')
