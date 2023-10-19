# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

import wandb

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from tqdm import tqdm

## Setting default hparams here to play nicely with wandb sweeps, remove later
DEFAULT_HPARAMS = '{\
"batch_size": 32, \
"beta1": 0.0, \
"class_balanced": false, \
"d_steps_per_g_step": 1, \
"data_augmentation": true, \
"lr": 1e-05, \
"lr_d": 1e-05, \
"lr_g": 1e-05, \
"mlp_dropout": 0, \
"resnet18": false, \
"resnet_dropout": 0.0, \
"weight_decay": 0, \
"weight_decay_d": 0, \
"weight_decay_g": 0\
}'

def parse_bool(v):
    if v.lower()=='true':
        return True
    elif v.lower()=='false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    # parser.add_argument('--data_dir', type=str, required=False)
    parser.add_argument('--dataset', type=str, default="WILDSCamelyon")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--dann_disc_loss', type=str, default="NegCE")
    parser.add_argument('--dann_lambda', type=float, default=1.0)
    parser.add_argument('--grad_penalty', type=float, default=0)
    parser.add_argument('--mlp_width', type=int, default=256)
    parser.add_argument('--mlp_depth', type=int, default=5)
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict', default=DEFAULT_HPARAMS)
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+')
    # parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--save_model', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--use_densenet', type=parse_bool, const=True, nargs='?')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    args = parser.parse_args()

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

    print(args)

    if args.dataset == 'WILDSCamelyon' and args.test_envs != [2]:
        raise Exception('Test env must be set to [2] for WILDSCamelyon')

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    run_id = str(uuid.uuid4())
    args.output_dir = os.path.join('/network/scratch/o/omar.salemohamed/domainbed/output', run_id)
    args.run_id = run_id


    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))
    
    hparams['dann_disc_loss'] = args.dann_disc_loss
    hparams['dann_lambda'] = args.dann_lambda
    hparams['grad_penalty'] = args.grad_penalty
    hparams['mlp_width'] = args.mlp_width
    hparams['mlp_depth'] = args.mlp_depth
    hparams['use_densenet'] = args.use_densenet

    wandb_config = args.__dict__
    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))
        wandb_config[f'hparam_{k}'] = v

    wandb.init(
        project='domainbed',
        dir='/network/scratch/o/omar.salemohamed/wandb',
        config=wandb_config,
        name=f'{wandb_config["dataset"]}_{wandb_config["algorithm"]}_{wandb_config["dann_disc_loss"]}_{wandb_config["seed"]}',
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        data_dir = '/home/mila/o/omar.salemohamed/DomainBed/domainbed/data/'
        # data_dir = os.getenv('SLURM_TMPDIR')
        # data_dir = '/network/scratch/o/omar.salemohamed/domainbed/data'
        dataset = vars(datasets)[args.dataset](data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
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

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in (args.test_envs + val_envs)]

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs + val_envs), hparams)
    
    if args.algorithm == 'DANN':
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        disc_params = count_parameters(algorithm.discriminator)
        feat_params = count_parameters(algorithm.featurizer)
        wandb.config['disc_params'] = disc_params
        wandb.config['feat_params'] = feat_params

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    def save_checkpoint(filename):
        if not args.save_model:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))


    last_results_keys = None
    for step in tqdm(range(start_step, n_steps)):
        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
            for x,y in next(train_minibatches_iterator)]
        if args.task == "domain_adaptation":
            uda_device = [x.to(device)
                for x,_ in next(uda_minibatches_iterator)]
        else:
            uda_device = None
        step_vals = algorithm.update(minibatches_device, uda_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)


        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)
            
            ## For camelyon, only do full eval every 1000 as it's costly
            if args.dataset != 'WILDSCamelyon' or (step % 1000 == 0) or (step == n_steps - 1):
                evals = zip(eval_loader_names, eval_loaders, eval_weights)
                for name, loader, weights in evals:
                    if args.dataset == 'WILDSCamelyon':
                        if '_in' in name and name != f'env{args.test_envs[0]}_in':
                            ## Skip eval on train splits during training as camelyon is very large
                            continue
                    acc = misc.accuracy(algorithm, loader, weights, device)
                    results[name+'_acc'] = acc
                    if name == f'env{args.test_envs[0]}_in':
                        results['testenv_in_acc'] = acc
                    elif name == f'env{args.test_envs[0]}_out':
                        results['testenv_out_acc'] = acc

            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)


            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)
            wandb.log(results)
            results.update({
                'hparams': hparams,
                'args': vars(args)
            })


            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')

    save_checkpoint('model.pkl')

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
