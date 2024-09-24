import argparse
import os
import torch

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--out_dir', default='RESULTS/Veres')
    # -- data options
    parser.add_argument('--data', default='Veres')
    parser.add_argument('--data_path', default='data/Veres/alltime/fate_train.pt')
    parser.add_argument('--data_dir', default='data/Veres')
    # -- model options
    parser.add_argument('--k_dims', default=[400,400])
    parser.add_argument('--activation', default='softplus')
    parser.add_argument('--sigma_type', default='const')
    parser.add_argument('--sigma_const', default=0.1)
    # -- train options
    parser.add_argument('--train_epochs', default=3000, type=int)
    parser.add_argument('--train_lr', default=0.005, type=float)
    parser.add_argument('--train_lambda', default=0.5, type=float)
    parser.add_argument('--train_batch', default=0.1, type=float)
    parser.add_argument('--train_clip', default=0.1, type=float)
    parser.add_argument('--save', default=500, type=int)
    # -- test options
    parser.add_argument('--evaluate_n', default=10000, type=int)
    parser.add_argument('--evaluate_data')
    parser.add_argument('--evaluate-baseline', action='store_true')
    # -- run options
    parser.add_argument('--task', default='fate')  # leaveout
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--evaluate')
    parser.add_argument('--config')
    # loss parameters
    parser.add_argument('--sinkhorn_scaling', default=0.7, type=float)
    parser.add_argument('--sinkhorn_blur', default=0.1, type=float)
    parser.add_argument('--ns', default=2000, type=float)

    parser.add_argument('--start_t', default=0, type=int)
    parser.add_argument('--train_t', default=[1, 2, 3, 4, 5, 6, 7])

    args = parser.parse_known_args()[0]
    args.layers = len(args.k_dims)
    
    return args


def init_config(args):

    args.layers= len(args.k_dims)

    args.kDims = '_'.join(map(str, args.k_dims))

    name = (
        "{activation}-{kDims}-"
        "{train_lambda}-{sigma_type}-{sigma_const}-"
        "{train_clip}-{train_lr}"
    ).format(**args.__dict__)

    args.out_dir = os.path.join(args.out_dir, name, 'seed_{}'.format(args.seed))

    if args.task == 'leaveout':
        args.out_dir = os.path.join(args.out_dir,args.leaveout_t)
    else:
        args.out_dir = os.path.join(args.out_dir,'alltime')

    if not os.path.exists(args.out_dir):
        print('Making directory at {}'.format(args.out_dir))
        os.makedirs(args.out_dir)
    else:
        print('Directory exists at {}'.format(args.out_dir))

    args.train_pt=os.path.join(args.out_dir, 'train.{}.pt')
    args.done_log=os.path.join(args.out_dir, 'done.log')
    args.config_pt=os.path.join(args.out_dir, 'config.pt')
    args.train_log = os.path.join(args.out_dir, 'train.log')

    return args
    

def load_data(args, base_dir="."):
    data_pt = torch.load(os.path.join(base_dir, args.data_path))
    x = data_pt['xp']
    y = data_pt['y']

    args.x_dim = x[0].shape[-1]


    return x, y, args

