from config_Veres import load_data
from src.model import ForwardSDE
import torch
import numpy as np
import pandas as pd
from geomloss import SamplesLoss 
import glob
import os
import src.train as train
from src.emd import earth_mover_distance
from types import SimpleNamespace

def init_device(args):
    args.cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device('cuda:{}'.format(args.device) if args.cuda else 'cpu')

    return device

def derive_model(args, ckpt_name='epoch_003000'):
    device = init_device(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x, y, config = load_data(args)
    model = ForwardSDE(config)

    train_pt = "./" + config.train_pt.format(ckpt_name)
    checkpoint = torch.load(train_pt)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    return model, x, y, device


def evaluate_fit(args, initial_config, use_loss='emd'):
    
    device = init_device(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = initial_config(args)
    x, y, config = load_data(config)
    config = SimpleNamespace(**torch.load(config.config_pt))

    file_info = 'interpolate' + use_loss + '.log'
    log_path = os.path.join(config.out_dir, file_info)
    
    if os.path.exists(log_path):
        print(log_path, 'exists. Skipping.')
        return              

    losses_xy = []
    train_pts = sorted(glob.glob(config.train_pt.format('*')))
    print(config.train_pt)
    print(train_pts)

    for train_pt in train_pts:

        model = ForwardSDE(config)
        checkpoint = torch.load(train_pt)
        print('Loading model from {}'.format(train_pt))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        name = os.path.basename(train_pt).split('.')[1]

        for t in config.train_t:
            loss_xy = _evaluate_impute_model(config, t, model, x, y,device, use_loss).item()
            losses_xy.append((name, 'train', y[t], loss_xy))
        try:
            for t in config.test_t: 
                loss_xy = _evaluate_impute_model(config, t, model, x, y,device, use_loss).item()
                losses_xy.append((name, 'test', y[t], loss_xy))
        except AttributeError:
            continue

    losses_xy = pd.DataFrame(losses_xy, columns = ['epoch', 'eval', 't', 'loss'])
    losses_xy.to_csv(log_path, sep = '\t', index = False)
    print(losses_xy)
    print('Wrote results to', log_path)
    


def evaluate_fit_leaveout(args,initial_config,leaveouts=None,use_loss='emd'):

    device = init_device(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.data == 'Veres':
        Train_ts = [1,2,3,4,5,6,7]
    elif args.data == 'Weinreb':
        Train_ts = [1,2]

    args.leaveout_t = 'leaveout' + '&'.join(map(str, leaveouts)) 
    args.train_t = list(sorted(set(Train_ts)-set(leaveouts)))   
    args.test_t = leaveouts 
    print('---------------Evaluation-------------------')                                   
    print('--------------------------------------------')
    print('----------leaveout_t=',leaveouts,'---------')
    print('----------train_t=', args.train_t)
    print('--------------------------------------------')

    args.out_dir = 'RESULTS/' + args.data

    config = initial_config(args)
    x, y, config = load_data(config)
    config = SimpleNamespace(**torch.load(config.config_pt))

    if os.path.exists(os.path.join(config.out_dir, 'train.log')): 
        print(os.path.join(config.out_dir, 'train.log'), ' exists.')

        file_info = 'interpolate-' + use_loss + '-all.log'
        log_path = os.path.join(config.out_dir, file_info)
        
        if os.path.exists(log_path):
            print(log_path, 'exists. Skipping.')
            return
        model = ForwardSDE(config)

        losses_xy = []
        train_pts = sorted(glob.glob(config.train_pt.format('*')))
        print(config.train_pt)
        print(train_pts)

        for train_pt in train_pts:
            checkpoint = torch.load(train_pt)
            print('Loading model from {}'.format(train_pt))
            model.load_state_dict(checkpoint['model_state_dict'])

            del checkpoint
            
            model.to(device)
            print(model)

            name = os.path.basename(train_pt).split('.')[1]

            for t in config.train_t:
                loss_xy = _evaluate_impute_model(config, t, model, x, y, device, use_loss).item()
                losses_xy.append((name, 'train', y[t], loss_xy))
            try:
                for t in config.test_t: 
                    loss_xy = _evaluate_impute_model(config, t, model, x, y, device, use_loss).item()
                    losses_xy.append((name, 'test', y[t], loss_xy))
            except AttributeError:
                continue

        losses_xy = pd.DataFrame(losses_xy, columns = ['epoch', 'eval', 't', 'loss'])

        out_dir = config.out_dir
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        losses_xy.to_csv(log_path, sep = '\t', index = False)
        print(losses_xy)
        print('Wrote results to', log_path)




        
def _evaluate_impute_model(config, t_cur, model, x,w, y,device, use_loss='emd'):

    torch.manual_seed(0)
    np.random.seed(0)

    ot_solver = SamplesLoss("sinkhorn", p=2, blur=config.sinkhorn_blur,
                                        scaling=config.sinkhorn_scaling)

    x_0, _ = train.p_samp(x[0], config.evaluate_n)
    r_0 = torch.zeros(int(config.evaluate_n)).unsqueeze(1)
    x_r_0 = torch.cat([x_0,r_0], dim=1)
    x_r_0 = x_r_0.to(device)

    x_r_s = []
    for i in range(int(config.evaluate_n / config.ns)):
        x_r_0_ = x_r_0[i * config.ns:(i + 1) * config.ns, ]
        x_r_s_ = model( [np.float64(y[0])] + [np.float64(y[t_cur])], x_r_0_)
        x_r_s.append(x_r_s_[-1].detach())

    x_r_s = torch.cat(x_r_s)
    y_t = x[t_cur]
    print('y_t',y_t.shape)

    if use_loss == 'ot':
        loss_xy = ot_solver(x_r_s[:,0:-1].contiguous(), y_t.contiguous().to(device))
    elif use_loss == 'emd':
        loss_xy = earth_mover_distance(x_r_s[:,0:-1].cpu().numpy(), y_t)
    
    return loss_xy        















































        