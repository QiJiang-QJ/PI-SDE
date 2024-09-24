import torch
from torch import optim
import numpy as np
from geomloss import SamplesLoss
import tqdm
from src.model import ForwardSDE
import os
from src.config_Veres import load_data


def p_samp(p, num_samp, w=None):
    repflag = p.shape[0] < num_samp
    p_sub = np.random.choice(p.shape[0], size=num_samp, replace=repflag)

    if w is None:
        w_ = torch.ones(len(p_sub))
    else:
        w_ = w[p_sub].clone()
    w_ = w_ / w_.sum()

    return p[p_sub, :].clone(), w_



def init_device(args):
    args.cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device('cuda:{}'.format(args.device) if args.cuda else 'cpu')

    return device





# ---- loss

class OTLoss():

    def __init__(self, config, device):
        self.ot_solver = SamplesLoss("sinkhorn", p=2, blur=config.sinkhorn_blur,
                                     scaling=config.sinkhorn_scaling, debias=True)
        self.device = device

    def __call__(self, a_i, x_i, b_j, y_j, requires_grad=True):
        a_i = a_i.to(self.device)
        x_i = x_i.to(self.device)
        b_j = b_j.to(self.device)
        y_j = y_j.to(self.device)

        if requires_grad:
            a_i.requires_grad_()
            x_i.requires_grad_()
            b_j.requires_grad_()

        loss_xy = self.ot_solver(a_i, x_i, b_j, y_j)
        return loss_xy


def run(args,initial_config):
    # ---- initialize 

    device = init_device(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = initial_config(args)
    x, y, config = load_data(config)

    if os.path.exists(os.path.join(config.out_dir, 'train.log')):
        print(os.path.join(config.out_dir, 'train.log'), ' exists. Skipping.')
        
    else:
        model = ForwardSDE(config)
        print(model)
        model.zero_grad()

        loss = OTLoss(config, device)
        torch.save(config.__dict__, config.config_pt)

        model.to(device)
        optimizer = optim.Adam(list(model.parameters()), lr=config.train_lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
        optimizer.zero_grad()

        pbar = tqdm.tqdm(range(config.train_epochs))

        # fit on time points
        best_train_loss_xy = np.inf

        with open(config.train_log, 'w') as log_handle:
            for epoch in pbar:

                losses_xy = []
                losses_r = []
                config.train_epoch = epoch

                dat_prev = x[config.start_t]
                x_i, a_i = p_samp(dat_prev, int(dat_prev.shape[0] * args.train_batch))
                r_i = torch.zeros(int(dat_prev.shape[0] * args.train_batch)).unsqueeze(1)
                x_r_i = torch.cat([x_i,r_i], dim=1)
                x_r_i = x_r_i.to(device)
                ts = [0] + config.train_t
                y_ts = [np.float64(y[ts_i]) for ts_i in ts ]
                x_r_s = model( y_ts, x_r_i)

                for j in config.train_t:
                    t_cur = j

                    dat_cur = x[t_cur]
                    y_j, b_j = p_samp(dat_cur, int(dat_cur.shape[0] * args.train_batch))

                    position = config.train_t.index(j)
                    loss_xy = loss(a_i, x_r_s[position+1][:,0:-1], b_j, y_j)

                    losses_xy.append(loss_xy.item())

                    if (config.train_lambda > 0) & (j == config.train_t[-1]):
                        loss_r = torch.mean(x_r_s[-1][:,-1] * config.train_lambda)
                        losses_r.append(loss_r.item())
                        loss_all = loss_xy + loss_r
                    else:
                        loss_all = loss_xy

                    loss_all.backward(retain_graph=True)

                train_loss_xy = np.mean(losses_xy)
                train_loss_r = np.mean(losses_r)

                # step
                if config.train_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.train_clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()


                # report
                desc = "[train] {}".format(epoch + 1)
                desc += " {:.6f}".format(train_loss_xy)
                if config.train_lambda > 0:
                    desc += " {:.6f}".format(train_loss_r)
                desc += " {:.6f}".format(best_train_loss_xy)
                pbar.set_description(desc)
                log_handle.write(desc + '\n')
                log_handle.flush()

                if train_loss_xy < best_train_loss_xy:
                    best_train_loss_xy = train_loss_xy
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'epoch': config.train_epoch + 1,
                    }, config.train_pt.format('best'))

                # save model every x epochs
                if (config.train_epoch + 1) % config.save == 0:
                    epoch_ = str(config.train_epoch + 1).rjust(6, '0')
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'epoch': config.train_epoch + 1,
                    }, config.train_pt.format('epoch_{}'.format(epoch_)))


    return config


def run_leaveout(args,initial_config,leaveouts=None):
    # ---- initialize 

    device = init_device(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.task = 'leaveout'

    Train_ts = args.train_t

    args.leaveout_t = 'leaveout' + '&'.join(map(str, leaveouts)) 
    args.train_t = list(sorted(set(Train_ts)-set(leaveouts)))   
    args.test_t = leaveouts                                     
    print('--------------------------------------------')
    print('----------leaveout_t=',leaveouts,'---------')
    print('----------train_t=', args.train_t)
    print('--------------------------------------------')

    config = initial_config(args)
    x, y, config = load_data(config)

    if os.path.exists(os.path.join(config.out_dir, 'train.log')):
        print(os.path.join(config.out_dir, 'train.log'), ' exists. Skipping.')

    else:
        model = ForwardSDE(config)
        model.zero_grad()
        model.to(device)

        loss = OTLoss(config, device)
        torch.save(config.__dict__, config.config_pt)

        optimizer = optim.Adam(list(model.parameters()), lr=config.train_lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
        optimizer.zero_grad()

        pbar = tqdm.tqdm(range(config.train_epochs))

        best_train_loss_xy = np.inf

        with open(config.train_log, 'w') as log_handle:
            for epoch in pbar:

                losses_xy = []
                losses_r = []
                config.train_epoch = epoch

                dat_prev = x[config.start_t]
                x_i, a_i = p_samp(dat_prev, int(dat_prev.shape[0] * args.train_batch))
                r_i = torch.zeros(int(dat_prev.shape[0] * args.train_batch)).unsqueeze(1)
                x_r_i = torch.cat([x_i,r_i], dim=1)
                x_r_i = x_r_i.to(device)
                ts = [0] + config.train_t
                y_ts = [np.float64(y[ts_i]) for ts_i in ts ]
                x_r_s = model( y_ts, x_r_i)

                for j in config.train_t:
                    t_cur = j
                    dat_cur = x[t_cur]
                    y_j, b_j = p_samp(dat_cur, int(dat_cur.shape[0] * args.train_batch))

                    position = config.train_t.index(j) 
                    loss_xy = loss(a_i, x_r_s[position+1][:,0:-1], b_j, y_j)

                    losses_xy.append(loss_xy.item())

                    if (config.train_lambda > 0) & (j == config.train_t[-1]):
                        loss_r = torch.mean(x_r_s[-1][:,-1] * config.train_lambda)
                        losses_r.append(loss_r.item())
                        loss_all = loss_xy + loss_r
                    else:
                        loss_all = loss_xy

                    loss_all.backward(retain_graph=True)

                train_loss_xy = np.mean(losses_xy)
                train_loss_r = np.mean(losses_r)

                # step
                if config.train_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.train_clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

                # report

                desc = "[train] {}".format(epoch + 1)
                desc += " {:.6f}".format(train_loss_xy)
                if config.train_lambda > 0:
                    desc += " {:.6f}".format(train_loss_r)
                desc += " {:.6f}".format(best_train_loss_xy)
                pbar.set_description(desc)
                log_handle.write(desc + '\n')
                log_handle.flush()

                if train_loss_xy < best_train_loss_xy:
                    best_train_loss_xy = train_loss_xy
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'epoch': config.train_epoch + 1,
                    }, config.train_pt.format('best'))

                # save model every x epochs

                if (config.train_epoch + 1) % config.save == 0:
                    epoch_ = str(config.train_epoch + 1).rjust(6, '0')
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'epoch': config.train_epoch + 1,
                    }, config.train_pt.format('epoch_{}'.format(epoch_)))


    return config










