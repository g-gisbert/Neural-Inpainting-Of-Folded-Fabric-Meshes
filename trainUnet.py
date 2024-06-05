"""
Original author: Jan Bednarik
Adapated: Guillaume Gisbert
"""

# Python std
import sys
from timeit import default_timer as timer

# project files
import helpers
from data_loader2D import SF_Dataset
from R2ATT_UNET import AUNet

# 3rd party
import torch
import numpy as np
from torch.utils.data import DataLoader

import os


def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))


force_cudnn_initialization()

# Settings.
print_loss_tr_every = 50
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Config
conf = {
'lr': 0.001,
'batch_size': 2, # Change this to 32 to reproduce paper results
'epochs': 500,
'lr_factor': 0.3,
'lr_patience': 25,
'lr_min': 0.00000001,
'lr_threshold': 0.0001,
'alpha': 10.0,
'beta': 0.0001,
'gamma': 1.0}

os.makedirs(f"./experiments", exist_ok=True)
n_directory = len(helpers.lsd("./experiments"))
os.makedirs(f"./output", exist_ok=True)
os.makedirs(f"./experiments/exp{n_directory}", exist_ok=True)

model = AUNet().to(device)
#model.load_state_dict(torch.load("./output/chkpt.tar")['weights'])

# Create data loaders.
ds_tr = SF_Dataset(train=True)
ds_va = SF_Dataset(train=False)
dl_tr = DataLoader(
    ds_tr, batch_size=conf['batch_size'], shuffle=True, num_workers=4,
    drop_last=True)
dl_va = DataLoader(
    ds_va, batch_size=conf['batch_size'], shuffle=False, num_workers=2,
    drop_last=True)

print('Train ds: {} samples'.format(len(ds_tr)))
print('Valid ds: {} samples'.format(len(ds_va)))

# Prepare training.
opt = torch.optim.Adam(model.parameters(), lr=conf['lr'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, factor=conf['lr_factor'], patience=conf['lr_patience'],
    min_lr=conf['lr_min'], threshold=conf['lr_threshold'], verbose=True)

# Prepare savers.
saver = helpers.TrainStateSaver(
    helpers.jn('./output', 'w.tar'), model=model, optimizer=opt,
    scheduler=scheduler)
saverBest = helpers.TrainStateSaver(
    helpers.jn('./output', 'wBest.tar'), model=model, optimizer=opt,
    scheduler=scheduler)

# Training loop.
iters_tr = int(np.ceil(len(ds_tr) / float(conf['batch_size'])))
iters_va = int(np.ceil(len(ds_va) / float(conf['batch_size'])))
losses_tr = helpers.RunningLoss()
losses_va = helpers.RunningLoss()
bestVa = float('inf')

for ep in range(1, conf['epochs'] + 1):

    # Training.
    tstart = timer()
    model.train()
    for bi, batch in enumerate(dl_tr, start=1):
        it = (ep - 1) * iters_tr + bi
        gt = batch['pcloud'].to(device)
        input_damaged = batch['input'].to(device)
        mask = batch['mask'].to(device)
        data = torch.cat((input_damaged, mask.unsqueeze(1)), dim=1)
        model(data)
        losses = model.loss(gt, conf)

        opt.zero_grad()
        losses['loss_tot'].backward()
        opt.step()

        losses_tr.update(**{k: v.item() for k, v in losses.items()})
        if bi % print_loss_tr_every == 0:
            losses_avg = losses_tr.get_losses()
            losses_tr.reset()

            strh = '\rep {}/{}, it {}/{}, {:.0f} s - '. \
                format(ep, conf['epochs'], bi, iters_tr, timer() - tstart)
            strl = ', '.join(['{}: {:.4f}'.format(k, v)
                              for k, v in losses_avg.items()])
            print(strh + strl, end='')


    # Validation.
    model.eval()
    it = ep * iters_tr
    loss_va_run = 0.
    for bi, batch in enumerate(dl_va):
        curr_bs = batch['pcloud'].shape[0]
        gt = batch['pcloud'].to(device)
        input_damaged = batch['input'].to(device)
        mask = batch['mask'].to(device)
        data = torch.cat((input_damaged, mask.unsqueeze(1)), dim=1)
        model(data)
        lv = model.loss(gt, conf)['loss_tot']
        loss_va_run += lv.item() * curr_bs

        if ep == 1 and bi == 0:
            helpers.transform_tensor_to_ply_4(gt, n_directory, bi, "_GT", False)

        if bi == 0 and (ep - 1) % 50 == 0:
            helpers.transform_tensor_to_ply_4(input_damaged, n_directory, bi, "_input" + str(ep), False)
            helpers.transform_tensor_to_ply_4(model.im_pred, n_directory, bi, "_pred" + str(ep), False)

    loss_va = loss_va_run / len(ds_va)
    print(' ltot_va: {:.4f}'.format(loss_va), end='')
    scheduler.step(loss_va)

    # Save train state.
    saver(epoch=ep, iterations=it)

    if loss_va < bestVa:
        saverBest(epoch=ep, iterations=it)
        bestVa = loss_va

    print(' || total time : {:.0f} s'.format(timer() - tstart))

