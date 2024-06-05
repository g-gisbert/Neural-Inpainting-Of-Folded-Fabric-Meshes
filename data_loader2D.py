# Python std.
import os
import random


# 3rd party.
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import helpers
# Project files.
from helpers import load_obj, sample_mesh, add_neighbours, apply_random_rotation



def normalize(pc, N):
    minX = torch.min(pc[0, :, :])
    minY = torch.min(pc[1, :, :])
    minZ = torch.min(pc[2, :, :])
    pc = torch.cat(
        ((pc[0, :, :] - minX).unsqueeze(0), (pc[1, :, :] - minY).unsqueeze(0), (pc[2, :, :] - minZ).unsqueeze(0)), 0)

    diff_x = torch.cat((torch.ones(3, N, 1), pc), 2) - \
             torch.cat((pc, torch.ones(3, N, 1)), 2)
    diff_x_loss = diff_x[:, :, 1:-1].norm(p=2, dim=0).mean()
    diff_y = torch.cat((torch.ones(3, 1, N), pc), 1) - \
             torch.cat((pc, torch.ones(3, 1, N)), 1)
    diff_y_loss = diff_y[:, 1:-1, :].norm(p=2, dim=0).mean()

    diff = 0.5 * (diff_y_loss + diff_x_loss)
    pc = pc / (N * diff)

    return pc


class SF_Dataset(Dataset):

    def __init__(self, train=True):
        super(SF_Dataset, self).__init__()

        self.datapath = []
        self.data = {}
        self.masks = {}
        self.n_masks = 5000
        self.N = 64

        
        if train:
            data_path = "./ScarfFolds/train"
        else:
            self.n_masks = 500
            data_path = "./ScarfFolds/test"

        
        fns_pc = sorted(os.listdir(data_path))

        fns = [val for val in fns_pc]
        print('Files ' + str(len(fns)))

        if len(fns) == 0:
            print("ERRROR : folder is empty")

        for fn in fns:
            self.datapath.append(os.path.join(data_path, fn))

        # Masks
        for i in range(self.n_masks):
            N = 64
            if i % 50 == 0:
                print(f"Preprocessing masks: {i + 1}/{self.n_masks}", end='\r')
            mask = torch.ones(N, N)
            source_x, source_y = random.randint(15, N - 1 - 15), random.randint(15, N - 1 - 15)
            type = random.uniform(0.0, 1.0)

            if type < 0.5:
                # Circle
                radius_small = random.randint(20, 40) #random.randint(10, 20)
                radius_big = random.randint(50, 70) #random.randint(25, 35)
                for a in range(N):
                    for b in range(N):
                        if (a - source_x)**2 + (b - source_y)**2 < radius_small**2 or (a - source_x)**2 + (b - source_y)**2 > radius_big**2:
                            mask[a, b] = 0

            else:
                # Classic hole
                n_missing = random.randint(1200, 2400) #2400, 4800

                mask[source_x, source_y] = 0
                visited = set()
                neighbours = set()
                visited.add((source_x, source_y))
                add_neighbours(source_x, source_y, N, neighbours, visited)
                for j in range(n_missing):
                    x, y = random.choice(tuple(neighbours))
                    visited.add((x, y))
                    neighbours.remove((x, y))
                    mask[x, y] = 0
                    add_neighbours(x, y, N, neighbours, visited)

            if random.uniform(0.0, 1.0) > 0.5:
                mask[source_x, source_y] = 1
            self.masks[i] = mask.numpy()
	
        print()
        # Cache data
        for i, path in enumerate(self.datapath):
            if i % 50 == 0:
                print(f"Caching data: {i + 1}/{len(self.datapath)}", end='\r')
            mesh, tri, normals = load_obj(path)
            pc_gt = sample_mesh(mesh, tri, normals, N)
            normals_gt = torch.zeros(N*N, 3)
            pc_gt = pc_gt.transpose(0, 1).view(3, N, N)
            normals_gt = normals_gt.transpose(0, 1).view(3, N, N)
            pc_gt = apply_random_rotation(pc_gt)
            data = normalize(pc_gt, N)
            self.data[i] = (data, normals_gt)
        print()

    def __getitem__(self, index):

        pc_gt, normals_gt = self.data[index]
        mask = random.choice(self.masks)
        im_input = (pc_gt * mask).detach()
        normals_input = (normals_gt * mask).detach()

        sample = {'pcloud': pc_gt, 'normals_gt': normals_gt, 'normals_input': normals_input,
                  'input': im_input, 'mask': mask}

        return sample

    def __len__(self):
        return len(self.datapath)


