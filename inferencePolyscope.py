# Python std
import argparse
from timeit import default_timer as timer
import math
import random

# project files
import helpers
from R2ATT_UNET import AUNet

# 3rd party
import torch
import numpy as np

import matplotlib.pyplot as plt
import helpers

import polyscope as ps
import polyscope.imgui as psim

def read_file(path, skip_lines=0):
    with open(path, 'r') as file:
        lines = file.readlines()[skip_lines:]
        return lines


def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))


def normalize(pc, mask):
    pc = pc.to(torch.device('cpu'))
    mask = mask.to(torch.device('cpu'))

    minX = torch.min(pc[0, :, :])
    minY = torch.min(pc[1, :, :])
    minZ = torch.min(pc[2, :, :])

    pc = torch.cat(
        ((pc[0, :, :] - minX).unsqueeze(0), (pc[1, :, :] - minY).unsqueeze(0), (pc[2, :, :] - minZ).unsqueeze(0)), 0)

    mask_x = torch.logical_and(torch.cat((mask[:, 1:], torch.ones((64, 1), dtype=torch.bool)), 1), mask)[:, :-1]
    mask_y = torch.logical_and(torch.cat((mask[1:, :], torch.ones((1, 64), dtype=torch.bool)), 0), mask)[:-1, :]

    diff_x = torch.cat((torch.ones(3, 64, 1), pc), 2) - \
             torch.cat((pc, torch.ones(3, 64, 1)), 2)
    diff_x_loss = diff_x[:, :, 1:-1].norm(p=2, dim=0).mean()
    final_x = mask_x * diff_x[:, :, 1:-1].norm(p=2, dim=0)
    final_x = final_x.sum() / torch.count_nonzero(final_x)

    diff_y = torch.cat((torch.ones(3, 1, 64), pc), 1) - \
             torch.cat((pc, torch.ones(3, 1, 64)), 1)
    diff_y_loss = diff_y[:, 1:-1, :].norm(p=2, dim=0).mean()
    final_y = mask_y * diff_y[:, 1:-1, :].norm(p=2, dim=0)
    final_y = final_y.sum() / torch.count_nonzero(final_y)

    diff = 0.5 * (final_y + final_x)
    pc = pc / (64 * diff)

    print("scale", 64 * diff)
    print("minX", minX)
    print("minY", minY)
    print("minZ", minZ)

    param = {"scale": 64 * diff, "minX": minX, "minY": minY, "minZ": minZ}

    return pc, param


def distanceLoss(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def computeDistortions(input_norm, mask_param, param):
    target_point = (0.322, 0.457, 0.225)
    cost = float('inf')
    pos = (0, 0)
    values = np.zeros(64 * 64)
    bestMesh = None
    for i in range(12, 50):
        for j in range(12, 50):
            if mask_param[i, j] == 1:
                continue
            mask_param[i, j] = 1
            input_net_in = input_norm * mask_param
            input_net_in[0, i, j] = target_point[0]
            input_net_in[1, i, j] = target_point[1]
            input_net_in[2, i, j] = target_point[2]

            data_in = torch.cat((input_net_in, mask_param.unsqueeze(0)), dim=0).to(torch.device("cuda"))
            data_in = data_in.unsqueeze(0)
            start = timer()
            model(data_in)
            #print(timer() - start)
            loss = distanceLoss(model.im_pred.squeeze(0).view(3, 4096).transpose(0, 1)[i * 64 + j], target_point)
            values[i * 64 + j] = loss
            if cost > loss:
                bestMesh = model.im_pred.squeeze(0).view(3, 4096).transpose(0, 1).detach().clone().cpu().numpy()
                cost = loss
                pos = (i, j)
            #print((i, j), loss)
            mask_param[i, j] = 0
    print("pos : " + str(pos[0]) + ", " + str(pos[1]))
    ref = bestMesh[pos[0] * 64 + pos[1]]
    print("Distance : ", math.sqrt(
        (ref[0] - target_point[0]) ** 2 + (ref[1] - target_point[1]) ** 2 + (ref[2] - target_point[2]) ** 2))
    psbest_cloud = ps.register_point_cloud("BestMatch", bestMesh)
    psbest_cloud.set_enabled(False)
    return values


def set_grab(input_n, mask_tmp):
    indices = [(32, 32)]
    for i, j in indices:
        mask_tmp[i, j] = 1
        input_n[0, i, j] = model.im_pred[0, 0, i, j]
        input_n[1, i, j] = model.im_pred[0, 1, i, j]
        input_n[2, i, j] = model.im_pred[0, 2, i, j]

    return input_n, mask_tmp

def set_pinch(input_n, mask_tmp):
    indices = [(28, 43), (28, 14)]
    for i, j in indices:
        mask_tmp[i, j] = 1
        input_n[0, i, j] = model.im_pred[0, 0, i, j]
        input_n[1, i, j] = model.im_pred[0, 1, i, j]
        input_n[2, i, j] = model.im_pred[0, 2, i, j]

    return input_n, mask_tmp


def set_twist(input_n, mask_tmp):
    for k in range(-e, e + 1, e):
        mask_tmp[32, 32] = 1
        mask_tmp[32, 32 + k] = 1
        mask_tmp[32 + k, 32] = 1
        input_n[0, 32, 32] = model.im_pred[0, 0, 32, 32]
        input_n[1, 32, 32] = model.im_pred[0, 1, 32, 32]
        input_n[2, 32, 32] = model.im_pred[0, 2, 32, 32]
        input_n[0, 32 + k, 32] = model.im_pred[0, 0, 32 + k, 32]
        input_n[1, 32 + k, 32] = model.im_pred[0, 1, 32 + k, 32]
        input_n[2, 32 + k, 32] = model.im_pred[0, 2, 32 + k, 32]
        input_n[0, 32, 32 + k] = model.im_pred[0, 0, 32, 32 + k]
        input_n[1, 32, 32 + k] = model.im_pred[0, 1, 32, 32 + k]
        input_n[2, 32, 32 + k] = model.im_pred[0, 2, 32, 32 + k]
    return input_n, mask_tmp


def grab(input_n, xyz):
    input_n[0, 32, 32] = x
    input_n[1, 32, 32] = y
    input_n[2, 32, 32] = z
    
    target_cloud.update_point_positions(np.array([[x, y, z]]))
    return input_n

def twist(input_n, xyz, thet):
    thet += 3.141592
    x, y, z = xyz
    for k in range(-e, e+1, e):
        input_n[0, 32, 32] = x
        input_n[1, 32, 32] = y
        input_n[2, 32, 32] = z
        input_n[0, 32 + k, 32] = x + k * 0.0329 * math.cos(thet)
        input_n[1, 32 + k, 32] = y
        input_n[2, 32 + k, 32] = z + k * 0.0329 * math.sin(thet)
        input_n[0, 32, 32 + k] = x + k * 0.0329 * math.sin(thet)
        input_n[1, 32, 32 + k] = y
        input_n[2, 32, 32 + k] = z - k * 0.0329 * math.cos(thet)

    target_cloud.update_point_positions(np.array([[x, y, z],
                                                  [x + (-e) * 0.0329 * math.cos(thet), y, z + (-e) * 0.0329 * math.sin(thet)],
                                                  [x + e * 0.0329 * math.cos(thet), y, z + e * 0.0329 * math.sin(thet)],
                                                  [x + (-e) * 0.0329 * math.sin(thet), y, z - (-e) * 0.0329 * math.cos(thet)],
                                                  [x + e * 0.0329 * math.sin(thet), y, z - e * 0.0329 * math.cos(thet)]]))
    return input_n


def pinch(input_n, pinch_pos, dist):
    indices = [(28, 43), (28, 14)]
    mid_point = (0.5*(pinch_pos[0][0] + pinch_pos[1][0]), 0.5*(pinch_pos[0][1] + pinch_pos[1][1]), 0.5*(pinch_pos[0][2] + pinch_pos[1][2]))

    input_n[0, indices[0][0], indices[0][1]] = dist * pinch_pos[0][0] + (1 - dist) * mid_point[0]
    input_n[1, indices[0][0], indices[0][1]] = dist * pinch_pos[0][1] + (1 - dist) * mid_point[1]
    input_n[2, indices[0][0], indices[0][1]] = dist * pinch_pos[0][2] + (1 - dist) * mid_point[2]

    input_n[0, indices[1][0], indices[1][1]] = dist * pinch_pos[1][0] + (1 - dist) * mid_point[0]
    input_n[1, indices[1][0], indices[1][1]] = dist * pinch_pos[1][1] + (1 - dist) * mid_point[1]
    input_n[2, indices[1][0], indices[1][1]] = dist * pinch_pos[1][2] + (1 - dist) * mid_point[2]
    aa = (dist * pinch_pos[0][0] + (1 - dist) * mid_point[0]).detach().clone().cpu()
    ab = (dist * pinch_pos[0][1] + (1 - dist) * mid_point[1]).detach().clone().cpu()
    ac = (dist * pinch_pos[0][2] + (1 - dist) * mid_point[2]).detach().clone().cpu()
    ad = (dist * pinch_pos[1][0] + (1 - dist) * mid_point[0]).detach().clone().cpu()
    ae = (dist * pinch_pos[1][1] + (1 - dist) * mid_point[1]).detach().clone().cpu()
    af = (dist * pinch_pos[1][2] + (1 - dist) * mid_point[2]).detach().clone().cpu()
    target_cloud.update_point_positions(np.array([[aa, ab, ac], [ad, ae, af]]))
    return input_n

def gaussian_2d(tensor_size, center, sigma=1.0):
    x, y = torch.meshgrid(torch.arange(tensor_size[0]), torch.arange(tensor_size[1]))
    x_center, y_center = center
    exponent = -((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * sigma ** 2)
    return torch.exp(exponent)

def minimize(target, uv, input_model, mask_model):
    target_tensor = target.view(-1, 1, 1)
    target_tensor = target_tensor.to('cuda')
    mask_cache = mask_model.detach().clone().to('cuda')
    u, v = uv
    visited = {}
    input_model_gpu = input_model.to('cuda')
    step_size = 25.0
    last_loss = 10000000
    last_grad = (0.0, 0.0)
    last_uv = (u, v)

    while True:
        coordinates = torch.tensor([float(u), float(v)], requires_grad=True)
        soft_mask = 0.999 * gaussian_2d((64, 64), center=coordinates, sigma=0.5)
        soft_mask = soft_mask.to('cuda')

        mask_tmp = mask_cache.clone()
        mask_tmp[u, v] = 1
        data_input = input_model_gpu * (1 - soft_mask).expand(3, 64, 64) + soft_mask * target_tensor
        data_input_model = data_input * mask_tmp
        data_input_model = torch.cat((data_input_model, mask_tmp.unsqueeze(0)), dim=0).to(torch.device("cuda"))
        data_input_model = data_input_model.unsqueeze(0)
        model(data_input_model)

        loss = torch.norm(model.im_pred.squeeze(0) * soft_mask.expand(3, 64, 64) - soft_mask * target_tensor)

        loss.backward(retain_graph=True)
        if last_loss < loss:
            (u, v) = last_uv
            step_size /= 2
            continue
        last_loss = loss

        normalized_grad = coordinates.grad / torch.norm(coordinates.grad)
        if (step_size * torch.norm(coordinates.grad) < 1.0):
            grad_u, grad_v = normalized_grad
        last_uv = (u, v)
        u = int(torch.round(u - step_size * grad_u).item())
        v = int(torch.round(v - step_size * grad_v).item())
        print((u, v), loss, coordinates.grad)
        if not (u, v) in visited:
            visited[(u, v)] = loss
        else:
            break

    minLoss = float('inf')
    bestPos = (0, 0)
    path = np.zeros(64 * 64)
    for coord in visited:
        path[coord[0] * 64 + coord[1]] = 1
        if visited[coord] < minLoss:
            minLoss = visited[coord]
            bestPos = coord

    ps_cloud.add_scalar_quantity("Path", path)
    print("Best position : ", bestPos)


def get_i_j(a, N):
    return a // N, a % N


def get_border_indices(mesh_x):
    N, _ = mesh_x.shape
    group = set([0, 63, 4032, 4095])
    to_visit = set([1, N])
    helpers.add_neighbours_idx(0, 63, N, to_visit, group)
    helpers.add_neighbours_idx(63, 0, N, to_visit, group)
    helpers.add_neighbours_idx(63, 63, N, to_visit, group)
    while to_visit:
        index = to_visit.pop()
        i, j = get_i_j(index, N)
        if mesh_x[i, j]:
            continue
        helpers.add_neighbours_idx(i, j, N, to_visit, group)
        group.add(index)
    in_v = []
    for i in range(N*N):
        if i not in group:
            in_v.append(i)
    return in_v

""" Parse args """
parser = argparse.ArgumentParser()
parser.add_argument('weights', type=str, help='Filepath to model weights')
parser.add_argument('datapath', type=str, help='Filepath to input data')
parser.add_argument('editingType', type=str, nargs='?', default='grab', help='Pretty Huge Depression')
args = parser.parse_args()

""" Initialization / Load model / Normalization """
ps.init()
force_cudnn_initialization()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N = 64

model = AUNet().to(device)

model.load_state_dict(torch.load(args.weights)['weights'])

# Validation.
model.eval()

# Not database
file = read_file(args.datapath)
point_set = np.loadtxt(file).astype(np.float32)
input_damaged = torch.from_numpy(point_set[:, :3])
input_damaged = input_damaged.transpose(0, 1)
input_damaged = input_damaged.view(3, 64, 64)

mask = (input_damaged[0] != 0) | (input_damaged[1] != 0) | (input_damaged[2] != 0)
in_indices = get_border_indices(mask)

input_normalized, param = normalize(input_damaged, mask)


""" Network Pass """
input_net = input_normalized * mask

arr = input_net.view(3, 4096).transpose(0, 1).detach().clone().cpu().numpy()
nonzero_columns_mask = ~np.all(arr == 0, axis=1)
ps_cloud = ps.register_point_cloud("Init", arr[nonzero_columns_mask, :])
#ps_cloud = ps.register_point_cloud("Init", tmp.view(3, 4096).transpose(0, 1).detach().clone().cpu().numpy())
helpers.write_ply_file(input_net.view(3, 4096).transpose(0, 1).detach().clone().cpu().numpy(), "input.ply")
data = torch.cat((input_net, mask.unsqueeze(0)), dim=0).to(torch.device("cuda"))
data = data.unsqueeze(0)
start = timer()
model(data)
end = timer()

e = 8
if args.editingType == 'grab':
    input_normalized, mask = set_grab(input_normalized, mask)
elif args.editingType == 'twist':
    input_normalized, mask = set_twist(input_normalized, mask)
elif args.editingType == 'pinch':
    input_normalized, mask = set_pinch(input_normalized, mask)

input_net = input_normalized * mask


data = torch.cat((input_net, mask.unsqueeze(0)), dim=0).to(torch.device("cuda"))
data = data.unsqueeze(0)
model(data)
print("timer : ", end - start)

pts = model.im_pred.squeeze(0)
pts = pts.view(3, 4096).transpose(0, 1).detach().clone().cpu().numpy()
faces = np.zeros((7938, 3))
for i in range(N - 1):
    for j in range(N - 1):
        faces[2 * ((N - 1) * i + j)] = [N * i + j, N * (i + 1) + j, N * i + j + 1]
        faces[2 * ((N - 1) * i + j) + 1] = [N * (i + 1) + j, N * (i + 1) + j + 1, N * i + j + 1]

ps_cloud = ps.register_point_cloud("Prediction pc", pts)
ps_mesh = ps.register_surface_mesh("Prediction mesh", pts, faces)
start = timer()
distortions = computeDistortions(input_normalized, mask, param)
end = timer()
print("Timing brute force : ", end - start)
ps_mesh.add_scalar_quantity("Distortion", distortions)

selected_u = 40
selected_v = 40
x = model.im_pred[0, 0, 32, 32]
y = model.im_pred[0, 1, 32, 32]
z = model.im_pred[0, 2, 32, 32]
theta = 0.0
d = 1.0
init_pinch = ((model.im_pred[0, 0, 28, 14], model.im_pred[0, 1, 28, 14], model.im_pred[0, 2, 28, 14]),
              (model.im_pred[0, 0, 28, 43], model.im_pred[0, 1, 28, 43], model.im_pred[0, 2, 28, 43]))

if args.editingType == 'grab':
    target_cloud = ps.register_point_cloud("Target", np.array([[x.item(), y.item(), z.item()]]))
elif args.editingType == 'twist':
    target_cloud = ps.register_point_cloud("Target", np.array([[0.322, 0.457, 0.225], [0.322, 0.457, 0.225], [0.322, 0.457, 0.225], [0.322, 0.457, 0.225], [0.322, 0.457, 0.225]]))
elif args.editingType == 'pinch':
    target_cloud = ps.register_point_cloud("Target", np.array([[0.322, 0.457, 0.225], [0.322, 0.457, 0.225]]))

def callback():
    global selected_u, selected_v, input_normalized, input_net, model, x, y, z, theta, d
    changed_x, x = psim.SliderFloat("change x", x, v_min=-1, v_max=2)
    changed_y, y = psim.SliderFloat("change y", y, v_min=-1, v_max=1)
    changed_z, z = psim.SliderFloat("change z", z, v_min=-1, v_max=2)
    changed_theta, theta = psim.SliderFloat("change theta", theta, v_min=-1.4, v_max=1.4)
    changed_d, d = psim.SliderFloat("change dist", d, v_min=-1.0, v_max=1.0)


    if changed_x or changed_y or changed_z or changed_theta or changed_d:
        if args.editingType == 'grab':
            input_normalized = grab(input_normalized, (x, y, z))
        elif args.editingType == 'twist':
            input_normalized = twist(input_normalized, (x, y, z), theta)
        elif args.editingType == 'pinch':
            input_normalized = pinch(input_normalized, init_pinch, d)

        input_net = input_normalized * mask
        data = torch.cat((input_net, mask.unsqueeze(0)), dim=0).to(torch.device("cuda"))
        data = data.unsqueeze(0)
        model(data)
        ps_cloud.update_point_positions(
            model.im_pred.squeeze(0).view(3, 4096).transpose(0, 1).detach().clone().cpu().numpy())
        ps_mesh.update_vertex_positions(
            model.im_pred.squeeze(0).view(3, 4096).transpose(0, 1).detach().clone().cpu().numpy())


    if psim.Button("Minimize Distortion"):

        start = timer()
        minimize(torch.tensor([0.322, 0.457, 0.225]), (selected_u, selected_v), input_normalized, mask)
        end = timer()
        print("Timing GD : ", end - start)


ps.set_user_callback(callback)
ps.show()
ps.clear_user_callback()
