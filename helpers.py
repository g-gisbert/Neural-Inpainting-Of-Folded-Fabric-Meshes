"""
Original author: Jan Bednarik
Adapated: Guillaume Gisbert
"""

# 3rd party.
import torch
import numpy as np
from math import sin
from plyfile import PlyData, PlyElement
import random

# Python std.
import yaml
import os
import re


class Device:
    """ Creates the computation device.

    Args:
        gpu (bool): Whether to use gpu.
    """

    def __init__(self, gpu=True):
        if gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            if gpu:
                print('[WARNING] cuda not available, using CPU.')


class TrainStateSaver:

    def __init__(self, path_file, model=None, optimizer=None, scheduler=None,
                 verbose=False):
        self._path_file = path_file
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._verbose = verbose

        if not os.path.exists(os.path.dirname(path_file)):
            raise Exception('Path "{}" does not exist.'.format(path_file))

        for var, name in zip([model, optimizer, scheduler],
                             ['model', 'optimizer', 'scheduler']):
            if var is None:
                print('[WARNING] TrainStateSaver: {} is None and will not be '
                      'saved'.format(name))

    def get_file_path(self):
        return self._path_file

    def __call__(self, file_path_override=None, **kwargs):
        state = kwargs
        if self._model:
            state['weights'] = self._model.state_dict()
        if self._optimizer:
            state['optimizer'] = self._optimizer.state_dict()
        if self._scheduler:
            state['scheduler'] = self._scheduler.state_dict()

        # Get the output file path and save.
        path_file = (file_path_override,
                     self._path_file)[file_path_override is None]
        pth_tmp = path_file

        # Try to save the file.
        try:
            torch.save(state, pth_tmp)
        except Exception as e:
            print('ERROR: The model weights file {} could not be saved and '
                  'saving is skipped. The exception: "{}"'.
                  format(pth_tmp, e))
            if os.path.exists(pth_tmp):
                os.remove(pth_tmp)
            return

        if self._verbose:
            print('[INFO] Saved training state to {}'.format(path_file))


class RunningLoss:
    def __init__(self):
        self.reset()

    def update(self, **kwargs):
        for k, v in kwargs.items():
            it = self._its.get(k, 0)
            self._data[k] = self._data.get(k, 0) * (it / (it + 1)) + v / (it + 1)
            self._its[k] = it + 1

    def reset(self):
        self._data = {}
        self._its = {}

    def get_losses(self):
        return self._data.copy()


def identity(x):
    """ Implements the identity function.
    """
    return x


def load_conf(path):
    """ Returns the loaded .cfg config file.

    Args:
        path (str): Aboslute path to .cfg file.

    Returns:
    dict: Loaded config file.
    """

    with open(path, 'r') as f:
        conf = yaml.full_load(f)
    return conf


def jn(*parts):
    """ Returns the file system path composed of `parts`.

    Args:
        *parts (str): Path parts.

    Returns:
        str: Full path.
    """
    return os.path.join(*parts)


def ls(path, exts=None, pattern_incl=None, pattern_excl=None,
       ignore_dot_underscore=True):
    """ Lists the directory and returns it sorted. Only the files with
    extensions in `ext` are kept. The output should match the output of Linux
    command "ls". It wraps os.listdir() which is not guaranteed to produce
    alphanumerically sorted items.

    Args:
        path (str): Absolute or relative path to list.
        exts (str or list of str or None): Extension(s). If None, files with
            any extension are listed. Each e within `exts` can (but does
            not have to) start with a '.' character. E.g. both
            '.tiff' and 'tiff' are allowed.
        pattern_incl (str): regexp pattern, if not found in the file name,
            the file is not listed.
        pattern_excl (str): regexp pattern, if found in the file name,
            the file is not listed.
        ignore_dot_underscore (bool): Whether to ignore files starting with
            '._' (usually spurious files appearing after manipulating the
            linux file system using sshfs)

    Returns:
        list of str: Alphanumerically sorted list of files contained in
        directory `path` and having extension `ext`.
    """
    if isinstance(exts, str):
        exts = [exts]

    files = [f for f in sorted(os.listdir(path))]

    if exts is not None:
        # Include patterns.
        extsstr = ''
        for e in exts:
            extsstr += ('.', '')[e.startswith('.')] + '{}|'.format(e)
        patt_ext = '({})$'.format(extsstr[:-1])
        re_ext = re.compile(patt_ext)
        files = [f for f in files if re_ext.search(f)]

    if ignore_dot_underscore:
        re_du = re.compile('^\._')
        files = [f for f in files if not re_du.match(f)]

    if pattern_incl is not None:
        re_incl = re.compile(pattern_incl)
        files = [f for f in files if re_incl.search(f)]

    if pattern_excl is not None:
        re_excl = re.compile(pattern_excl)
        files = [f for f in files if not re_excl.search(f)]

    return files


def lsd(path, pattern_incl=None, pattern_excl=None):

    if pattern_incl is None and pattern_excl is None:
        dirs = [d for d in ls(path) if os.path.isdir(jn(path, d))]
    else:
        if pattern_incl is None:
            pattern_incl = '^.'
        if pattern_excl is None:
            pattern_excl = '^/'

        pincl = re.compile(pattern_incl)
        pexcl = re.compile(pattern_excl)
        dirs = [d for d in ls(path) if
                os.path.isdir(jn(path, d)) and
                pincl.search(d) is not None and
                pexcl.search(d) is None]

    return dirs


def load_obj(pth):
    """ Loads mesh from .obj file.

    Args:
        pth (str): Absolute path.

    Returns:
        np.array (float): Mesh, (V, 3).
        np.array (int32): Triangulation, (T, 3).
    """
    with open(pth, 'r') as f:
        lines = f.readlines()

    mesh = []
    tri = []
    normals = []
    for l in lines:
        # vals = l.split(' ')
        vals = l.split()
        if len(vals) > 0:
            if vals[0] == 'v':
                mesh.append([float(n) for n in vals[1:]])
            elif vals[0] == 'f':
                tri.append([int(n.split('/')[0]) - 1 for n in vals[1:]])
            elif vals[0] == 'vn':
                normals.append([float(n) for n in vals[1:]])

    mesh = np.array(mesh, dtype=np.float32)
    tri = np.array(tri, dtype=np.int32)
    normals = np.array(normals, dtype=np.float32)

    return mesh, tri, normals


def mesh_area(verts, faces):
    """ Computes the area of the surface as a sum of areas of all the triangles.

    Args:
        verts (np.array of float): Vertices, shape (V, 3).
        faces (np.array of int32): Faces, shape (F, 3)

    Returns:
        float: Total area [m^2].
    """
    assert verts.ndim == 2 and verts.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3

    v1s, v2s, v3s = verts[faces.flatten()]. \
        reshape((-1, 3, 3)).transpose((1, 0, 2))  # each (F, 3)
    v21s = v2s - v1s
    v31s = v3s - v1s
    return np.sum(0.5 * np.linalg.norm(np.cross(v21s, v31s), axis=1))


def write_ply_file(points, filename):
    #vertex = np.core.records.fromarrays(points.transpose(), names='x, y, z', formats='f4, f4, f4')
    #ply_data = PlyData([PlyElement.describe(vertex, 'vertex')], text=True)
    #ply_data.write(filename)
    data = ""
    for i in range(0, len(points)):
        data += str(points[i][0]) + " " + str(points[i][1]) + " " + str(points[i][2]) + "\n"
    with open(filename, "w") as f:
        f.write(data[:-1])


def transform_tensor_to_ply_4(tensor, n, bi, end, verbose=True):
    B, K, N1, N2 = tensor.shape
    tensor = tensor.reshape(B, K, N1 * N2).transpose(1, 2)
    transform_tensor_to_ply(tensor, n, bi, end, verbose)


def transform_tensor_to_ply(tensor, n, bi, end, verbose=True):
    batch_size, num_points, _ = tensor.shape

    for i in range(batch_size):
        points = tensor[i].detach().cpu().numpy()
        filename = f'/home/guillaume/Documents/data/experiments/exp{n}/file_{batch_size * bi + i}{end}.ply'
        write_ply_file(points, filename)
        if verbose:
            print(f'PLY file {filename}{end} saved.')


def save_to_npy(data, num, in_out):
    data0, data1 = np.split(data, 2, axis=0)
    data0 = np.squeeze(data0)
    data1 = np.squeeze(data1)
    np.save("/home/guillaume/Documents/data/experiments/exp2/" + str(num) + "_" + in_out + ".npy", data0)
    np.save("/home/guillaume/Documents/data/experiments/exp2/" + str(num + 1) + "_" + in_out + ".npy", data1)


def index_points(point_clouds, index):
    """
    Given a batch of tensor and index, select sub-tensor.

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, N, k]
    Return:
        new_points:, indexed points data, [B, N, k, C]
    """
    device = point_clouds.device
    batch_size = point_clouds.shape[0]
    view_shape = list(index.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(index.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = point_clouds[batch_indices, index, :]
    return new_points


def knn(x, k):
    """
    K nearest neighborhood.

    Parameters
    ----------
        x: a tensor with size of (B, C, N)
        k: the number of nearest neighborhoods

    Returns
    -------
        idx: indices of the k nearest neighborhoods with size of (B, N, k)
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # (B, N, N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # (B, 1, N)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (B, 1, N), (B, N, N), (B, N, 1) -> (B, N, N)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)
    return idx


def sample_mesh(mesh, tri, normals, n):
    epsilon = 10e-4
    N_line = n
    xx = np.linspace(-0.3 + epsilon, 0.3 - epsilon, N_line, dtype=np.float32)
    yy = np.linspace(-0.3 + epsilon, 0.3 - epsilon, N_line, dtype=np.float32)
    grid = np.meshgrid(xx, yy)
    grid = torch.Tensor(np.array(grid)).view(2, -1)  # (2, n*n)
    grid = grid.transpose(0, 1)
    grid = grid.unsqueeze(0)

    # 2D samples to triangles id
    B = 1
    M = grid.shape[1]
    a = 0.3  # 0.3/(0.75-0.5)  # 0.3
    b = 2*a  # 0.6
    scaled_uv = 50.0 * (grid + a) / b # 50.0 * (grid + a) / b
    scaled_uv_decimal = scaled_uv - torch.floor(scaled_uv)
    lower_triangle = ((scaled_uv_decimal[..., 0] + scaled_uv_decimal[..., 1])) > 1
    triangles_id = (torch.floor(scaled_uv) * torch.tensor([1, 50])).sum(dim=2)
    triangles_id[lower_triangle] += 2500 # 2500
    triangles_id = triangles_id.to(torch.int64)

    tri_id_cpu = triangles_id.to(torch.device('cpu')).detach().numpy()

    tri = torch.from_numpy(tri).unsqueeze(0)
    mesh = torch.from_numpy(mesh).unsqueeze(0)
    #normals = torch.from_numpy(normals).unsqueeze(0)

    # Barycentric coordinates
    idx = tri.gather(1, triangles_id.unsqueeze(-1).expand(-1, -1, 3)).type(torch.int64)
    triangle_vertices = torch.zeros(B, M, 3, 3)
    #triangle_normals = torch.zeros(B, M, 3, 3)
    scaled_uv_decimal = scaled_uv_decimal.detach().cpu()
    positions_3d = torch.zeros(B, M, 3)
    normals_3D = torch.zeros(B, M, 3)
    for batch_idx in range(B):
        triangle_vertices[batch_idx] = mesh[batch_idx, idx[batch_idx]]
        #triangle_normals[batch_idx] = normals[batch_idx, idx[batch_idx]]

        v0 = torch.zeros(M, 2)
        v1 = torch.zeros(M, 2)
        v0[~lower_triangle[batch_idx, ...]] = torch.tensor([0., 1.]) - torch.tensor([1., 0.])  # B - A
        v0[lower_triangle[batch_idx, ...]] = torch.tensor([1., 1.]) - torch.tensor([1., 0.])
        v1[~lower_triangle[batch_idx, ...]] = torch.tensor([0., 0.]) - torch.tensor([1., 0.])  # C - A
        v1[lower_triangle[batch_idx, ...]] = torch.tensor([0., 1.]) - torch.tensor([1., 0.])
        v2 = scaled_uv_decimal[batch_idx, :, :] - torch.tensor([1., 0.])

        d00 = (v0 * v0).sum(dim=1)
        d01 = (v0 * v1).sum(dim=1)
        d11 = (v1 * v1).sum(dim=1)
        d20 = (v2 * v0).sum(dim=1)
        d21 = (v2 * v1).sum(dim=1)
        denom = d00 * d11 - d01 * d01

        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        triangle_vertices_cpu = triangle_vertices.to(torch.device('cpu')).detach().numpy()

        positions_3d[batch_idx, :, :] = (
                triangle_vertices[batch_idx, :, 0, :] * u.unsqueeze(1) + triangle_vertices[batch_idx, :, 1,
                                                                         :] * v.unsqueeze(1) +
                triangle_vertices[batch_idx, :, 2, :] * w.unsqueeze(1))
        """normals_3D[batch_idx, :, :] = (
                triangle_normals[batch_idx, :, 0, :] * u.unsqueeze(1) + triangle_normals[batch_idx, :, 1,
                                                                         :] * v.unsqueeze(1) +
                triangle_normals[batch_idx, :, 2, :] * w.unsqueeze(1))
    normals_3D = torch.nn.functional.normalize(normals_3D, dim=2)"""

    return positions_3d.squeeze(0)#, normals_3D.squeeze(0)


def add_neighbours(x: int, y: int, size: int, s: set, visited: set):
    def check_and_add(value):
        if not(value in visited):
            s.add(value)

    if x > 0:
        check_and_add((x-1, y))
    if x < size-1:
        check_and_add((x+1, y))
    if y > 0:
        check_and_add((x, y-1))
    if y < size-1:
        check_and_add((x, y+1))


def add_neighbours_idx(x: int, y: int, size: int, s: set, visited: set):
    def check_and_add(value):
        if not(value in visited):
            s.add(value)

    if x > 0:
        check_and_add((x-1)*size + y)
    if x < size-1:
        check_and_add((x+1)*size + y)
    if y > 0:
        check_and_add(x*size + y-1)
    if y < size-1:
        check_and_add(x*size + y+1)


def compute_normals(x):  # x :shape(B, 3, H, W)
    x_kernel = torch.tensor([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(3, 3, -1, -1).to(torch.device('cuda'))
    y_kernel = torch.tensor([[0, 1, 0], [0, 0, 0], [0, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(3, 3, -1, -1).to(torch.device('cuda'))
    replicate_padding = torch.nn.ReplicationPad2d(1)
    data = replicate_padding(x)
    b = torch.nn.functional.conv2d(data, x_kernel)
    c = torch.nn.functional.conv2d(data, y_kernel)
    normal = torch.cross(b, c, dim=1)

    return normal


def gaussian_curvature(x): # x :shape(1, 3, H, W)
    B, C, H, W = x.shape
    for i in range(1, H-1):
        for j in range(1, W-1):
            pass


def sin_input(x):  # x : shape(3, H, W)
    for i in range(0, 64):
        for j in range(0, 25):
            x[0, j, i] += 0.1*sin(4.0 * 3.141592 * i / 64.0)
        for j in range(25, 40):
            x[0, j, i] = 0.0
            x[1, j, i] = 0.0
            x[2, j, i] = 0.0
        for j in range(40, 64):
            x[0, j, i] += 0.1*sin(4.0 * 3.141592 * i / 64.0 + 3.141592)
    return x


def random_rotation_matrix():
    angle = random.uniform(0, 2 * 3.141592653589793)
    cos_theta = torch.cos(torch.tensor(angle))
    sin_theta = torch.sin(torch.tensor(angle))

    rotation_matrix = torch.tensor([[cos_theta, -sin_theta, 0],
                                    [sin_theta, cos_theta, 0],
                                    [0, 0, 1]])

    angle = random.uniform(0, 2 * 3.141592653589793)
    cos_theta = torch.cos(torch.tensor(angle))
    sin_theta = torch.sin(torch.tensor(angle))

    rotation_matrix = torch.tensor([[cos_theta, 0, sin_theta],
                                    [0, 1, 0],
                                    [-sin_theta, 0, cos_theta]]) @ rotation_matrix

    angle = random.uniform(0, 2 * 3.141592653589793)
    cos_theta = torch.cos(torch.tensor(angle))
    sin_theta = torch.sin(torch.tensor(angle))

    rotation_matrix = torch.tensor([[1, 0, 0],
                                    [0, cos_theta, -sin_theta],
                                    [0, sin_theta, cos_theta]]) @ rotation_matrix

    return rotation_matrix

def apply_random_rotation(pc):
    rot_matrix = random_rotation_matrix()
    tmp = rot_matrix @ pc.view(3, 64*64)
    return tmp.view(3, 64, 64)

""" Regu normals
    n_pref = compute_normals(self.im_pred)
    torch.nn.functional.normalize(n_pref, dim=2)
    x_kernel = torch.tensor([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1).to(torch.device('cuda'))
    y_kernel = torch.tensor([[0, 1, 0], [0, 0, 0], [0, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1).to(torch.device('cuda'))
    replicate_padding = torch.nn.ReplicationPad2d(1)
    data = replicate_padding(n_pref)
    b = torch.nn.functional.conv2d(data, x_kernel)
    c = torch.nn.functional.conv2d(data, y_kernel)
    loss_normals = b.mean() + c.mean()
    losses['loss_normals'] = beta * (loss_normals * loss_normals)**2
"""

def compute_mean_curvature(x): # x :shape(B, 3, H, W)
    B, _, H, W = x.shape
    x = x.to('cuda')
    laplacian_kernel = torch.tensor([[0, -1, 0],
                                     [-1, 4, -1],
                                     [0, -1, 0]], dtype=torch.float32, device='cuda')

    laplacian_kernel = laplacian_kernel.expand(B, 3, 3, 3)
    laplacian_output = torch.abs(torch.nn.functional.conv2d(x, laplacian_kernel, padding=1))

    mean_curvature = laplacian_output[:, :, 1:-1, 1:-1].sum() / 2.0

    return mean_curvature


def write_obj(points, faces, filename):
    num_rows = 64
    num_cols = 64

    u_coords = np.linspace(0, 1, num_cols).reshape(1, -1)
    v_coords = np.linspace(0, 1, num_rows).reshape(-1, 1)

    u_coords = np.tile(u_coords, (num_rows, 1))
    v_coords = np.tile(v_coords, (1, num_cols))

    texture_coords = np.stack((u_coords.flatten(), v_coords.flatten()), axis=1)

    with open(filename, 'w') as file:
        # Write the vertices
        for vertex, tex_coord in zip(points, texture_coords):
            file.write(f"v {' '.join(map(str, vertex))}\n")

        # Write the texture coordinates
        for tex_coord in texture_coords:
            file.write(f"vt {' '.join(map(str, tex_coord))}\n")

        # Write the faces
        for face in faces:
            face_vertices = [f"{int(idx) + 1}/{int(idx) + 1}" for idx in face]
            file.write("f " + " ".join(face_vertices) + "\n")


def save_to_file(tensor, params, filename):

    flattened_tensor = tensor.flatten()

    with open(filename, 'w') as file:
        ite = 0
        for number in flattened_tensor:
            if ite % 3 == 0 :
                add = params['minX']
            if ite % 3 == 1:
                add = params['minY']
            if ite % 3 == 2 :
                add = params['minZ']
            final_val = float(params['scale']) * number + float(add)
            file.write("%s\n" % final_val)
            ite += 1
