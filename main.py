import os
import argparse
from solver import Solver
import h5py
from torch.backends import cudnn
from torch.utils import data
import torch
import numpy as np
import shutil
from scipy.spatial.transform import Rotation as R


def str2bool(v):
    return v.lower() in ('true')


class dataset_facescape(data.Dataset):
    def __init__(self, path, random_scale=True, random_rot=True):
        self.file_path = path
        self.shape_data = None
        self.label_data = None
        # load pre-defined grid
        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file['shape'])
            self.ix = torch.LongTensor(np.round(file['IX'])).squeeze(1)
            self.iy = torch.LongTensor(np.round(file['IY'])).squeeze(1)
            self.iz = torch.LongTensor(np.round(file['IZ'])).squeeze(1)
            self.wx = torch.FloatTensor(file['WX'])
            self.wy = torch.FloatTensor(file['WY'])
            self.wz = torch.FloatTensor(file['WZ'])
            self.coordxy = 2 * torch.FloatTensor(file['gridxy']) / 127. - 1
            self.random_scale = random_scale
            self.random_rot = random_rot

    def __getitem__(self, index):
        if self.shape_data is None or self.label_data is None:
            data_loader = h5py.File(self.file_path, 'r')
            self.shape_data = data_loader['shape']
            self.label_data = data_loader['label']
        # data augmentation with random scaling and rotation
        if self.random_scale:
            scale = (0.9 + 0.2 * np.random.rand())
        else:
            scale = 1.
        if self.random_rot:
            random_Euler = -10 + np.random.rand(3) * 20
            r = R.from_euler('xyz', random_Euler, degrees=True).as_matrix()
        else:
            r=np.identity(3)
        return torch.FloatTensor(np.dot(self.shape_data[index] * scale, r)), torch.FloatTensor(self.label_data[index]), torch.FloatTensor(r), torch.FloatTensor([scale])

    def __len__(self):
        return self.dataset_len

    def all_grid_item(self):
        return self.ix, self.iy, self.iz, self.wx, self.wy, self.wz, self.coordxy


def main(config):
    # Create directories if not exist.
    if not os.path.exists(os.path.join(config.base_path + config.log_dir)):
        os.makedirs(os.path.join(config.base_path + config.log_dir))
    if not os.path.exists(os.path.join(config.base_path, config.model_save_dir)):
        os.makedirs(os.path.join(config.base_path, config.model_save_dir))
    if not os.path.exists(os.path.join(config.base_path, config.sample_dir)):
        os.makedirs(os.path.join(config.base_path, config.sample_dir))
    if not os.path.exists(os.path.join(config.base_path, config.result_dir)):
        os.makedirs(os.path.join(config.base_path, config.result_dir))

    # save training codes
    d_path = os.path.join(config.base_path, 'main.py')
    shutil.copyfile('main.py', d_path)
    d_path = os.path.join(config.base_path, 'solver.py')
    shutil.copyfile('solver.py', d_path)
    d_path = os.path.join(config.base_path, 'model.py')
    shutil.copyfile('model.py', d_path)

    # save configurations
    print(config, file=open(os.path.join(config.base_path, 'config.txt'), 'w'))
    cudnn.benchmark = True

    # print("Random Seed: 1")
    # np.random.seed(1)
    # torch.manual_seed(1)
    # torch.cuda.manual_seed(1)
    # torch.cuda.manual_seed_all(1)
    # random.seed(1)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = True

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    # data_loader
    if config.data_aug:
        facescape = dataset_facescape(config.data_path)
    else:
        facescape = dataset_facescape(config.data_path, random_scale=False, random_rot=False)

    facescape_loader = data.DataLoader(dataset=facescape, batch_size=config.batch_size,
                                       shuffle=(config.mode == 'train'), num_workers=config.num_workers)
    # Training
    solver = Solver(facescape_loader, config, facescape.all_grid_item())
    solver.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Network architecture and hyper-parameters.
    parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
    parser.add_argument('--data_path', type=str, default='training_data.hdf5')
    parser.add_argument('--base_path', type=str, default='./training_results/')
    parser.add_argument('--c_dim', type=int, default=25, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=0.02, help='weight for domain classification loss')
    parser.add_argument('--lambda_reg', type=float, default=0.05, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=2, help='weight for reconstruction loss')
    parser.add_argument('--lambda_idn', type=float, default=0.1, help='weight for identity loss')
    parser.add_argument('--lambda_gp', type=float, default=0.2, help='weight for gradient penalty')
    parser.add_argument('--lambda_para', type=float, default=0.01, help='parameter for gradient penalty')
    parser.add_argument('--lambda_sym', type=float, default=0.5, help='weight for symmetry penalty')

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=400000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=200000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=1, help='number of D updates per each G update')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    parser.add_argument('--data_aug', type=str2bool, default=True)

    # Directories.
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--model_save_dir', type=str, default='models')
    parser.add_argument('--sample_dir', type=str, default='samples')
    parser.add_argument('--result_dir', type=str, default='results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--sample_step', type=int, default=10000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    main(config)
