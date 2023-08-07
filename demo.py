from model import Generator
from torch.backends import cudnn
from torch.autograd import Variable
import torch
import numpy as np
import os
import scipy.io as scio
import h5py
import hdf5storage


# create target labels
def create_labels(c_org):
    c_trg_list = []
    for i in range(30):
        c_trg = c_org.copy()
        if i < 20: # Set expression attribute to 1 and the rest to 0.
            c_trg[:, i] = 1
            for j in range(20):
                if j != i:
                    c_trg[:, j] = 0
            if i in [4, 5, 7, 8]:
                c_trg[:, 22] = 0
                c_trg[:, 23] = 1
            else:
                c_trg[:, 22] = 1
                c_trg[:, 23] = 0

        elif i == 20:
            c_trg[:, 20] = 1  # Reverse gender attribute.
            c_trg[:, 21] = 0  # Reverse gender attribute.
        elif i == 21:
            c_trg[:, 20] = 0  # Reverse gender attribute.
            c_trg[:, 21] = 1  # Reverse gender attribute.
        else:
            c_trg[:, 24]= 2*(i-22)/7.-1  # set age ranging linearly from 17 to 70
        c_trg_list.append(c_trg)
    return c_trg_list

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
if not torch.cuda.is_available():
    raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
cudnn.benchmark = True

# setting path for input, output, and model
input_data = './test_data/input.mat'
input_label='./test_data/input_label.mat'
output_data = './test_data/output.mat'
model_path = './saved_model/g_model.pth'

# load pre-defined grid
with h5py.File('grid.hdf5', 'r') as file:
    ix = torch.LongTensor(np.round(file['IX'])).squeeze(1)
    iy = torch.LongTensor(np.round(file['IY'])).squeeze(1)
    iz = torch.LongTensor(np.round(file['IZ'])).squeeze(1)
    wx = torch.FloatTensor(file['WX'])
    wy = torch.FloatTensor(file['WY'])
    wz = torch.FloatTensor(file['WZ'])
    coordxy = 2 * torch.FloatTensor(file['gridxy']) / 127. - 1
all_grid_item_T = ix, iy, iz, wx, wy, wz, coordxy

# load trained generator
model = Generator(conv_dim=64, c_dim=25, repeat_num=6, all_grid_item=all_grid_item_T)
model.load_state_dict(torch.load(model_path), strict=True)
model = model.cuda()

# load input data
label_in = hdf5storage.loadmat(input_label)
input_label = label_in['input_label']
c_trg_list = create_labels(input_label)

with torch.no_grad():
    input = hdf5storage.loadmat(input_data)
    input = input['input']
    # formalize input
    input = Variable(torch.from_numpy(input).float()).cuda()
    rot_angle_fixed = torch.eye(3).unsqueeze(0).repeat(input.size(0), 1, 1).cuda()
    x_fake_list = input.clone().cpu().unsqueeze(0) #cat input to output
    for c_fixed in c_trg_list:
        c_fixed=Variable(torch.from_numpy(c_fixed).float()).cuda()
        x_output_fake, _ = model(input, c_fixed, rot_angle_fixed)
        x_fake_list = torch.cat((x_fake_list, x_output_fake.cpu().unsqueeze(0)), dim=0)
    scio.savemat(output_data, {'shape': x_fake_list.numpy()})
    print('Saved generated data into {}...'.format(output_data))










