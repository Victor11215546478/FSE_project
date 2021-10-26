"""
Code for testing Deep Kd-Networks
"""

import numpy as np

import torch
import torch.nn.functional as F

from torch import nn
from torch.autograd import Variable

from kdtree import make_cKDTree
from datasets import PartDataset

NUM_POINTS = 2048

# automatic choice of device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# make it reproducible
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True


class KDNet(nn.Module):
    """
    Implementation of Kd-network
    """
    def __init__(self, k=16):
        """
        Initialization of convolution layers
        """
        super().__init__()
        self.conv1 = nn.Conv1d(3, 8 * 3, 1, 1)
        self.conv2 = nn.Conv1d(8, 32 * 3, 1, 1)
        self.conv3 = nn.Conv1d(32, 64 * 3, 1, 1)
        self.conv4 = nn.Conv1d(64, 64 * 3, 1, 1)
        self.conv5 = nn.Conv1d(64, 64 * 3, 1, 1)
        self.conv6 = nn.Conv1d(64, 128 * 3, 1, 1)
        self.conv7 = nn.Conv1d(128, 256 * 3, 1, 1)
        self.conv8 = nn.Conv1d(256, 512 * 3, 1, 1)
        self.conv9 = nn.Conv1d(512, 512 * 3, 1, 1)
        self.conv10 = nn.Conv1d(512, 512 * 3, 1, 1)
        self.conv11 = nn.Conv1d(512, 1024 * 3, 1, 1)
        self.f_c = nn.Linear(1024, k)

    def forward(self, in_x, c_dict):
        """
        Forward pass
        """

        def kdconv(x_in, dim, featdim, sel, conv):
            """
            Definition of special kdtree convolution layers
            """
            # batchsize = x.size(0)
            # print(batchsize)
            x_in = F.relu(conv(x_in))
            x_in = x_in.view(-1, featdim, 3, dim)
            x_in = x_in.view(-1, featdim, 3 * dim)

            sel = Variable(sel + (torch.arange(0, dim) * 3).long())
            if x_in.is_cuda:
                sel = sel.to(device)
            x_in = torch.index_select(x_in, dim=2, index=sel)
            x_in = x_in.view(-1, featdim, dim // 2, 2)
            x_in = torch.squeeze(torch.max(x_in, dim=-1, keepdim=True)[0], 3)
            return x_in


        in_x = kdconv(in_x, 2048, 8, c_dict[0], self.conv1)
        in_x = kdconv(in_x, 1024, 32, c_dict[1], self.conv2)
        in_x = kdconv(in_x, 512, 64, c_dict[2], self.conv3)
        in_x = kdconv(in_x, 256, 64, c_dict[3], self.conv4)
        in_x = kdconv(in_x, 128, 64, c_dict[4], self.conv5)
        in_x = kdconv(in_x, 64, 128, c_dict[5], self.conv6)
        in_x = kdconv(in_x, 32, 256, c_dict[6], self.conv7)
        in_x = kdconv(in_x, 16, 512, c_dict[7], self.conv8)
        in_x = kdconv(in_x, 8, 512, c_dict[8], self.conv9)
        in_x = kdconv(in_x, 4, 512, c_dict[9], self.conv10)
        in_x = kdconv(in_x, 2, 1024, c_dict[10], self.conv11)

        out = F.log_softmax(self.f_c(in_x.view(-1, 1024)))

        return out


d = PartDataset(
    root="shapenetcore_partanno_segmentation_benchmark_v0",
    classification=True,
    train=False,
)
print(len(d.classes), len(d))

levels = (np.log(NUM_POINTS) / np.log(2)).astype(int)

NET = KDNet().to(device)

# model_name = sys.argv[1]
NET.load_state_dict(torch.load("save_model_100.pth"))
NET.eval()


corrects = []
for j in range(len(d)):

    point_set, class_label = d[j]

    target = Variable(class_label).to(device)
    if target != 0:
        pass

    point_set = point_set[:NUM_POINTS]
    if point_set.size(0) < NUM_POINTS:
        point_set = torch.cat(
            [point_set, point_set[0 : NUM_POINTS - point_set.size(0)]], 0
        )

    cutdim, tree = make_cKDTree(point_set.numpy(), depth=levels)

    cutdim_v = [(torch.from_numpy(np.array(item).astype(np.int64))) for item in cutdim]

    points = torch.FloatTensor(tree[-1])
    points_v = (
        Variable(torch.unsqueeze(torch.squeeze(points), 0)).transpose(2, 1).to(device)
    )

    pred = NET.forward(points_v, cutdim_v)

    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    corrects.append(correct)
    print(f"{j}/{len(d)} , {float(sum(corrects)) / float(len(corrects))}")


print(float(sum(corrects)) / float(len(d)))
