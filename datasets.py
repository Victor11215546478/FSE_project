"""
Dataset preparation for train and test
"""

import os
import numpy as np

import torch
from torch.utils import data


class PartDataset(data.Dataset):
    """
    Implementation of dataset preparation
    """

    def __init__(
        self, root, npoints=2048, classification=False, class_choice=None, train=True
    ):
        """
        Initialization
        """
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, "synsetoffset2category.txt")
        self.cat = {}

        self.classification = classification

        with open(self.catfile, "r", encoding="UTF-8") as file:
            for line in file:
                ls_ = line.strip().split()
                self.cat[ls_[0]] = ls_[1]
        # print(self.cat)
        if class_choice is not None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.meta = {}
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], "points")
            dir_seg = os.path.join(self.root, self.cat[item], "points_label")
            # print(dir_point, dir_seg)
            fns = sorted(os.listdir(dir_point))
            if train:
                fns = fns[: int(len(fns) * 0.9)]
            else:
                fns = fns[int(len(fns) * 0.9) :]

            # print(os.path.basename(fns))
            for fn_ in fns:
                token = os.path.splitext(os.path.basename(fn_))[0]
                self.meta[item].append(
                    (
                        os.path.join(dir_point, token + ".pts"),
                        os.path.join(dir_seg, token + ".seg"),
                    )
                )

        self.datapath = []
        for item in self.cat:
            for fn_ in self.meta[item]:
                self.datapath.append((item, fn_[0], fn_[1]))

        self.classes = dict(zip(self.cat, range(len(self.cat))))
        print(self.classes)
        self.num_seg_classes = 0
        if not self.classification:
            for i in range(len(self.datapath) // 50):
                num_cl = len(np.unique(np.loadtxt(self.datapath[i][-1]).astype(np.uint8)))
                if num_cl > self.num_seg_classes:
                    self.num_seg_classes = num_cl
        # print(self.num_seg_classes)

    def __getitem__(self, index):
        fn_ = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn_[1]).astype(np.float32)
        seg = np.loadtxt(fn_[2]).astype(np.int64)
        # print(point_set.shape, seg.shape)

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        dist = np.expand_dims(np.expand_dims(dist, 0), 1)
        point_set = point_set / dist

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        point_set = point_set + 1e-5 * np.random.rand(*point_set.shape)

        seg = seg[choice]
        point_set = torch.from_numpy(point_set.astype(np.float32))
        seg = torch.from_numpy(seg.astype(np.int64))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        return (point_set, cls) if self.classification else (point_set, seg)

    def __len__(self):
        return len(self.datapath)


if __name__ == "__main__":
    print("test")
    d = PartDataset(
        root="shapenetcore_partanno_segmentation_benchmark_v0", class_choice=["Chair"]
    )
    print(len(d))
    ps, seg_ = d[0]
    print(ps.size(), ps.type(), seg_.size(), seg_.type())

    d = PartDataset(
        root="shapenetcore_partanno_segmentation_benchmark_v0", classification=True
    )
    print(len(d))
    ps, cls_ = d[0]
    print(ps.size(), ps.type(), cls_.size(), cls_.type())
