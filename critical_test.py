import numpy as np

import datasets
import unittest

from datasets import PartDataset
from kdtree import make_cKDTree




class DatasetPrep(unittest.TestCase):


    def test_check_len_of_class(self):
        d = PartDataset(root="shapenetcore_partanno_segmentation_benchmark_v0", class_choice=["Earphone"])
        self.assertEqual(len(d), 62)


    def test_check_ps_type(self):
        d = PartDataset(root="shapenetcore_partanno_segmentation_benchmark_v0", class_choice=["Earphone"])
        ps, cls = d[0]
        self.assertEqual(ps.type(), 'torch.FloatTensor')

    def test_check_cls_type(self):
        d = PartDataset(root="shapenetcore_partanno_segmentation_benchmark_v0", class_choice=["Earphone"])
        ps, cls = d[0]
        self.assertEqual(cls.type(), 'torch.LongTensor')

    def test_check_total_len(self):
        d = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = True)
        self.assertEqual(len(d), 15990)



class Check_make_cKDTree(unittest.TestCase):


    def test_check_len(self):
        self.assertEqual(len(make_cKDTree(np.random.rand(100, 300), depth=5)[0]), 5)


if __name__ == '__main__':
    unittest.main()