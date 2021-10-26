#!/usr/bin/env python3

"""
Code for critical testing
"""

import unittest

import numpy as np

import datasets
from datasets import PartDataset

from kdtree import make_cKDTree


class CheckTree(unittest.TestCase):
    """
    Check tree building
    """

    def test_check_len(self):
        """
        Check length
        """
        self.assertEqual(len(make_cKDTree(np.random.rand(100, 200), depth=4)[0]), 4)


class DatasetPrep(unittest.TestCase):
    """
    Check dataset
    """

    def test_check_len_of_chair_class(self):
        """
        Check length: Chair class
        """
        data = PartDataset(
            root="shapenetcore_partanno_segmentation_benchmark_v0", class_choice=["Chair"]
        )
        self.assertEqual(len(data), 3371)

    def test_check_len_of_all_class(self):
        """
        Check length
        """
        data = PartDataset(
            root="shapenetcore_partanno_segmentation_benchmark_v0", class_choice=None
        )
        self.assertGreater(len(data), 15000)
        self.assertEqual(len(data), 15990)

    def test_check_class_num_map(self):
        """
        Check map between classes and map
        """
        data = PartDataset(
            root="shapenetcore_partanno_segmentation_benchmark_v0", class_choice=None
        )
        self.assertEqual(data.classes['Airplane'], 0)

    def test_check_2multiplicity(self):
        """
        Check if size divided by 2 without residue
        """
        data = PartDataset(
            root="shapenetcore_partanno_segmentation_benchmark_v0", class_choice=None
        )
        self.assertEqual(data[0][1].size()[0] % 2, 0)


if __name__ == '__main__':
    unittest.main()
