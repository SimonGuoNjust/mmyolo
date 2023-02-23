# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import unittest
import cv2

import numpy as np
import torch
from mmdet.structures.bbox import HorizontalBoxes
from mmdet.structures.mask import BitmapMasks,PolygonMasks

from mmyolo.datasets import YOLOv5CocoDataset
from mmyolo.datasets.transforms import Mosaic, Mosaic9, YOLOv5MixUp, YOLOXMixUp
from mmyolo.utils import register_all_modules
from mmyolo.mmyolo_custom.transforms.transforms import MosaicMask
from mmyolo.mmyolo_custom.structrues.mask_transform import TransPolygonMasks

register_all_modules()


class TestMosaic(unittest.TestCase):


    def setUp(self):
        """Setup the data info which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        rng = np.random.RandomState(0)
        self.pre_transform = [
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
            dict(type='YOLOv5RandomAffineWithMask',
                 max_rotate_degree=0.0,
                 max_shear_degree=0.0,
                 scaling_ratio_range=(0.5, 1.5),
                 # img_scale is (width, height)
                 border=(-360 // 2, -640 // 2),
                 border_val=(114, 114, 114))
        ]

        # self.dataset = YOLOv5CocoDataset(
        #     data_prefix=dict(
        #         img=osp.join(osp.dirname(__file__), '../../data')),
        #     ann_file=osp.join(
        #         osp.dirname(__file__), '../../data/coco_sample_color.json'),
        #     filter_cfg=dict(filter_empty_gt=False, min_size=32),
        #     pipeline=[])

        self.dataset = YOLOv5CocoDataset(
            data_prefix=dict(
                img=osp.join('D:\Files\dev\datasets\MinneApple_test','images')),
            ann_file="D:\Files\dev\datasets\Minne_apple_0.json",
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=[])

        self.results = {
            # 'img':
            #     (np.random.random((1280, 720, 3))*255).astype('uint8'),
            'img':
                cv2.imread(r"D:\Files\dev\datasets\MinneApple_test\images\20150919_174151_image1.jpg"),
            'img_shape': (1280, 720),
            'gt_bboxes_labels':
            np.array([1, 2, 3], dtype=np.int64),
            'gt_bboxes':
            np.array([[10, 10, 20, 20], [20, 20, 40, 40], [40, 40, 80, 80]],
                     dtype=np.float32),
            'gt_ignore_flags':
            np.array([0, 0, 1], dtype=bool),
            'gt_masks':
            TransPolygonMasks.random(3,height=1280, width=720),
            'dataset':
            self.dataset
        }

    def test_transform(self):
        # # test assertion for invalid img_scale
        # with self.assertRaises(AssertionError):
        #     transform = Mosaic(img_scale=640)
        #
        # # test assertion for invalid probability
        # with self.assertRaises(AssertionError):
        #     transform = Mosaic(prob=1.5)
        #
        # # test assertion for invalid max_cached_images
        # with self.assertRaises(AssertionError):
        #     transform = Mosaic(use_cached=True, max_cached_images=1)

        transform = MosaicMask(
            img_scale=(640, 640), pre_transform=self.pre_transform)
        results = transform(copy.deepcopy(self.results))
        # self.assertTrue(results['img'].shape[:2] == (20, 24))
        # self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
        #                 results['gt_bboxes'].shape[0])
        # self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        # self.assertTrue(results['gt_bboxes'].dtype == np.float32)
        # self.assertTrue(results['gt_ignore_flags'].dtype == bool)
        img = copy.deepcopy(results['img'])
        print(len(results['gt_bboxes']))
        print(results['gt_masks'].to_bitmap().masks.shape)
        wholemask = torch.sum(results['gt_masks'].to_tensor(torch.uint8,'cpu'), dim=0).type(torch.uint8)
        wholemask = wholemask.numpy().astype('uint8')*255
        print(wholemask.shape)
        # cv2.rectangle(img,results['gt_bboxes'][25].astype('int')[:2],results['gt_bboxes'][25].astype('int')[:2],(255,0,0))
        cv2.imshow('3',img)
        cv2.imshow('0',cv2.cvtColor(wholemask,cv2.COLOR_GRAY2RGB))
        cv2.imshow('1',cv2.bitwise_and(results['img'],results['img'],mask=wholemask))
        cv2.waitKey()


    def test_transform_with_no_gt(self):
        self.results['gt_bboxes'] = np.empty((0, 4), dtype=np.float32)
        self.results['gt_bboxes_labels'] = np.empty((0, ), dtype=np.int64)
        self.results['gt_ignore_flags'] = np.empty((0, ), dtype=bool)
        transform = Mosaic(
            img_scale=(12, 10), pre_transform=self.pre_transform)
        results = transform(copy.deepcopy(self.results))
        self.assertIsInstance(results, dict)
        self.assertTrue(results['img'].shape[:2] == (20, 24))
        self.assertTrue(
            results['gt_bboxes_labels'].shape[0] == results['gt_bboxes'].
            shape[0] == results['gt_ignore_flags'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == np.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == bool)

    def test_transform_with_box_list(self):
        transform = Mosaic(
            img_scale=(12, 10), pre_transform=self.pre_transform)
        results = copy.deepcopy(self.results)
        results['gt_bboxes'] = HorizontalBoxes(results['gt_bboxes'])
        results = transform(results)
        self.assertTrue(results['img'].shape[:2] == (20, 24))
        self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
                        results['gt_bboxes'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == torch.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == bool)


class TestMosaic9(unittest.TestCase):

    def setUp(self):
        """Setup the data info which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        rng = np.random.RandomState(0)
        self.pre_transform = [
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True)
        ]

        self.dataset = YOLOv5CocoDataset(
            data_prefix=dict(
                img=osp.join(osp.dirname(__file__), '../../data')),
            ann_file=osp.join(
                osp.dirname(__file__), '../../data/coco_sample_color.json'),
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=[])
        self.results = {
            'img':
            np.random.random((224, 224, 3)),
            'img_shape': (224, 224),
            'gt_bboxes_labels':
            np.array([1, 2, 3], dtype=np.int64),
            'gt_bboxes':
            np.array([[10, 10, 20, 20], [20, 20, 40, 40], [40, 40, 80, 80]],
                     dtype=np.float32),
            'gt_ignore_flags':
            np.array([0, 0, 1], dtype=bool),
            'gt_masks':
            BitmapMasks(rng.rand(3, 224, 224), height=224, width=224),
            'dataset':
            self.dataset
        }

    def test_transform(self):
        # test assertion for invalid img_scale
        with self.assertRaises(AssertionError):
            transform = Mosaic9(img_scale=640)

        # test assertion for invalid probability
        with self.assertRaises(AssertionError):
            transform = Mosaic9(prob=1.5)

        # test assertion for invalid max_cached_images
        with self.assertRaises(AssertionError):
            transform = Mosaic9(use_cached=True, max_cached_images=1)

        transform = Mosaic9(
            img_scale=(12, 10), pre_transform=self.pre_transform)
        results = transform(copy.deepcopy(self.results))
        self.assertTrue(results['img'].shape[:2] == (20, 24))
        self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
                        results['gt_bboxes'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == np.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == bool)

    def test_transform_with_no_gt(self):
        self.results['gt_bboxes'] = np.empty((0, 4), dtype=np.float32)
        self.results['gt_bboxes_labels'] = np.empty((0, ), dtype=np.int64)
        self.results['gt_ignore_flags'] = np.empty((0, ), dtype=bool)
        transform = Mosaic9(
            img_scale=(12, 10), pre_transform=self.pre_transform)
        results = transform(copy.deepcopy(self.results))
        self.assertIsInstance(results, dict)
        self.assertTrue(results['img'].shape[:2] == (20, 24))
        self.assertTrue(
            results['gt_bboxes_labels'].shape[0] == results['gt_bboxes'].
            shape[0] == results['gt_ignore_flags'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == np.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == bool)

    def test_transform_with_box_list(self):
        transform = Mosaic9(
            img_scale=(12, 10), pre_transform=self.pre_transform)
        results = copy.deepcopy(self.results)
        results['gt_bboxes'] = HorizontalBoxes(results['gt_bboxes'])
        results = transform(results)
        self.assertTrue(results['img'].shape[:2] == (20, 24))
        self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
                        results['gt_bboxes'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == torch.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == bool)


class TestYOLOv5MixUp(unittest.TestCase):

    def setUp(self):
        """Setup the data info which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        rng = np.random.RandomState(0)
        self.pre_transform = [
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True)
        ]
        self.dataset = YOLOv5CocoDataset(
            data_prefix=dict(
                img=osp.join(osp.dirname(__file__), '../../data')),
            ann_file=osp.join(
                osp.dirname(__file__), '../../data/coco_sample_color.json'),
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=[])

        self.results = {
            'img':
            np.random.random((288, 512, 3)),
            'img_shape': (288, 512),
            'gt_bboxes_labels':
            np.array([1, 2, 3], dtype=np.int64),
            'gt_bboxes':
            np.array([[10, 10, 20, 20], [20, 20, 40, 40], [40, 40, 80, 80]],
                     dtype=np.float32),
            'gt_ignore_flags':
            np.array([0, 0, 1], dtype=bool),
            'gt_masks':
            BitmapMasks(rng.rand(3, 288, 512), height=288, width=512),
            'dataset':
            self.dataset
        }

    def test_transform(self):
        transform = YOLOv5MixUp(pre_transform=self.pre_transform)
        results = transform(copy.deepcopy(self.results))
        self.assertTrue(results['img'].shape[:2] == (288, 512))
        self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
                        results['gt_bboxes'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == np.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == bool)

        # test assertion for invalid max_cached_images
        with self.assertRaises(AssertionError):
            transform = YOLOv5MixUp(use_cached=True, max_cached_images=1)

    def test_transform_with_box_list(self):
        results = copy.deepcopy(self.results)
        results['gt_bboxes'] = HorizontalBoxes(results['gt_bboxes'])

        transform = YOLOv5MixUp(pre_transform=self.pre_transform)
        results = transform(results)
        self.assertTrue(results['img'].shape[:2] == (288, 512))
        self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
                        results['gt_bboxes'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == torch.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == bool)


class TestYOLOXMixUp(unittest.TestCase):

    def setUp(self):
        """Setup the data info which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        rng = np.random.RandomState(0)
        self.pre_transform = [
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True)
        ]
        self.dataset = YOLOv5CocoDataset(
            data_prefix=dict(
                img=osp.join(osp.dirname(__file__), '../../data')),
            ann_file=osp.join(
                osp.dirname(__file__), '../../data/coco_sample_color.json'),
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=[])
        self.results = {
            'img':
            np.random.random((224, 224, 3)),
            'img_shape': (224, 224),
            'gt_bboxes_labels':
            np.array([1, 2, 3], dtype=np.int64),
            'gt_bboxes':
            np.array([[10, 10, 20, 20], [20, 20, 40, 40], [40, 40, 80, 80]],
                     dtype=np.float32),
            'gt_ignore_flags':
            np.array([0, 0, 1], dtype=bool),
            'gt_masks':
            BitmapMasks(rng.rand(3, 224, 224), height=224, width=224),
            'dataset':
            self.dataset
        }

    def test_transform(self):
        # test assertion for invalid img_scale
        with self.assertRaises(AssertionError):
            transform = YOLOXMixUp(img_scale=640)

        # test assertion for invalid max_cached_images
        with self.assertRaises(AssertionError):
            transform = YOLOXMixUp(use_cached=True, max_cached_images=1)

        transform = YOLOXMixUp(
            img_scale=(10, 12),
            ratio_range=(0.8, 1.6),
            pad_val=114.0,
            pre_transform=self.pre_transform)

        # self.results['mix_results'] = [copy.deepcopy(self.results)]
        results = transform(copy.deepcopy(self.results))
        self.assertTrue(results['img'].shape[:2] == (224, 224))
        self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
                        results['gt_bboxes'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == np.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == bool)

    def test_transform_with_boxlist(self):
        results = copy.deepcopy(self.results)
        results['gt_bboxes'] = HorizontalBoxes(results['gt_bboxes'])

        transform = YOLOXMixUp(
            img_scale=(10, 12),
            ratio_range=(0.8, 1.6),
            pad_val=114.0,
            pre_transform=self.pre_transform)
        results = transform(results)
        self.assertTrue(results['img'].shape[:2] == (224, 224))
        self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
                        results['gt_bboxes'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == torch.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == bool)
