from test_datasets.test_transforms.test_transforms import TestYOLOv5RandomAffineMask
from test_datasets.test_transforms.test_mix_img_transforms import TestMosaic
from mmdet.structures.mask.structures import BitmapMasks
from pycocotools import mask as maskUtils

from mmyolo.mmyolo_custom.structrues.mask_transform import TransPolygonMasks
import numpy as np
import cv2

b = TestMosaic()
b.setUp()
b.test_transform()