from typing import List, Tuple, Union

import numpy as np

from mmdet.structures.bbox import (HorizontalBoxes, autocast_box_type,
                                   get_box_type)
from mmyolo.registry import TRANSFORMS
from mmyolo.datasets.transforms import YOLOv5RandomAffine
import cv2

@TRANSFORMS.register_module()
class YOLOv5RandomAffineWithMask(YOLOv5RandomAffine):
    """Random affine transform data augmentation in YOLOv5. It is different
    from the implementation in YOLOX.

    This operation randomly generates affine transform matrix which including
    rotation, translation, shear and scaling transforms.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)

    Args:
        max_rotate_degree (float): Maximum degrees of rotation transform.
            Defaults to 10.
        max_translate_ratio (float): Maximum ratio of translation.
            Defaults to 0.1.
        scaling_ratio_range (tuple[float]): Min and max ratio of
            scaling transform. Defaults to (0.5, 1.5).
        max_shear_degree (float): Maximum degrees of shear
            transform. Defaults to 2.
        border (tuple[int]): Distance from width and height sides of input
            image to adjust output shape. Only used in mosaic dataset.
            Defaults to (0, 0).
        border_val (tuple[int]): Border padding values of 3 channels.
            Defaults to (114, 114, 114).
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
    """


    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """The YOLOv5 random affine transform function.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        img = results['img']
        # self.border is wh format
        height = img.shape[0] + self.border[1] * 2
        width = img.shape[1] + self.border[0] * 2

        # Note: Different from YOLOX
        center_matrix = np.eye(3, dtype=np.float32)
        center_matrix[0, 2] = -img.shape[1] / 2
        center_matrix[1, 2] = -img.shape[0] / 2

        warp_matrix, scaling_ratio = self._get_random_homography_matrix(
            height, width)
        warp_matrix = warp_matrix @ center_matrix

        img = cv2.warpPerspective(
            img,
            warp_matrix,
            dsize=(width, height),
            borderValue=self.border_val)
        results['img'] = img
        results['img_shape'] = img.shape

        bboxes = results['gt_bboxes']
        masks = results['gt_masks']
        num_bboxes = len(bboxes)
        if num_bboxes:

            orig_bboxes = bboxes.clone()

            bboxes.project_(warp_matrix)
            masks.project_(warp_matrix)

            if self.bbox_clip_border:
                bboxes.clip_([height, width])

            # filter bboxes
            orig_bboxes.rescale_([scaling_ratio, scaling_ratio])

            # Be careful: valid_index must convert to numpy,
            # otherwise it will raise out of bounds when len(valid_index)=1
            valid_index = self.filter_gt_bboxes(orig_bboxes, bboxes).numpy()
            results['gt_bboxes'] = bboxes[valid_index]
            results['gt_bboxes_labels'] = results['gt_bboxes_labels'][
                valid_index]
            results['gt_ignore_flags'] = results['gt_ignore_flags'][
                valid_index]
            results['gt_masks'] = masks
            #
            # if 'gt_masks' in results:
            #     raise NotImplementedError('RandomAffine only supports bbox.')
        return results
