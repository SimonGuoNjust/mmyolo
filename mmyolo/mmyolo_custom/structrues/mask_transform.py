import numpy as np
import torch

from mmdet.structures.mask import PolygonMasks, BitmapMasks
import cv2
import numpy
from pycocotools import mask as maskUtils


class TransPolygonMasks(PolygonMasks):

    @staticmethod
    def create_from_parent(parent : PolygonMasks):
        return TransPolygonMasks(parent.masks, parent.height, parent.width)

    def clip_(self, maxw, maxh):
        objects = list()
        for object_mask in self.masks:
            one_object = list()
            for contour in object_mask:
                polys = contour.reshape(-1,2)
                polys[:,0] = np.clip(polys[:,0],a_min=0, a_max=maxw)
                polys[:,1] = np.clip(polys[:,1],a_min=0, a_max=maxh)
                polys = polys.reshape(-1)
                one_object.append(polys)
            objects.append(one_object)
        return TransPolygonMasks(objects, self.height, self.width)

    def biclip_(self, width, height):
        objects = list()
        for object_mask in self.masks:
            one_object = list()
            for contour in object_mask:
                polys = contour.reshape(-1,2)
                polys[:,0] = np.clip(polys[:,0],a_min=width[0], a_max=width[1],)
                polys[:,1] = np.clip(polys[:,1],a_min=height[0], a_max=height[1],)
                polys = polys.reshape(-1)
                one_object.append(polys)
            objects.append(one_object)
        return TransPolygonMasks(objects, self.height, self.width)

    def project_(self, homography_matrix, clip=True):
        objects = list()
        for object_mask in self.masks:
            one_object = list()
            for contour in object_mask:
                polys = torch.Tensor(contour).reshape(-1,2).float()
                polys = torch.cat([polys, polys.new_ones(*polys.shape[:-1], 1)], dim=-1)
                polys_T = torch.transpose(polys, -1, -2)
                polys_T = torch.matmul(torch.Tensor(homography_matrix), polys_T)
                polys = torch.transpose(polys_T, -1, -2)
                polys = polys[:, :2]
                polys = polys.reshape(-1).numpy()
                one_object.append(polys)
            objects.append(one_object)
        if clip:
            self.clip_(self.width, self.height)
        return TransPolygonMasks(objects,self.height,self.width)

    def translate(self,
                  out_shape,
                  offset,
                  direction='horizontal',
                  border_value=None,
                  interpolation=None):
        result = super(TransPolygonMasks, self).translate(out_shape,offset,direction,border_value,interpolation)
        return self.create_from_parent(result)

    def rescale(self, scale, interpolation=None):
        result = super(TransPolygonMasks, self).rescale(scale,interpolation)
        return self.create_from_parent(result)

    def resize(self, out_shape, interpolation=None):
        result = super(TransPolygonMasks, self).resize(out_shape,interpolation)
        return self.create_from_parent(result)

    def crop(self, bbox):
        result = super(TransPolygonMasks, self).crop(bbox)
        return self.create_from_parent(result)
