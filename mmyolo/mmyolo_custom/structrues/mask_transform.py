import torch

from mmdet.structures.mask import PolygonMasks


class TransPolygonMasks(PolygonMasks):

    def project_(self, homography_matrix):
        polys = torch.Tensor(self.masks[0][0]).reshape(-1, 2)
        polys = torch.cat([polys, polys.new_ones(*polys.shape[:-1], 1)], dim=-1)
        polys_T = torch.transpose(polys, -1, -2)
        polys_T = torch.matmul(torch.Tensor(homography_matrix), polys_T)
        polys = torch.transpose(polys_T, -1, -2)
        polys = polys[:, :2]
        self.mask = [[polys.numpy()]]