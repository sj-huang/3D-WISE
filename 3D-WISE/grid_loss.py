from torch import nn
import torch
from torch.autograd import Variable
import SimpleITK as sitk
class Grad(nn.Module):
    """
    Bending Penalty of the spatial transformation (2D)
    """

    def __init__(self):
        super(Grad, self).__init__()

    def _diffs(self, y, dim):  # y shape(bs, nfeat, vol_shape)
        ndims = y.ndimension() - 2
        d = dim + 2
        # permute dimensions to put the ith dimension first
        #       r = [d, *range(d), *range(d + 1, ndims + 2)]
        y = y.permute(d, *range(d), *range(d + 1, ndims + 2))
        dfi=torch.zeros_like(y)
        dfi_ = y[1:, ...] - y[:-1, ...]
        dfi[:-1]=dfi_
        # permute back
        # note: this might not be necessary for this loss specifically,
        # since the results are just summed over anyway.
        #       r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
        df = dfi.permute(*range(1, d + 1), 0, *range(d + 1, ndims + 2))

        return df

    def forward(self, pred,atlas_prior):  # shape(B,C,H,W)
        Ty = self._diffs(pred, dim=0)
        Tx = self._diffs(pred, dim=1)
        Tz = self._diffs(pred, dim=2)
        Tyy = self._diffs(Ty, dim=0)
        Txx = self._diffs(Tx, dim=1)
        Tzz = self._diffs(Tz, dim=2)

        Ty_atlas = self._diffs(atlas_prior, dim=0)
        Tx_atlas = self._diffs(atlas_prior, dim=1)
        Tz_atlas = self._diffs(atlas_prior, dim=2)
        Tyy_atlas = self._diffs(Ty_atlas, dim=0)
        Txx_atlas = self._diffs(Tx_atlas, dim=1)
        Tzz_atlas = self._diffs(Tz_atlas, dim=2)



        Txy = self._diffs(Tx, dim=0)
        Txz = self._diffs(Tx, dim=2)
        Tyz = self._diffs(Ty, dim=2)

        Txy_atlas = self._diffs(Tx_atlas, dim=0)
        Txz_atlas = self._diffs(Tx_atlas, dim=2)
        Tyz_atlas = self._diffs(Ty_atlas, dim=2)

        # p_1 = Tyy.pow(2).mean() + Txx.pow(2).mean() + Tzz.pow(2).mean()+\
        #     2 * Txy.pow(2).mean()+ 2 * Txz.pow(2).mean()+ 2 * Tyz.pow(2).mean()
        p_2 = (Tyy-0.2*Tyy_atlas).pow(2).mean() + (Txx-0.2*Txx_atlas).pow(2).mean() + (Tzz-0.2*Tzz_atlas).pow(2).mean()+\
            2*(Txy-0.2*Txy_atlas).pow(2).mean() + 2*(Txz-0.2*Txz_atlas).pow(2).mean() + 2*(Tyz-0.2*Tyz_atlas).pow(2).mean()
        # p=p_1+p_2
        return p_2
# import numpy as np
# # img_sitk=sitk.ReadImage("/home/shijie/DL_SRR_T2/motion_generator/data_atlas/atlas/Chinese_23.nii.gz")
# # img_sitk=sitk.ReadImage("/home/shijie/Downloads/pred_0.nii.gz")
# img_sitk=sitk.ReadImage("/home/shijie/Downloads/gt_0.nii.gz")
# np_img=sitk.GetArrayFromImage(img_sitk)[np.newaxis,np.newaxis]
# np_img=torch.tensor(np_img)
# df=Grad()(np_img)
# print(df)
import numpy as np
from numpy import *
def get_psnr(img1, img2):
    img1 = sitk.GetArrayFromImage(img1)
    img2 = sitk.GetArrayFromImage(img2)
    img1 = (img1 - img1.min()) / (img1.max() - img1.min())
    img2 = (img2 - img2.min()) / (img2.max() - img2.min())

    mse = np.mean((img1 - img2) ** 2)
    # if mse < 1.0e-10:
    #   return 100
    return 10 * math.log10(1 / mse), mse
class Grad_z(nn.Module):
    """
    Bending Penalty of the spatial transformation (2D)
    """

    def __init__(self):
        super(Grad_z, self).__init__()

    def _diffs(self, y, dim):  # y shape(bs, nfeat, vol_shape)
        ndims = y.ndimension() - 2
        d = dim + 2
        # permute dimensions to put the ith dimension first
        #       r = [d, *range(d), *range(d + 1, ndims + 2)]
        y = y.permute(d, *range(d), *range(d + 1, ndims + 2))
        dfi=torch.zeros_like(y)
        dfi_ = y[1:, ...] - y[:-1, ...]
        dfi[:-1]=dfi_
        # permute back
        # note: this might not be necessary for this loss specifically,
        # since the results are just summed over anyway.
        #       r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
        df = dfi.permute(*range(1, d + 1), 0, *range(d + 1, ndims + 2))

        return df

    def forward(self, pred):  # shape(B,C,H,W)
        Ty = self._diffs(pred, dim=0)
        Tx = self._diffs(pred, dim=1)
        Tz = self._diffs(pred, dim=2)
        Tyy = self._diffs(Ty, dim=0)
        Txx = self._diffs(Tx, dim=1)
        Tzz = self._diffs(Tz, dim=2)

        # Ty_atlas = self._diffs(atlas_prior, dim=0)
        # Tx_atlas = self._diffs(atlas_prior, dim=1)
        # Tz_atlas = self._diffs(atlas_prior, dim=2)
        # Tyy_atlas = self._diffs(Ty_atlas, dim=0)
        # Txx_atlas = self._diffs(Tx_atlas, dim=1)
        # Tzz_atlas = self._diffs(Tz_atlas, dim=2)



        Txy = self._diffs(Tx, dim=0)
        Txz = self._diffs(Tx, dim=2)
        Tyz = self._diffs(Ty, dim=2)

        p = Tyy.pow(2).mean() + Txx.pow(2).mean() + Tzz.pow(2).mean()+\
            2 * Txy.pow(2).mean()+ 2 * Txz.pow(2).mean()+ 2 * Tyz.pow(2).mean()
        # p = (Tyy-Tyy_atlas).pow(2).mean() + (Txx-Txx_atlas).pow(2).mean() + (Tzz-Tzz_atlas).pow(2).mean()

        return p