import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# from medpy.io import load, save
# import cv2
# from skimage.measure import label as la
import SimpleITK as sitk
# from scipy.ndimage import zoom
import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from coord import make_coord
def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img)))
def resize_image(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    # print('--resize ing--')
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 原来的体素块尺寸
    originSpacing = itkimage.GetSpacing()

    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int)    # spacing肯定不能是整数

    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itk_img_res = resampler.Execute(itkimage)  # 得到重新采样后的图像
    # print('--resize finish--')
    return itk_img_res
def resize_image_itk(ori_img, target_img, resamplemethod=sitk.sitkLinear):

    target_Size = target_img.GetSize()  # 目标图像大小  [x,y,z]
    target_Spacing = target_img.GetSpacing()  # 目标的体素块尺寸    [x,y,z]
    target_origin = target_img.GetOrigin()  # 目标的起点 [x,y,z]
    target_direction = target_img.GetDirection()  # 目标的方向 [冠,矢,横]=[z,y,x]

    # itk的方法进行resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ori_img)  # 需要重新采样的目标图像
    # 设置目标图像的信息
    resampler.SetSize(target_Size)  # 目标图像大小
    resampler.SetOutputOrigin(target_origin)
    resampler.SetOutputDirection(target_direction)
    resampler.SetOutputSpacing(target_Spacing)
    # 根据需要重采样图像的情况设置不同的dype
    if resamplemethod == sitk.sitkNearestNeighbor:
        resampler.SetOutputPixelType(sitk.sitkUInt8)  # 近邻插值用于mask的，保存uint8
    else:
        resampler.SetOutputPixelType(sitk.sitkFloat32)  # 线性插值用于PET/CT/MRI之类的，保存float32
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itk_img_resampled = resampler.Execute(ori_img)  # 得到重新采样后的图像
    return itk_img_resampled
from skimage.measure import label as la
def crop_func(img):
    np_image=sitk.GetArrayFromImage(img)
    np_image[np_image!=0]=1
    loc_img, num = la(np_image, background=0, return_num=True)
    loc_img[loc_img!=0]=1
    max_label = 0
    max_num = 0
    for i in range(1, num + 1):
        if np.sum(loc_img == i) > max_num:
            max_num = np.sum(loc_img == i)
            max_label = i
    mcr = (loc_img == max_label)
    mcr = mcr + 0
    y_true, x_true, z_true = np.where(mcr)
    box = np.array([[np.min(y_true), np.max(y_true)],
                    [np.min(x_true), np.max(x_true)],
                    [np.min(z_true), np.max(z_true)]])
    y_min, y_max = box[0]
    x_min, x_max = box[1]
    z_min, z_max = box[2]
    res=img[z_min:z_max,x_min:x_max,y_min:y_max]
    return res


def crop_image(HR,sda, patch_size, stride):
    HR=resize_image_itk(HR,sda)
    size=HR.GetSize()
    image_shape = np.array(size)
    patch_shape = np.array((patch_size,patch_size,patch_size))

    HR_patches = []
    sda_patches = []
    for i in range(0, image_shape[0], stride):
        for j in range(0, image_shape[1], stride):
            for k in range(0, image_shape[2], stride):
                # 裁剪的起始和结束位置
                start_i = i
                end_i = i + patch_shape[0]
                start_j = j
                end_j = j + patch_shape[1]
                start_k = k
                end_k = k + patch_shape[2]

                # 处理最后一部分无法完全适应补丁尺寸的情况
                if end_i > image_shape[0]:
                    start_i = image_shape[0] - patch_shape[0]
                    end_i = image_shape[0]
                if end_j > image_shape[1]:
                    start_j = image_shape[1] - patch_shape[1]
                    end_j = image_shape[1]
                if end_k > image_shape[2]:
                    start_k = image_shape[2] - patch_shape[2]
                    end_k = image_shape[2]

                HR_patch = HR[start_i:end_i, start_j:end_j, start_k:end_k]
                sda_patch = sda[start_i:end_i, start_j:end_j, start_k:end_k]
                HR_patches.append(HR_patch)
                sda_patches.append(sda_patch)
    return HR_patches,sda_patches

class custom_dataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform  # 传入数据预处理
        self.B_data = []
        self.A_data = []
        self.C_data = []
        self.A_name = []
        self.A_coord = []
        self.B_coord = []
        self.C_coord = []
        self.size = []

        path="/public/bme/home/huangshj/code/MICCAI2023/dataset/atlas"
        path_LR="/public/bme/home/huangshj/code/AAAI_3/datasets/sda_5"
        # path_list=["/media/shijie/HDD/Data/good/ZJ/sda","/media/shijie/HDD/Data/good/FD/sda"]
        data_path_list_ = os.listdir(path)
        for data_path_ in data_path_list_:
            term=data_path_
            t1_name = os.path.join(path, data_path_)
            gw=data_path_.split("_")[0]
            print("the label name is:", t1_name)
            for i in range(5):

                HR_sitk = sitk.ReadImage(t1_name)
                sda_sitk = sitk.ReadImage(os.path.join(path_LR, data_path_.replace(".nii","_{}.nii".format(i))))
                sda_sitk=resize_image_itk(sda_sitk,HR_sitk)
                atlas_sitk = sitk.ReadImage(t1_name)

                HR=HR_sitk
                sda=sda_sitk
                atlas=atlas_sitk
                HR=resize_image(HR,(HR.GetSize()[0]//2,HR.GetSize()[1]//2,HR.GetSize()[2]//2))
                HR_img=sitk.GetArrayFromImage(HR).transpose(2, 1, 0)
                HR_img=(HR_img-HR_img.min())/(HR_img.max()-HR_img.min()+1e-10)


                sda = resize_image(sda, (sda.GetSize()[0] //2, sda.GetSize()[1] //2, sda.GetSize()[2] //2))
                sda_img = sitk.GetArrayFromImage(sda).transpose(2, 1, 0)
                sda_img = (sda_img - sda_img.min()) / (sda_img.max() - sda_img.min()+1e-10)


                atlas = resize_image(atlas, (atlas.GetSize()[0] //2, atlas.GetSize()[1] //2, atlas.GetSize()[2] //2))
                atlas_img = sitk.GetArrayFromImage(atlas).transpose(2, 1, 0)
                atlas_img = (atlas_img - atlas_img.min()) / (atlas_img.max() - atlas_img.min()+1e-10)

                C_hr_sda=make_coord(HR, sda)
                C_hr=make_coord(HR, HR)

                self.A_data.append(torch.tensor(HR_img).unsqueeze(0))
                self.A_coord.append(C_hr)
                self.B_data.append(torch.tensor(sda_img).unsqueeze(0))
                self.B_coord.append(C_hr_sda)
                self.A_name.append(t1_name)
                self.C_data.append(torch.tensor(atlas_img).unsqueeze(0))
                # if not os.path.exists(data_path_.replace("cropped_image","feature")):
                #     os.makedirs(data_path_.replace("cropped_image","feature"))




    def __getitem__(self, idx):  # 根据 idx 取出其中一个name

        hr = self.A_data[idx % len(self.A_data)]
        coord_gt = self.A_coord[idx % len(self.A_coord)]
        sda = self.B_data[idx % len(self.B_data)]
        atlas = self.C_data[idx % len(self.C_data)]
        coord_sda = self.B_coord[idx % len(self.B_coord)]
        name = self.A_name[idx % len(self.A_name)]

        return {'hr': hr, 'coord_gt': coord_gt,'sda': sda,"atlas": atlas, 'coord_sda': coord_sda, "name": name}
    def __len__(self):  # 总数据的多少
        return len(self.A_data)

from torch.utils.data import DataLoader


def load_data_train():
    train_dataset = custom_dataset() # 读入 .pkl 文件
    loader = DataLoader(train_dataset, batch_size=1,shuffle=False, num_workers=0, pin_memory=True)
    return loader

