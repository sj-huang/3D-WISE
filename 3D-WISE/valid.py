
# -*- coding:utf-8 -*-
# -----------------------------------------
#   Filename: train.py
#   Author  : Qing Wu
#   Email   : wuqing@shanghaitech.edu.cn
#   Date    : 2021/9/19
# -----------------------------------------

import torch
import model
import argparse
import os
from data_coord_v import custom_dataset,resize_image
import SimpleITK as sitk
import numpy as np
import torch
import tqdm
from torch.utils.data import SubsetRandomSampler
import SimpleITK as sitk
from numpy import *
import torch
import torch.nn.functional as F
def get_correlation(img1, img2):
    I = sitk.GetArrayFromImage(img1)
    J = sitk.GetArrayFromImage(img2)
    if I.shape != J.shape:
        raise AssertionError("The inputs must be the same size.")
    u = I.reshape((I.shape[0] * I.shape[1] * I.shape[2], 1))
    v = J.reshape((J.shape[0] * J.shape[1] * J.shape[2], 1))
    u = u - u.mean(keepdims=True)
    v = v - v.mean(keepdims=True)
    # CC = (np.transpose(d).dot(u * v)) / (np.sqrt(np.transpose(u).dot(u)).dot(np.sqrt(np.transpose(v).dot(v))))
    NCC = np.mean((np.multiply(u,v)) / (np.std(u)*(np.std(v))))
    # print("CC: ", CC)
    return NCC
def get_ssim(img1, img2):
    img1 = sitk.GetArrayFromImage(img1)
    img2 = sitk.GetArrayFromImage(img2)
    X = (img1 - img1.min()) / (img1.max() - img1.min())
    Y = (img2 - img2.min()) / (img2.max() - img2.min())
    X=torch.DoubleTensor(X).unsqueeze(1)
    Y=torch.DoubleTensor(Y).unsqueeze(1)
    assert not torch.is_complex(X)
    assert not torch.is_complex(Y)
    win_size = 7
    k1 = 0.01
    k2 = 0.03
    w = torch.ones(1, 1, win_size, win_size).to(X) / win_size ** 2
    NP = win_size ** 2
    cov_norm = NP / (NP - 1)
    data_range = Y.max()
    #data_range = data_range[:, None, None, None]
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2
    ux = F.conv2d(X, w)
    uy = F.conv2d(Y, w)
    uxx = F.conv2d(X * X, w)
    uyy = F.conv2d(Y * Y, w)
    uxy = F.conv2d(X * Y, w)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)
    A1, A2, B1, B2 = (
        2 * ux * uy + C1,
        2 * vxy + C2,
        ux ** 2 + uy ** 2 + C1,
        vx + vy + C2,
    )
    D = B1 * B2
    S = (A1 * A2) / D
    return S.mean().item()
def get_psnr(img1, img2):
    img1=sitk.GetArrayFromImage(img1)
    img2=sitk.GetArrayFromImage(img2)
    img1 = (img1 - img1.min()) / (img1.max() - img1.min())
    img2=(img2-img2.min())/(img2.max()-img2.min())
    mse = np.mean((img1 - img2) ** 2)
    rmse = np.sqrt(mse)

    # if mse < 1.0e-10:
    #   return 100
    return 10 * math.log10(1/mse),rmse

if __name__ == '__main__':

    # writer = SummaryWriter('./log')

    # -----------------------
    # parameters settings
    # -----------------------
    parser = argparse.ArgumentParser()

    # about ArSSR model
    parser.add_argument('-encoder_name', type=str, default='RDN', dest='encoder_name',
                        help='the type of encoder network, including RDN (default), ResCNN, and SRResnet.')
    parser.add_argument('-decoder_depth', type=int, default=8, dest='decoder_depth',
                        help='the depth of the decoder network (default=8).')
    parser.add_argument('-decoder_width', type=int, default=192, dest='decoder_width',
                        help='the width of the decoder network (default=256).')
    parser.add_argument('-feature_dim', type=int, default=96, dest='feature_dim',
                        help='the dimension size of the feature vector (default=128)')

    # about training and validation data
    parser.add_argument('-hr_data_train', type=str, default='./data/hr_train', dest='hr_data_train',
                        help='the file path of HR patches for training')
    parser.add_argument('-hr_data_val', type=str, default='./data/hr_val', dest='hr_data_val',
                        help='the file path of HR patches for validation')

    # about training hyper-parameters
    parser.add_argument('-lr', type=float, default=1e-4, dest='lr',
                        help='the initial learning rate')
    parser.add_argument('-lr_decay_epoch', type=int, default=200, dest='lr_decay_epoch',
                        help='learning rate multiply by 0.5 per lr_decay_epoch .')
    parser.add_argument('-epoch', type=int, default=2500, dest='epoch',
                        help='the total number of epochs for training')
    parser.add_argument('-summary_epoch', type=int, default=200, dest='summary_epoch',
                        help='the current model will be saved per summary_epoch')
    parser.add_argument('-bs', type=int, default=1, dest='batch_size',
                        help='the number of LR-HR patch pairs (i.e., N in Equ. 3)')
    parser.add_argument('-ss', type=int, default=8000, dest='sample_size',
                        help='the number of sampled voxel coordinates (i.e., K in Equ. 3)')
    parser.add_argument('-gpu', type=int, default=0, dest='gpu',
                        help='the number of GPU')

    args = parser.parse_args()
    encoder_name = args.encoder_name
    decoder_depth = args.decoder_depth
    decoder_width = args.decoder_width
    feature_dim = args.feature_dim
    hr_data_train = args.hr_data_train
    hr_data_val = args.hr_data_val
    lr = args.lr
    lr_decay_epoch = args.lr_decay_epoch
    epoch = args.epoch
    summary_epoch = args.summary_epoch
    batch_size = args.batch_size
    sample_size = args.sample_size
    gpu = args.gpu
    N_EPOCHS=5000
    import time
    time_start = time.time()  # 记录开始时间
    # -----------------------
    # load data
    # -----------------------
    # train_loader = load_data_train(1)  # create a dataset given opt.dataset_mode and other options
    dataset = custom_dataset()
    # dataset = MyCustomDataset(my_path)
    batch_size = 1
    validation_split = .2
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    # split = int(np.floor(validation_split * dataset_size))
    # split=33
    # if shuffle_dataset :
    #     np.random.seed(random_seed)
    #     np.random.shuffle(indices)
    val_indices = indices[:]

    # Creating PT data samplers and loaders:
    valid_sampler = SubsetRandomSampler(val_indices)

    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)

    # -----------------------
    # model & optimizer
    # -----------------------
    DEVICE = torch.device('cuda:{}'.format(str(gpu) if torch.cuda.is_available() else 'cpu'))
    ArSSR = model.ArSSR(encoder_name=encoder_name, feature_dim=feature_dim,
                        decoder_depth=int(decoder_depth / 2), decoder_width=decoder_width).to(DEVICE)
    ArSSR.load_state_dict(torch.load("ArSSR_best.pth"), strict=True)
    # ArSSR = torch.load("ArSSR_motion_correct.pth").to(DEVICE)
    loss_fun = torch.nn.L1Loss()
    loss_fun2 = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=ArSSR.parameters(), lr=lr)

    # -----------------------
    # training & validation
    # -----------------------
    best_psnr = 0

    with tqdm.tqdm(total=len(val_loader)) as pbar:
        if not os.path.exists("./res/"):
            os.makedirs("./res/")
        with torch.no_grad():
            ArSSR=ArSSR.eval()
            for step, batch in enumerate(val_loader):
                for k, v in batch.items():
                    batch[k] = v

                hr = batch["hr"].cuda().float()
                coord_gt = batch["coord_gt"].cuda().float()
                sda = batch["sda"].cuda().float()
                atlas = batch["atlas"].cuda().float()
                coord_sda = batch["coord_sda"].cuda().float()
                name=batch["name"][0]

                img_hr = hr.float().reshape(batch_size, 1,-1).transpose(2,1)  # N×K×1 (K Equ. 3)
                intensity = ArSSR(sda,coord_sda,atlas,8)
                intensity=intensity.transpose(2,1).reshape(hr.shape).squeeze().detach().cpu().numpy()


                # img_prediction=sitk.GetImageFromArray(intensity)
                # sitk.WriteImage(img_prediction,"./res/pred_{}".format(str(name)))

                # intensity = hr.squeeze().detach().cpu().numpy()
                # img_gt = sitk.GetImageFromArray(intensity)
                # sitk.WriteImage(img_gt, "./res/gt_{}".format(str(name)))



                ref=sitk.ReadImage("/public/bme/home/huangshj/code/MICCAI2023/dataset/atlas/{}".format(name))
                img_prediction=sitk.GetImageFromArray(intensity.transpose(2,1,0))
                img_prediction=resize_image(img_prediction,ref.GetSize(),sitk.sitkLinear)
                img_prediction.CopyInformation(ref)
                sitk.WriteImage(img_prediction,"./res/pred_{}".format(name))

                intensity = hr.squeeze().detach().cpu().numpy()
                img_gt = sitk.GetImageFromArray(intensity.transpose(2,1,0))
                img_gt = resize_image(img_gt, ref.GetSize(),sitk.sitkLinear)
                img_gt.CopyInformation(ref)
                sitk.WriteImage(img_gt, "./res/gt_{}".format(name))





                

                pbar.update()
                pbar.set_description(f"step {step}/{len(val_loader)}")

    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print(time_sum)





def psnr_writer(list_a,metric):
    mid_np = np.array(list_a)  # 列表转数组
    mid_np_2f = np.round(mid_np, 4)  # 对数组中的元素保留两位小数
    list_a = list(mid_np_2f)

    print(metric, list_a)
    print(metric,round(mean(list_a),4),round(std(list_a),4))

root_path= "./res/"
psnr_list=[]
ssim_list=[]
correlation_list=[]
root_path_list=os.listdir(root_path)
for path in root_path_list:
    if "gt" in path:
        gt=os.path.join(root_path,path)
        pre=os.path.join(root_path,path.replace("gt","pred"))
        gt=sitk.ReadImage(gt)
        pre=sitk.ReadImage(pre)
        psnr,rmse=get_psnr(gt,pre)
        ssim=get_ssim(gt,pre)
        correlation=get_correlation(gt,pre)

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        correlation_list.append(rmse)

psnr_writer(psnr_list,"psnr")
psnr_writer(ssim_list,"ssim")
psnr_writer(correlation_list,"rmse")
print("\n")