# -*- coding:utf-8 -*-
# -----------------------------------------
#   Filename: train.py
#   Author  : Qing Wu
#   Email   : wuqing@shanghaitech.edu.cn
#   Date    : 2021/9/19
# -----------------------------------------
import argparse
import model
# from torch.utils.tensorboard import SummaryWriter
from data_coord import custom_dataset
import SimpleITK as sitk
import torch
import tqdm
from grid_loss import Grad,get_psnr,Grad_z
import numpy as np

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
    sample_size = args.sample_size
    gpu = args.gpu
    N_EPOCHS=1000

    # -----------------------
    # load data
    # -----------------------
    from torch.utils.data import SubsetRandomSampler
    dataset = custom_dataset()
    # dataset = MyCustomDataset(my_path)
    batch_size = 1
    validation_split = .2
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)

    # -----------------------
    # model & optimizer
    # -----------------------
    DEVICE = torch.device('cuda:{}'.format(str(gpu) if torch.cuda.is_available() else 'cpu'))
    ArSSR = model.ArSSR(encoder_name=encoder_name, feature_dim=feature_dim,
                        decoder_depth=int(decoder_depth / 2), decoder_width=decoder_width).to(DEVICE)
    # ArSSR.load_state_dict(torch.load("ArSSR_best.pth"), strict=True)
    # ArSSR = torch.load("ArSSR_motion_correct.pth").to(DEVICE)
    loss_fun = torch.nn.L1Loss()
    loss_fun2 = torch.nn.MSELoss()
    loss_smooth=Grad()
    loss_smooth_z=Grad_z()
    optimizer = torch.optim.Adam(params=ArSSR.parameters(), lr=lr)

    # -----------------------
    # training & validation
    # -----------------------
    best_psnr = 0
    best_psnr_L = 0
    for epoch in range(N_EPOCHS):
        with tqdm.tqdm(total=len(train_loader)) as pbar:
            ArSSR.train()
            loss_train = 0
            for step, batch in enumerate(train_loader):
                for k, v in batch.items():
                    batch[k] = v

                hr = batch["hr"].cuda().float()
                coord_gt = batch["coord_gt"].cuda().float()
                sda = batch["sda"].cuda().float()
                atlas = batch["atlas"].cuda().float()
                coord_sda = batch["coord_sda"].cuda().float()

                img_hr = hr.float().reshape(batch_size, 1,-1).transpose(2,1)  # N×K×1 (K Equ. 3)
                intensity = ArSSR(sda,coord_sda,atlas,8)
                # intensity_lr=intensity.clone().detach()
                # Z=torch.zeros_like(A)
                loss = loss_fun(intensity, img_hr)
                # loss = loss_fun(intensity, img_hr)

                # if step%30==0:
                #     intensity_lr = intensity.clone().detach()
                #     intensity = intensity.transpose(2, 1).reshape(hr.shape).squeeze().detach().cpu().numpy()
                #     img_prediction = sitk.GetImageFromArray(intensity)
                #     sitk.WriteImage(img_prediction,"pred.nii.gz")
                #     intensity = hr.squeeze().detach().cpu().numpy()
                #     img_gt = sitk.GetImageFromArray(intensity)
                #     sitk.WriteImage(img_gt, "gt.nii.gz")
                #     intensity = atlas.squeeze().detach().cpu().numpy()
                #     img_atlas = sitk.GetImageFromArray(intensity)
                #     sitk.WriteImage(img_atlas, "atlas.nii.gz")
                #     intensity = sda.squeeze().detach().cpu().numpy()
                #     img_sda = sitk.GetImageFromArray(intensity)
                #     sitk.WriteImage(img_sda, "sda.nii.gz")
                #     psnr, _ = get_psnr(img_prediction, img_gt)
                #     print("PSNR: ", psnr)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update()
                pbar.set_description(f"Epoch {epoch}/{N_EPOCHS} | step {step}/{len(train_loader)} Loss: {loss.item()}")





        with tqdm.tqdm(total=len(val_loader)) as pbar:
            with torch.no_grad():
                ArSSR=ArSSR.eval()
                psnr_total=0
                psnr_total_L=0
                for step, batch in enumerate(val_loader):
                    for k, v in batch.items():
                        batch[k] = v

                    hr = batch["hr"].cuda().float()
                    coord_gt = batch["coord_gt"].cuda().float()
                    sda = batch["sda"].cuda().float()
                    coord_sda = batch["coord_sda"].cuda().float()
                    atlas = batch["atlas"].cuda().float()

                    img_hr = hr.float().reshape(batch_size, 1, -1).transpose(2, 1)  # N×K×1 (K Equ. 3)
                    intensity = ArSSR(sda, coord_sda,atlas,8)
                    intensity_lr=intensity.clone().detach()
                    intensity=intensity.transpose(2,1).reshape(hr.shape).squeeze().detach().cpu().numpy()
                    img_prediction=sitk.GetImageFromArray(intensity)
                    intensity = hr.squeeze().detach().cpu().numpy()
                    img_gt = sitk.GetImageFromArray(intensity)

                    psnr,_=get_psnr(img_prediction,img_gt)
                    psnr_total+=psnr




                    pbar.update()
                    pbar.set_description(f"step {step}/{len(val_loader)}")
                if psnr_total>best_psnr:
                    best_psnr=psnr_total
                    torch.save(ArSSR.state_dict(), "ArSSR_best.pth")
                intensity = atlas.squeeze().detach().cpu().numpy()
                img_atlas = sitk.GetImageFromArray(intensity)
                intensity = sda.squeeze().detach().cpu().numpy()
                img_sda = sitk.GetImageFromArray(intensity)
                sitk.WriteImage(img_gt, "gt_{}.nii.gz".format(str(step)))
                sitk.WriteImage(img_prediction, "pred_{}.nii.gz".format(str(step)))
                sitk.WriteImage(img_atlas, "atlas_{}.nii.gz".format(str(step)))
                sitk.WriteImage(img_sda, "sda_{}.nii.gz".format(str(step)))
                torch.save(ArSSR.state_dict(), "ArSSR_motion_correct.pth")
                print("BEST_PSNR: ", best_psnr/(step+1), psnr_total/(step+1))

        lr = None
        coord_lr = None
        hr = None
        coord_hr = None
        coord_gt = None
