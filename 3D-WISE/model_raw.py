# -*- coding:utf-8 -*-
# -----------------------------------------
#   Filename: models.py
#   Author  : Qing Wu
#   Email   : wuqing@shanghaitech.edu.cn
#   Date    : 2021/9/19
# -----------------------------------------
import torch.nn as nn
import torch
import torch.nn.functional as F
import decoder
import encoder
import SimpleITK as sitk

class ArSSR(nn.Module):
    def __init__(self, encoder_name, feature_dim, decoder_depth, decoder_width):
        super(ArSSR, self).__init__()
        if encoder_name == 'RDN':
            self.encoder_a = encoder.RDN(feature_dim=feature_dim)
            self.encoder_c = encoder.RDN(feature_dim=feature_dim)
            self.encoder_s = encoder.RDN(feature_dim=feature_dim)
            self.encoder_atlas = encoder.RDN(feature_dim=feature_dim)
            # self.encoder_c2 = encoder.RDN(feature_dim=feature_dim)
            # self.encoder_s2 = encoder.RDN(feature_dim=feature_dim)
        if encoder_name == 'SRResnet':
            self.encoder = encoder.SRResnet(feature_dim=feature_dim)
        if encoder_name == 'ResCNN':
            self.encoder = encoder.ResCNN(feature_dim=feature_dim)
        self.decoder = decoder.MLP(in_dim=feature_dim + 3, out_dim=1, depth=decoder_depth, width=decoder_width)

        self.layers = nn.Sequential(
            nn.Linear(in_features=256, out_features=256, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=3, bias=True),
            # nn.Sigmoid()
        )
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)
        # self.query_linear_axi = nn.Linear(280, 280)
        # self.key_linear_axi = nn.Linear(280, 280)
        # self.value_linear_axi = nn.Linear(280, 280)
        #
        # self.query_linear_cor = nn.Linear(224, 224)
        # self.key_linear_cor = nn.Linear(224, 224)
        # self.value_linear_cor = nn.Linear(224, 224)
        #
        # self.query_linear_sag = nn.Linear(240, 240)
        # self.key_linear_sag = nn.Linear(240, 240)
        # self.value_linear_sag = nn.Linear(240, 240)

        self.query_linear_axi = nn.Linear(140, 140)
        self.key_linear_axi = nn.Linear(140, 140)
        self.value_linear_axi = nn.Linear(140, 140)

        self.query_linear_cor = nn.Linear(112, 112)
        self.key_linear_cor = nn.Linear(112, 112)
        self.value_linear_cor = nn.Linear(112, 112)

        self.query_linear_sag = nn.Linear(160, 160)
        self.key_linear_sag = nn.Linear(160, 160)
        self.value_linear_sag = nn.Linear(160, 160)

    def pad_to_nearest_ten(self,tensor):
        # 获取原始tensor的后三维大小
        shape = tensor.shape[-3:]

        # 计算每个维度距离最近的10的倍数
        padded_shape = [int(((size + 9) // 10) * 10) for size in shape]

        # 计算每个维度需要填充的大小
        padding = [padded - original for padded, original in zip(padded_shape, shape)]

        # 使用pad函数进行填充
        padded_tensor = torch.nn.functional.pad(tensor, (padding[2]//2, padding[2]-padding[2]//2, padding[1]//2, padding[1]-padding[1]//2, padding[0]//2, padding[0]-padding[0]//2))

        unfolded_axi = padded_tensor.unfold(2, padded_shape[0]//10, padded_shape[0]//10).unfold(3, padded_shape[1]//10, padded_shape[1]//10).unfold(4, padded_shape[2]//10, padded_shape[2]//10)
        return unfolded_axi,padded_tensor,padding[0]//2, padding[0]-padding[0]//2, padding[1]//2, padding[1]-padding[1]//2, padding[2]//2, padding[2]-padding[2]//2

    def forward(self, lr, coord_lr, coord_hr,atlas):
        """
        :param img_lr: N×1×h×w×d
        :param xyz_hr: N×K×3
        Note that,
            N: batch size  (N in Equ. 3)
            K: coordinate sample size (K in Equ. 3)
            {h,w,d}: dimensional size of LR input image
        """

        lr[0] = lr[0].float().cuda()
        lr[1]=lr[1].float().cuda()
        lr[2]=lr[2].float().cuda()
        coord_lr[0] = coord_lr[0].float().cuda()
        coord_lr[1] = coord_lr[1].float().cuda()
        coord_lr[2] = coord_lr[2].float().cuda()

        cell = torch.zeros((1, coord_hr[0].reshape(3, -1).shape[1], 3)).cuda()
        cell[:, :, 0] = 2 / coord_hr[0].shape[2]
        cell[:, :, 1] = 2 / coord_hr[0].shape[3]
        cell[:, :, 2] = 2 / coord_hr[0].shape[4]

        coord_hr_axi = coord_hr[0].cuda().float().reshape(1, 3, -1).permute(0, 2, 1)
        coord_hr_cor = coord_hr[1].cuda().float().reshape(1, 3, -1).permute(0, 2, 1)
        coord_hr_sag = coord_hr[2].cuda().float().reshape(1, 3, -1).permute(0, 2, 1)
        self.axi = self.encoder_a(lr[0])
        self.cor = self.encoder_c(lr[1])
        self.sag = self.encoder_s(lr[2])

        self.atlas = self.encoder_atlas(atlas)



        unfolded_atlas,pad_atlas,_,_,_,_,_,_ = self.pad_to_nearest_ten(self.atlas)
        atlas_cube = unfolded_atlas.reshape(1, -1, unfolded_atlas.shape[-3], unfolded_atlas.shape[-2],
                                            unfolded_atlas.shape[-1])

        unfolded_axi,pad_axi,p1,p2,p3,p4,p5,p6 = self.pad_to_nearest_ten(self.axi)
        axi_cube=unfolded_axi.reshape(1,-1,unfolded_axi.shape[-3],unfolded_axi.shape[-2],unfolded_axi.shape[-1])
        atlas_cube=torch.nn.functional.interpolate(atlas_cube, axi_cube.shape[-3:])
        A = atlas_cube.reshape(atlas_cube.shape[0], atlas_cube.shape[1], -1)  # Shape: 1x96x13824
        B = axi_cube.reshape(axi_cube.shape[0], axi_cube.shape[1], -1)  # Shape: 1x96x13824
        query = self.query_linear_axi(A)  # Shape: 1x96x13824
        key = self.key_linear_axi(B)  # Shape: 1x96x13824
        value = self.value_linear_axi(B)  # Shap e: 1x96x13824
        attn_weights = torch.softmax(torch.bmm(query.transpose(1, 2), key), dim=-1)  # Shape: 1x13824x13824
        cross_attention = torch.bmm(value, attn_weights.transpose(1, 2))  # Shape: 1x96x13824
        self.feat_axi = cross_attention.reshape(unfolded_axi.shape).permute(0,1,2,5,3,6,4,7).reshape(pad_axi.shape)[:,:,p1:-p2,p3:-p4,p5:-p6]




        unfolded_cor, pad_cor, p1, p2, p3, p4, p5, p6 = self.pad_to_nearest_ten(self.cor.permute(0,1,2,4,3))
        cor_cube = unfolded_cor.reshape(1, -1, unfolded_cor.shape[-3], unfolded_cor.shape[-2], unfolded_cor.shape[-1])
        atlas_cube = torch.nn.functional.interpolate(atlas_cube, cor_cube.shape[-3:])
        A = atlas_cube.reshape(atlas_cube.shape[0], atlas_cube.shape[1], -1)  # Shape: 1x96x13824
        B = cor_cube.reshape(cor_cube.shape[0], cor_cube.shape[1], -1)  # Shape: 1x96x13824
        query = self.query_linear_cor(A)  # Shape: 1x96x13824
        key = self.key_linear_cor(B)  # Shape: 1x96x13824
        value = self.value_linear_cor(B)  # Shap e: 1x96x13824
        attn_weights = torch.softmax(torch.bmm(query.transpose(1, 2), key), dim=-1)  # Shape: 1x13824x13824
        cross_attention = torch.bmm(value, attn_weights.transpose(1, 2))  # Shape: 1x96x13824
        self.feat_cor = cross_attention.reshape(unfolded_cor.shape).permute(0, 1, 2, 5, 3, 6, 4, 7).reshape(
            pad_cor.shape)[:, :, p1:-p2, p3:-p4, p5:-p6].permute(0,1,2,4,3)



        unfolded_sag, pad_sag, p1, p2, p3, p4, p5, p6 = self.pad_to_nearest_ten(self.sag.permute(0,1,4,3,2))
        sag_cube = unfolded_sag.reshape(1, -1, unfolded_sag.shape[-3], unfolded_sag.shape[-2], unfolded_sag.shape[-1])
        atlas_cube = torch.nn.functional.interpolate(atlas_cube, sag_cube.shape[-3:])
        A = atlas_cube.reshape(atlas_cube.shape[0], atlas_cube.shape[1], -1)  # Shape: 1x96x13824
        B = sag_cube.reshape(sag_cube.shape[0], sag_cube.shape[1], -1)  # Shape: 1x96x13824
        query = self.query_linear_sag(A)  # Shape: 1x96x13824
        key = self.key_linear_sag(B)  # Shape: 1x96x13824
        value = self.value_linear_sag(B)  # Shap e: 1x96x13824
        attn_weights = torch.softmax(torch.bmm(query.transpose(1, 2), key), dim=-1)  # Shape: 1x13824x13824
        cross_attention = torch.bmm(value, attn_weights.transpose(1, 2))  # Shape: 1x96x13824
        self.feat_sag = cross_attention.reshape(unfolded_sag.shape).permute(0, 1, 2, 5, 3, 6, 4, 7).reshape(
            pad_sag.shape)[:, :, p1:-p2, p3:-p4, p5:-p6].permute(0,1,4,3,2)


        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        vz_lst = [-1, 1]

        feat = self.feat_axi
        eps_shift = 1e-6
        rx = 2 / feat.shape[-3] / 2
        ry = 2 / feat.shape[-2] / 2
        rz = 2 / feat.shape[-1] / 2

        feat_coord = coord_lr[0]
        inp_o = lr[0].clone()

        preds = torch.zeros((24, 1, coord_hr_axi.shape[1], 1)).cuda()
        p_index=0
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                for vz in vz_lst:
                    coord_ = coord_hr_axi.clone()
                    coord_[:, :, 0] += vx * rx + eps_shift
                    coord_[:, :, 1] += vy * ry + eps_shift
                    coord_[:, :, 2] += vz * rz + eps_shift
                    coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                    q_feat = F.grid_sample(
                        feat, coord_.flip(-1).unsqueeze(1).unsqueeze(1),
                        mode='bilinear', align_corners=False)[:, :, 0, 0, :] \
                        .permute(0, 2, 1)
                    q_coord = F.grid_sample(
                        feat_coord, coord_.flip(-1).unsqueeze(1).unsqueeze(1),
                        mode='bilinear', align_corners=False)[:, :, 0, 0, :] \
                        .permute(0, 2, 1)
                    rel_coord = coord_hr_axi - q_coord
                    rel_coord[:, :, 0] *= feat.shape[-3]
                    rel_coord[:, :, 1] *= feat.shape[-2]
                    rel_coord[:, :, 2] *= feat.shape[-1]
                    inp = torch.cat([q_feat, rel_coord], dim=-1)

                    # if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-3]
                    rel_cell[:, :, 1] *= feat.shape[-2]
                    rel_cell[:, :, 2] *= feat.shape[-1]

                    inp = torch.cat([inp, rel_cell], dim=-1)
                    bs, q = coord_hr_axi.shape[:2]
                    pred = self.decoder(inp.view(bs * q, -1)).view(bs, q, -1)
                    preds[p_index]=pred
                    p_index+=1
                    area=F.grid_sample(
                        inp_o, coord_.flip(-1).unsqueeze(1).unsqueeze(1),
                        mode='bilinear', align_corners=False)[:, :, 0, 0, :] \
                        .permute(0, 2, 1)

                    # area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1] * rel_coord[:, :, 2])
                    areas.append(area + 1e-9)




        feat = self.feat_cor

        eps_shift = 1e-6
        rx = 2 / feat.shape[-3] / 2
        ry = 2 / feat.shape[-2] / 2
        rz = 2 / feat.shape[-1] / 2

        feat_coord = coord_lr[1]
        inp_o = lr[1].clone()

        # preds = []
        # areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                for vz in vz_lst:
                    coord_ = coord_hr_cor.clone()
                    coord_[:, :, 0] += vx * rx + eps_shift
                    coord_[:, :, 1] += vy * ry + eps_shift
                    coord_[:, :, 2] += vz * rz + eps_shift
                    coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                    q_feat = F.grid_sample(
                        feat, coord_.flip(-1).unsqueeze(1).unsqueeze(1),
                        mode='bilinear', align_corners=False)[:, :, 0, 0, :] \
                        .permute(0, 2, 1)
                    q_coord = F.grid_sample(
                        feat_coord, coord_.flip(-1).unsqueeze(1).unsqueeze(1),
                        mode='bilinear', align_corners=False)[:, :, 0, 0, :] \
                        .permute(0, 2, 1)
                    rel_coord = coord_hr_cor - q_coord
                    rel_coord[:, :, 0] *= feat.shape[-3]
                    rel_coord[:, :, 1] *= feat.shape[-2]
                    rel_coord[:, :, 2] *= feat.shape[-1]
                    inp = torch.cat([q_feat, rel_coord], dim=-1)

                    # if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-3]
                    rel_cell[:, :, 1] *= feat.shape[-2]
                    rel_cell[:, :, 2] *= feat.shape[-1]

                    inp = torch.cat([inp, rel_cell], dim=-1)
                    bs, q = coord_hr_cor.shape[:2]
                    pred = self.decoder(inp.view(bs * q, -1)).view(bs, q, -1)
                    preds[p_index]=pred
                    p_index+=1

                    area = F.grid_sample(
                        inp_o, coord_.flip(-1).unsqueeze(1).unsqueeze(1),
                        mode='bilinear', align_corners=False)[:, :, 0, 0, :] \
                        .permute(0, 2, 1)
                    # area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1] * rel_coord[:, :, 2])
                    areas.append(area + 1e-9)



        feat = self.feat_sag

        eps_shift = 1e-6
        rx = 2 / feat.shape[-3] / 2
        ry = 2 / feat.shape[-2] / 2
        rz = 2 / feat.shape[-1] / 2

        feat_coord = coord_lr[2]
        inp_o = lr[2].clone()

        # preds = []
        # areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                for vz in vz_lst:
                    coord_ = coord_hr_sag.clone()
                    coord_[:, :, 0] += vx * rx + eps_shift
                    coord_[:, :, 1] += vy * ry + eps_shift
                    coord_[:, :, 2] += vz * rz + eps_shift
                    coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                    q_feat = F.grid_sample(
                        feat, coord_.flip(-1).unsqueeze(1).unsqueeze(1),
                        mode='bilinear', align_corners=False)[:, :, 0, 0, :] \
                        .permute(0, 2, 1)
                    q_coord = F.grid_sample(
                        feat_coord, coord_.flip(-1).unsqueeze(1).unsqueeze(1),
                        mode='bilinear', align_corners=False)[:, :, 0, 0, :] \
                        .permute(0, 2, 1)
                    rel_coord = coord_hr_sag - q_coord
                    rel_coord[:, :, 0] *= feat.shape[-3]
                    rel_coord[:, :, 1] *= feat.shape[-2]
                    rel_coord[:, :, 2] *= feat.shape[-1]
                    inp = torch.cat([q_feat, rel_coord], dim=-1)

                    # if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-3]
                    rel_cell[:, :, 1] *= feat.shape[-2]
                    rel_cell[:, :, 2] *= feat.shape[-1]

                    inp = torch.cat([inp, rel_cell], dim=-1)
                    bs, q = coord_hr_sag.shape[:2]
                    pred = self.decoder(inp.view(bs * q, -1)).view(bs, q, -1)
                    preds[p_index]=pred
                    p_index+=1

                    area = F.grid_sample(
                        inp_o, coord_.flip(-1).unsqueeze(1).unsqueeze(1),
                        mode='bilinear', align_corners=False)[:, :, 0, 0, :] \
                        .permute(0, 2, 1)
                    # area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1] * rel_coord[:, :, 2])
                    areas.append(area + 1e-9)



        ret = 0
        preds = self.softmax(preds)
        for pred, area in zip(preds, areas):
            ret = ret + pred * area
        # ret = self.tanh(ret)
        return ret,self.feat_axi,self.feat_cor,self.feat_sag
