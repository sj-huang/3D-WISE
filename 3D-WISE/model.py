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


def get_embedder(multires=10, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,  # 如果为真，最终的编码结果包含原始坐标
        'input_dims': 3,  # 输入给编码器的数据的维度
        'max_freq_log2': multires - 1,
        'num_freqs': multires,  # 即论文中 5.1 节位置编码公式中的 L
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)  # embed 现在相当于一个编码器，具体的编码公式与论文中的一致。
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0

        # 如果包含原始位置
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)  # 把一个不对数据做出改变的匿名函数添加到列表中
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)  # 得到 [2^0, 2^1, ... ,2^(L-1)] 参考论文 5.1 中的公式
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)  # 得到 [2^0, 2^(L-1)] 的等差数列，列表中有 L 个元素

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))  # sin(x * 2^n)  参考位置编码公式
                out_dim += d  # 每使用子编码公式一次就要把输出维度加 3，因为每个待编码的位置维度是 3

        self.embed_fns = embed_fns  # 相当于是一个编码公式列表
        self.out_dim = out_dim

    def embed(self, inputs):
        # 对各个输入进行编码，给定一个输入，使用编码列表中的公式分别对他编码
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class ArSSR(nn.Module):
    def __init__(self, encoder_name, feature_dim, decoder_depth, decoder_width):
        super(ArSSR, self).__init__()
        if encoder_name == 'RDN':
            self.encoder = encoder.RDN(feature_dim=feature_dim)
            self.encoder_atlas = encoder.RDN(feature_dim=feature_dim)
        if encoder_name == 'SRResnet':
            self.encoder = encoder.SRResnet(feature_dim=feature_dim)
        if encoder_name == 'ResCNN':
            self.encoder = encoder.ResCNN(feature_dim=feature_dim)
        self.decoder = decoder.MLP(in_dim=feature_dim + 3, out_dim=1, depth=decoder_depth, width=decoder_width)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)
        self.query_linear = nn.Linear(96, 96)
        self.key_linear = nn.Linear(96, 96)
        self.value_linear = nn.Linear(96, 96)
        self.pool=nn.AdaptiveAvgPool3d(8)
    def pad_to_nearest_ten(self,tensor,n=10):
        # 获取原始tensor的后三维大小
        shape = tensor.shape[-3:]

        # 计算每个维度距离最近的10的倍数
        padded_shape = [int(((size + n-1) // n) * n) for size in shape]

        # 计算每个维度需要填充的大小
        padding = [padded - original for padded, original in zip(padded_shape, shape)]

        # 使用pad函数进行填充
        padded_tensor = torch.nn.functional.pad(tensor, (padding[2]//2, padding[2]-padding[2]//2, padding[1]//2, padding[1]-padding[1]//2, padding[0]//2, padding[0]-padding[0]//2))

        unfolded_axi = padded_tensor.unfold(2, padded_shape[0]//n, padded_shape[0]//n).unfold(3, padded_shape[1]//n, padded_shape[1]//n).unfold(4, padded_shape[2]//n, padded_shape[2]//n)
        return unfolded_axi,padded_tensor,padding[0]//2, padding[0]-padding[0]//2, padding[1]//2, padding[1]-padding[1]//2, padding[2]//2, padding[2]-padding[2]//2

    def forward(self, hr, coord_gt,atlas,cube_size):
        """
        :param img_lr: N×1×h×w×d
        :param xyz_hr: N×K×3
        Note that,
            N: batch size  (N in Equ. 3)
            K: coordinate sample size (K in Equ. 3)
            {h,w,d}: dimensional size of LR input image
        """
        feat_coord = coord_gt
        self.axi = self.encoder(hr)  # N×1×h×w×d
        self.atlas = self.encoder_atlas(atlas)


        
        A = self.atlas.reshape(self.atlas.shape[0], self.atlas.shape[1], -1).transpose(-1,-2)  # Shape: 1x96x13824
        B = self.axi.reshape(self.axi.shape[0], self.axi.shape[1], -1).transpose(-1,-2)
        query = self.query_linear(A)  # Shape: 1x96x13824
        key = self.key_linear(B)  # Shape: 1x96x13824
        value = self.value_linear(B)  # Shap e: 1x96x13824
        attn_weights = torch.softmax(torch.bmm(query.transpose(1, 2), key), dim=-1)  # Shape: 1x13824x13824
        cross_attention = torch.bmm(value, attn_weights.transpose(1, 2))  # Shape: 1x96x13824
        self.feat_axi = cross_attention.transpose(-1,-2).reshape(self.axi.shape)


        coord_gt = coord_gt.reshape(1, 3, -1).permute(0, 2, 1)

        position_emb,_=get_embedder()

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        vz_lst = [-1, 1]

        feat = self.feat_axi
        eps_shift = 1e-6
        rx = 2 / feat.shape[-3] / 2
        ry = 2 / feat.shape[-2] / 2
        rz = 2 / feat.shape[-1] / 2


        inp_o = hr.clone()
        areas=[]
        intens=[]
        preds = torch.zeros((8, 1, coord_gt.shape[1], 1)).cuda()
        p_index=0
        for vx in vx_lst:
            for vy in vy_lst:
                for vz in vz_lst:
                    coord_ = coord_gt.clone()
                    coord_[:, :, 0] += vx * rx + eps_shift
                    coord_[:, :, 1] += vy * ry + eps_shift
                    coord_[:, :, 2] += vz * rz + eps_shift
                    coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                    q_feat = F.grid_sample(
                        feat, coord_.flip(-1).unsqueeze(1).unsqueeze(1),
                        mode='nearest', align_corners=False)[:, :, 0, 0, :] \
                        .permute(0, 2, 1)
                    q_coord = F.grid_sample(
                        feat_coord, coord_.flip(-1).unsqueeze(1).unsqueeze(1),
                        mode='nearest', align_corners=False)[:, :, 0, 0, :] \
                        .permute(0, 2, 1)
                    rel_coord = coord_gt - q_coord
                    rel_coord[:, :, 0] *= feat.shape[-3]
                    rel_coord[:, :, 1] *= feat.shape[-2]
                    rel_coord[:, :, 2] *= feat.shape[-1]

                    rel_coord = position_emb(rel_coord)

                    inp = torch.cat([q_feat, rel_coord], dim=-1)

                    # inp = torch.cat([inp, rel_cell], dim=-1)
                    bs, q = coord_gt.shape[:2]
                    pred = self.decoder(inp.view(bs * q, -1)).view(bs, q, -1)
                    preds[p_index]=pred
                    p_index+=1
                    # area=F.grid_sample(
                    #     inp_o, coord_.flip(-1).unsqueeze(1).unsqueeze(1),
                    #     mode='nearest', align_corners=False)[:, :, 0, 0, :] \
                    #     .permute(0, 2, 1)

                    # # area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1] * rel_coord[:, :, 2])
                    # areas.append(area + 1e-9)

                    area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1] * rel_coord[:, :, 2])
                    areas.append(area+1e-9)
                    inten = F.grid_sample(
                        inp_o, coord_.flip(-1).unsqueeze(1).unsqueeze(1),
                        mode='bilinear', align_corners=False)[:, :, 0, 0, :] \
                        .permute(0, 2, 1)
                    intens.append((inten+1e-9)/8)

        ret = 0
        tot_area = torch.stack(areas).sum(dim=0)
        tot_inten = torch.stack(intens).sum(dim=0)
        for pred, area, inten in zip(preds, areas, intens):
            ret = ret + inten * pred * (area / tot_area).unsqueeze(-1)
        return ret