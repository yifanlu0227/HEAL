"""
Implementation of Simple transformer fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia


class TransformerMessage(nn.Module):
    def __init__(self, 
                 in_channels=64,
                 trans_layer=[3]):
        super(TransformerMessage, self).__init__()
        self.in_channels = in_channels

        self.trans_layer = trans_layer

        dropout = 0
        nhead = 8
        for c_layer in self.trans_layer:
            d_model = in_channels
            cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            # Implementation of Feedforward model
            linear1 = nn.Linear(d_model, d_model)
            linear2 = nn.Linear(d_model, d_model)

            norm1 = nn.LayerNorm(d_model)
            norm2 = nn.LayerNorm(d_model)
            dropout0 = nn.Dropout(dropout)
            dropout1 = nn.Dropout(dropout)
            dropout2 = nn.Dropout(dropout)
            self.__setattr__('cross_attn'+str(c_layer), cross_attn)
            self.__setattr__('linear1_'+str(c_layer), linear1)
            self.__setattr__('linear2_'+str(c_layer), linear2)
            self.__setattr__('norm1_'+str(c_layer), norm1)
            self.__setattr__('norm2_'+str(c_layer), norm2)
            self.__setattr__('dropout0_'+str(c_layer), dropout0)
            self.__setattr__('dropout1_'+str(c_layer), dropout1)
            self.__setattr__('dropout2_'+str(c_layer), dropout2)

    def add_pe_map(self, x, normalized=True):
        """ Add positional encoding to feature map.
        Args:
            x: torch.Tensor
                [N, C, H, W]
        
        """
        # scale = 2 * math.pi
        temperature = 10000 
        num_pos_feats = x.shape[-3] // 2  # d

        mask = torch.zeros([x.shape[-2], x.shape[-1]], dtype=torch.bool, device=x.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(0, dtype=torch.float32)
        x_embed = not_mask.cumsum(1, dtype=torch.float32)

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)

        if len(x.shape) == 5:
            x = x + pos[None,None,:,:,:]
        elif len(x.shape) == 6:
            x = x + pos[None,None,None,:,:,:]
        return x

    def forward(self, x, shift_mats, shift_mats_rev, agent_mask):
        batch, max_agent_num, c, h, w = x.shape

        # ================================
        # First, transform to each coord
        feat_shifted = []
        for agent_i in range(max_agent_num):
            # shift_mat_i = shift_mats[:, agent_i, :, :]
            shift_mat_rev_i = shift_mats_rev[:, agent_i, :, :]
            feat_i = x[:, agent_i, :, :, :]
            feat_shifted_i = []
            for agent_j in range(max_agent_num):
                shift_mat_j = shift_mats[:, agent_j, :, :]
                shift_mat = shift_mat_j.view(batch, 3, 3) @ shift_mat_rev_i.view(batch, 3, 3)
                feat = kornia.warp_perspective(feat_i, shift_mat, dsize=(100 * 2, 100 * 2), align_corners=False)
                feat_shifted_i.append(feat)
            feat_shifted_i = torch.cat([f.unsqueeze(1) for f in feat_shifted_i], dim=1)
            feat_shifted.append(feat_shifted_i)
        feat_shifted = torch.cat([f.unsqueeze(1) for f in feat_shifted], dim=1)


        # ================================
        # x_fuse, _, _ = self.TRANSFORMER_MESSAGE([[],[],[],local_com_mat], [transformed_feature], num_agent_tensor)

        for i, c_layer in enumerate(self.trans_layer):
            batch_updated_features = torch.zeros(batch, max_agent_num, c, h, w).to(shift_mats.device)
            for batch_i in range(batch):
                N = int(torch.sum(agent_mask[batch_i]))
                feat_map = x[batch_i:batch_i+1, :N, :, :, :]
                val_feat = feat_shifted[batch_i:batch_i+1, :N, :N, :, :, :]

                feat_map = self.add_pe_map(feat_map)
                # [b,N,C,H,W] -> [b,N,H,W,C]
                # [b,N,N,C,H,W] -> [N,b,N,H,W,C]
                src = feat_map.permute(0,1,3,4,2).contiguous().view(N*h*w,c).contiguous().unsqueeze(0)
                tgt = val_feat.permute(1,0,2,4,5,3).contiguous().view(N, N*h*w,c).contiguous()
                # print(src.shape)  # torch.Size([1, 120000, 64])
                # print(tgt.shape)  # torch.Size([N, 120000, 64])

                src2, weight_mat = eval('self.cross_attn'+str(c_layer))(src, tgt, value=tgt, attn_mask=None, key_padding_mask=None)
                src = src + eval('self.dropout1_'+str(c_layer))(src2)
                src = eval('self.norm1_'+str(c_layer))(src)
                src2 = eval('self.linear2_'+str(c_layer))(eval('self.dropout0_'+str(c_layer))(F.relu(eval('self.linear1_'+str(c_layer))(src))))
                src = src + eval('self.dropout2_'+str(c_layer))(src2)
                src = eval('self.norm2_'+str(c_layer))(src)

                feat_fuse = src.view(1, N, h, w, c).contiguous().permute(0, 1, 4, 2, 3).contiguous()
                # print(feat_fuse.shape)  # torch.Size([1, N, 64, 200, 200])
                batch_updated_features[batch_i, :N, :, :, :] = feat_fuse.squeeze(0)
        
        return batch_updated_features, None


if __name__=="__main__":
    from icecream import ic
    x = torch.rand((64,6,8))  # [C,H,W]
    temperature = 10000
    num_pos_feats = x.shape[-3] // 2  # [d]

    mask = torch.zeros([x.shape[-2], x.shape[-1]], dtype=torch.bool, device=x.device)  #[H, W]
    not_mask = ~mask
    y_embed = not_mask.cumsum(0, dtype=torch.float32)  # [H, W]
    x_embed = not_mask.cumsum(1, dtype=torch.float32)  # [H, W]

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)  # [0,1,2,...,d]
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)  # 10000^(2k/d), k is [0,0,1,1,...,d/2,d/2]

    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t

    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)
