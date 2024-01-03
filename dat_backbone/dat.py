# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Vision Transformer with Deformable Attention
# Modified by Zhuofan Xia 
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple

from .slide import SlideAttention
from .dat_blocks import *
from .nat import NeighborhoodAttention2D
from .qna import FusedKQnA

from mmcv.runner import auto_fp16


class LayerScale(nn.Module):

    def __init__(self,
                 dim: int,
                 inplace: bool = False,
                 init_values: float = 1e-5):
        super().__init__()
        self.inplace = inplace
        self.weight = nn.Parameter(torch.ones(dim) * init_values)

    def forward(self, x):
        if self.inplace:
            return x.mul_(self.weight.view(-1, 1, 1))
        else:
            return x * self.weight.view(-1, 1, 1)

class TransformerStage(nn.Module):

    def __init__(self,
                 fmap_size,
                 window_size,
                 ns_per_pt,
                 dim_in,
                 dim_embed,
                 depths,
                 stage_spec,
                 n_groups, 
                 use_pe,
                 sr_ratio,
                 heads,
                 heads_q,
                 stride,
                 offset_range_factor,
                 local_orf,
                 local_kv_size,
                 dwc_pe,
                 no_off,
                 fixed_pe,
                 attn_drop,
                 proj_drop,
                 expansion,
                 drop,
                 drop_path_rate, 
                 use_dwc_mlp,
                 ksize,
                 nat_ksize,
                 k_qna,
                 nq_qna,
                 qna_activation,
                 deform_groups, 
                 layer_scale_value, 
                 use_lpu,
                 use_cmt_mlp,
                 log_cpb, 
                 stage_i, 
                 use_checkpoint):

        super().__init__()
        self.fp16_enabled = False
        self.use_checkpoint = use_checkpoint
        fmap_size = to_2tuple(fmap_size)
        local_kv_size = to_2tuple(local_kv_size)
        self.depths = depths
        hc = dim_embed // heads
        assert dim_embed == heads * hc
        self.proj = nn.Conv2d(dim_in, dim_embed, 1, 1, 0) if dim_in != dim_embed else nn.Identity()
        self.use_lpu = use_lpu
        self.stage_spec = stage_spec

        self.ln_cnvnxt = nn.ModuleDict(
            {str(d): LayerNormProxy(dim_embed) for d in range(depths) if stage_spec[d] == 'X'}
        )
        self.layer_norms = nn.ModuleList(
            [LayerNormProxy(dim_embed) if stage_spec[d // 2] != 'X' else nn.Identity() for d in range(2 * depths)]
        )

        mlp_fn = TransformerMLP
        if use_dwc_mlp:
            if use_cmt_mlp:
                mlp_fn = TransformerMLPWithConv_CMT
            else:
                mlp_fn = TransformerMLPWithConv

        self.mlps = nn.ModuleList(
            [ 
                mlp_fn(dim_embed, expansion, drop) for _ in range(depths)
            ]
        )
        self.attns = nn.ModuleList()
        self.drop_path = nn.ModuleList()
        self.layer_scales = nn.ModuleList(
            [
                LayerScale(dim_embed, init_values=layer_scale_value) if layer_scale_value > 0.0 else nn.Identity() 
                for _ in range(2 * depths)
            ]
        )
        self.local_perception_units = nn.ModuleList(
            [
                nn.Conv2d(dim_embed, dim_embed, kernel_size=3, stride=1, padding=1, groups=dim_embed) if use_lpu else nn.Identity()
                for _ in range(depths)
            ]
        )
        for i in range(depths):
            if stage_spec[i] == 'L':
                self.attns.append(
                    LocalAttention(dim_embed, heads, window_size, attn_drop, proj_drop)
                )
            elif stage_spec[i] == 'D':
                self.attns.append(
                    DAttentionBaseline(
                        fmap_size, 
                        fmap_size, 
                        heads, 
                        hc, 
                        n_groups, 
                        attn_drop, 
                        proj_drop, 
                        stride,
                        offset_range_factor,
                        use_pe,
                        dwc_pe, 
                        no_off,
                        fixed_pe,
                        ksize,
                        log_cpb,
                        stage_i
                    )
                )
            elif stage_spec[i] == 'S':
                shift_size = math.ceil(window_size / 2)
                self.attns.append(
                    ShiftWindowAttention(dim_embed, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size)
                )
            elif stage_spec[i] == 'N':
                self.attns.append(
                    NeighborhoodAttention2D(dim_embed, nat_ksize, heads, attn_drop, proj_drop)
                )
            elif stage_spec[i] == 'A':
                self.attns.append(
                    LDABaseline(fmap_size, local_kv_size, heads, hc, n_groups, use_pe, no_off, local_orf)
                )
            elif stage_spec[i] == 'P':
                self.attns.append(
                    PyramidAttention(dim_embed, heads, attn_drop, proj_drop, sr_ratio)
                )
            elif self.stage_spec[i] == 'X':
                self.attns.append(
                    nn.Conv2d(dim_embed, dim_embed, kernel_size=window_size, padding=window_size // 2, groups=dim_embed)
                )
            elif self.stage_spec[i] == 'E':
                self.attns.append(
                    SlideAttention(dim_embed, heads, 3)
                )
            else:
                raise NotImplementedError(f'Spec: {stage_spec[i]} is not supported.')
            
            self.drop_path.append(DropPath(drop_path_rate[i]) if drop_path_rate[i] > 0.0 else nn.Identity())
    
    @auto_fp16(apply_to=('x', ))
    def _inner_forward(self, x):

        # assert x.dtype == torch.float16, f"AMP failed!, dtype={x.dtype}"
        x = self.proj(x)

        # positions = []
        # references = []
        for d in range(self.depths):

            if self.use_lpu:
                x0 = x
                x = self.local_perception_units[d](x.contiguous())
                x = x + x0
            
            if self.stage_spec[d] == 'X':
                x0 = x
                x = self.attns[d](x.contiguous())
                x = self.mlps[d](self.ln_cnvnxt[str(d)](x))
                x = self.drop_path[d](x) + x0
            else:
                x0 = x
                x, pos, ref = self.attns[d](self.layer_norms[2 * d](x))
                x = self.layer_scales[2 * d](x)
                x = self.drop_path[d](x) + x0
                x0 = x
                x = self.mlps[d](self.layer_norms[2 * d + 1](x))
                x = self.layer_scales[2 * d + 1](x)
                x = self.drop_path[d](x) + x0
            # positions.append(pos)
            # references.append(ref)

        # return x, positions, references
        # assert x.dtype == torch.float16, f"AMP failed!, dtype={x.dtype}"
        return x

    @auto_fp16(apply_to=('x', ))
    def forward(self, x):
        if self.training and x.requires_grad and self.use_checkpoint:
            return cp.checkpoint(self._inner_forward, x)
        else:
            return self._inner_forward(x)


'''
cfg.MODEL.BACKBONE.LAYERS = [0,1,2,3]
        cfg.MODEL.BACKBONE.FEATURE_DIMS = [128, 256, 512, 1024]
        cfg.MODEL.BACKBONE.CLS_DIMS = [256, 512, 1024, 2048]#!!!!SEARCH
        cfg.MODEL.BACKBONE_SMALL.LAYERS = [1,3]
        cfg.MODEL.BACKBONE_SMALL.CLS_DIMS = [256, 1024]
        cfg.MODEL.BACKBONE.SKIP_CONNECTION = False
'''

class DAT(nn.Module):

    def __init__(self, img_size=224, patch_size=4, num_classes=1000, expansion=4,
                 dim_stem=128, dims=[128, 256, 512, 1024], depths=[2, 4, 18, 2], 
                 heads=[4, 8, 16, 32], heads_q=[6, 12, 24, 48],
                 window_sizes=[7, 7, 7, 7],
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.8, 
                 strides=[8, 4, 2, 1],
                 offset_range_factor=[-1, -1, -1, -1],
                 local_orf=[-1, -1, -1, -1],
                 local_kv_sizes=[-1, -1, -1, -1],
                 offset_pes=[False, False, False, False],
                 stage_spec=[['N', 'D'], ['N', 'D', 'N', 'D'], ['N', 'D', 'N', 'D', 'N', 'D', 'N', 'D', 'N', 'D', 'N', 'D', 'N', 'D', 'N', 'D', 'N', 'D'], ['D', 'D']], 
                 groups=[2, 4, 8, 16],
                 use_pes=[True, True, True, True], 
                 dwc_pes=[False, False, False, False], 
                 sr_ratios=[8, 4, 2, 1], 
                 lower_lr_kvs={},
                 fixed_pes=[False, False, False, False],
                 no_offs=[False, False, False, False],
                 ns_per_pts=[4, 4, 4, 4],
                 use_dwc_mlps=[True, True, True, True],
                 use_conv_patches=True,
                 ksizes=[9, 7, 5, 3],
                 ksize_qnas=[3, 3, 3, 3],
                 nqs=[2, 2, 2, 2],
                 qna_activation='exp',
                 deform_groups=[0, 0, 0, 0],
                 nat_ksizes=[7, 7, 7, 7],
                 layer_scale_values=[-1,-1,-1,-1],
                 use_lpus=[True, True, True, True],
                 use_cmt_mlps=[False, False, False, False],
                 log_cpb=[False, False, False, False],
                 out_indices=(0, 1, 2, 3),
                 use_checkpoint=True,
                 init_cfg=dict(type='Pretrained', checkpoint='/home/admin2/Algonaut/dat_ckpts/dat.pth'),#'/home/admin2/Algonaut/dat_models/cmrcn_dat_b_3x.pth'),
                 **kwargs):
        super().__init__()
        
        self.fp16_enabled = False
        self.out_indices = out_indices
        
        self.log_cpb = log_cpb[0]
        self.dwc_pe = dwc_pes[0]
        self.slide = stage_spec[0][0] == "E"

        self.patch_proj = nn.Sequential(
            nn.Conv2d(3, dim_stem // 2, 3, patch_size // 2, 1),
            LayerNormProxy(dim_stem // 2),
            nn.GELU(),
            nn.Conv2d(dim_stem // 2, dim_stem, 3, patch_size // 2, 1),
            LayerNormProxy(dim_stem)
        ) if use_conv_patches else nn.Sequential(
            nn.Conv2d(3, dim_stem, patch_size, patch_size, 0),
            LayerNormProxy(dim_stem)
        )

        img_size = img_size // patch_size
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stages = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(4):
            dim1 = dim_stem if i == 0 else dims[i - 1] * 2
            dim2 = dims[i]

            self.stages.append(
                TransformerStage(
                    fmap_size=img_size,
                    window_size=window_sizes[i],
                    ns_per_pt=ns_per_pts[i],
                    dim_in=dim1,
                    dim_embed=dim2,
                    depths=depths[i],
                    stage_spec=stage_spec[i],
                    n_groups=groups[i],
                    use_pe=use_pes[i],
                    sr_ratio=sr_ratios[i],
                    heads=heads[i],
                    heads_q=heads_q[i],
                    stride=strides[i],
                    offset_range_factor=offset_range_factor[i],
                    local_orf=local_orf[i],
                    local_kv_size=local_kv_sizes[i],
                    dwc_pe=dwc_pes[i],
                    no_off=no_offs[i],
                    fixed_pe=fixed_pes[i],
                    attn_drop=attn_drop_rate,
                    proj_drop=drop_rate,
                    expansion=expansion,
                    drop=drop_rate,
                    drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                    use_dwc_mlp=use_dwc_mlps[i],
                    ksize=ksizes[i],
                    nat_ksize=nat_ksizes[i],
                    k_qna=ksize_qnas[i],
                    nq_qna=nqs[i],
                    qna_activation=qna_activation,
                    deform_groups=deform_groups[i],
                    layer_scale_value=layer_scale_values[i],
                    use_lpu=use_lpus[i],
                    use_cmt_mlp=use_cmt_mlps[i],
                    log_cpb=log_cpb[i],
                    stage_i=i,
                    use_checkpoint=use_checkpoint
                )
            )
            if i in self.out_indices:
                self.norms.append(
                    LayerNormProxy(dim2)
                )
            else:
                self.norms.append(nn.Identity())
            img_size = img_size // 2

        self.down_projs = nn.ModuleList()
        for i in range(3):
            self.down_projs.append(
                nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1, bias=False),
                    LayerNormProxy(dims[i + 1])
                ) if use_conv_patches else nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 2, 2, 0, bias=False),
                    LayerNormProxy(dims[i + 1])
                )
            )

        self.lower_lr_kvs = lower_lr_kvs
        self.init_cfg = init_cfg
        self.reset_parameters()
        

    def reset_parameters(self):

        for m in self.parameters():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    @auto_fp16(apply_to=('x', ))
    def get_tokens(self, x):
        x = self.patch_proj(x)
        
        local_tokens, global_tokens = {}, {}
        for i in range(4):
            
            x = self.stages[i](x)
            
            y = self.norms[i](x)
            if i < 3:
                x = self.down_projs[i](x)
            x_save = y.contiguous().clone()
            local_tokens[f"{i}"] = x_save
            global_tokens[f"{i}"] = x_save.mean(dim=(2, 3))
            local_tokens[f"{i}"] = x_save
            global_tokens[f"{i}"] = x_save.mean(dim=(2, 3))
        return local_tokens, global_tokens
