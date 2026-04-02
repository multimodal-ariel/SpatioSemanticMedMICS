# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Optional, Tuple, Type
# from .transformer import TwoWayTransformer
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor, nn

import math
from typing import Tuple, Type


class MLPBlock3D(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class TwoWayTransformer3D(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock3D(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, x, y, z = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock3D(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock3D(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out



class LayerNorm3d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class MaskDecoder3D_MLAM(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        num_multimask_outputs: int = 3,
        encoder_channels: Tuple[int,int,int,int],
        original_image_channels: int = 1,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ):
        """
        encoder_channels: number of feature‐map channels at each ViT stage,
          highest resolution last, e.g. [(384,16³),(384,32³),(384,64³),(384,128³)]→we only need channels
        original_image_channels: e.g. 1 for CT volumes
        """
        super().__init__()
        D0, D1, D2, D3 = encoder_channels  # e.g. (384,384,384,384)
        
        # 1) Transformer & tokens unchanged
        self.num_multimask_outputs = num_multimask_outputs
        self.transformer = TwoWayTransformer3D(depth=2,
                                               embedding_dim=transformer_dim,
                                               num_heads=8, mlp_dim=2048)
        self.iou_token   = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1  # +1 for the iou token
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)  # num_multimask=3 + 1

        # 2) Progressive upsampling from 8³→16³→32³→64³→128³
        # define the channel sizes at each stage
        up_chs = [
            transformer_dim // 2,       # 192 @16³
            transformer_dim // 4,       #  96 @32³
            transformer_dim // 8,       #  48 @64³
            transformer_dim // 16,      #  24 @128³
        ]
        in_chs = [transformer_dim] + up_chs[:-1]

        # transpose‐conv layers
        self.up_transposes = nn.ModuleList([
            nn.ConvTranspose3d(in_chs[i], up_chs[i], kernel_size=2, stride=2)
            for i in range(4)
        ])
        # fusion convs to merge with encoder features at each resolution
        self.fuse_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(up_chs[i] + enc_ch, up_chs[i], kernel_size=3, padding=1),
                LayerNorm3d(up_chs[i]),
                activation()
            )
            for i, enc_ch in enumerate([D0, D1, D2, D3])
        ])

        # 3) Final image‐concatenation fusion
        self.final_fuse = nn.Sequential(
            nn.Conv3d(up_chs[-1] + original_image_channels, up_chs[-1], kernel_size=3, padding=1),
            activation(),
            nn.Conv3d(up_chs[-1], 1, kernel_size=1)  # single‐channel mask
        )

        # 4) (Optional) IOU head unchanged
        self.iou_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,     # [B,384,8,8,8]
        image_pe: torch.Tensor,             # [1,384,8,8,8]
        dense_prompt_embeddings: torch.Tensor,  # [B,384,8,8,8]
        encoder_feats: Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor],  
            # from ViT: [(B,D0,16,16,16), (B,D1,32,32,32), (B,D2,64,64,64), (B,D3,128,128,128)]
        original_image: torch.Tensor,       # [B,1,128,128,128]
        multimask_output: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # 1) Build tokens
        B = image_embeddings.shape[0]
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        tokens = output_tokens.unsqueeze(0).expand(B, -1, -1)
        # 2) Run transformer
        # similar expansion as original code
        src = image_embeddings + dense_prompt_embeddings
        pos_src = image_pe
        hs, src = self.transformer(src, pos_src, tokens)
        iou_out   = hs[:, 0, :]
        mask_tok  = hs[:, 1 : 1+self.num_mask_tokens, :]

        # 3) Reshape for upsampling
        # src: [B, num_patches, C]; reshape back to [B, C, D, H, W]
        b, num_patches, c = src.shape
        # spatial dims from the image_embeddings tensor
        _, _, D, H, W = image_embeddings.shape
        x = src.transpose(1, 2).view(b, c, D, H, W)

        # 4) Progressive up‐sampling + MLAM fusions
        for i in range(4):
            # upsample
            x = self.up_transposes[i](x)
            # fuse with corresponding encoder feature
            feat = encoder_feats[i]                          # e.g. [B, D_i, size_i³]
            x = torch.cat([x, feat], dim=1)                  # channel concat
            x = self.fuse_convs[i](x)                       # reduce+norm+act

        # 5) Final fusion with original image
        x = torch.cat([x, original_image], dim=1)            # [B, 24+1,128³]
        masks = self.final_fuse(x)                           # [B,1,128,128,128]

        # 6) IOU prediction
        iou_pred = self.iou_head(iou_out)

        # 7) select single/multi mask branch
        if not multimask_output:
            masks = masks  # here masks is already one mask; for multi, you could replicate

        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

