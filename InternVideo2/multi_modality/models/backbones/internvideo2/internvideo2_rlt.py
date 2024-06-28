import ipdb;
import math
import logging
import torch
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import nn

import torch.utils.checkpoint as checkpoint
from functools import partial
from einops import rearrange

from .pos_embed import get_3d_sincos_pos_embed, get_2d_sincos_pos_embed, get_1d_sincos_pos_embed, interpolate_pos_embed_internvideo2
from .flash_attention_class import FlashAttention
from .internvideo2 import (
    Block,
    Attention,
    Mlp,
    LayerScale,
    RMSNorm, 
    CrossAttention,
    AttentiveBlock,
    Linear_Decoder, 
    AttentionPoolingBlock
)

logger = logging.getLogger(__name__)

# Attempt to load flash attention custom ops. If unable to
# load, fall back to naive implementation.
try:
    from flash_attn.ops.fused_dense import FusedMLP
    use_flash_attn = True
except:
    logger.warn(f'FusedMLP of flash_attn is not installed!!!')
    use_flash_attn = False

try:
    from flash_attn.ops.rms_norm import DropoutAddRMSNorm
    use_flash_attn = True
except:
    logger.warn(f'DropoutAddRMSNorm of flash_attn is not installed!!!')
    use_flash_attn = False

class RLTPatchEmbed(nn.Module):
    """ 3D Image to Patch Embedding
    """

    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
            num_frames=8, tubelet_size=1, norm_layer=None
        ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.grid_size = (
            num_frames // tubelet_size,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1]
        ) # (T, H, W)
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.num_img_patches = self.grid_size[1] * self.grid_size[2]

        self.proj = nn.Conv3d(
            in_channels=in_chans, out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
            stride=(tubelet_size, patch_size[0], patch_size[1])
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def batched_find_idxs_to_keep(self, x: torch.Tensor, 
                              threshold: int= 0.05, 
                              tubelet_size: int=1,
                              patch_size: int=14) -> torch.Tensor:
        """
        Find the static tokens in a video tensor, and return a mask
        that selects tokens that are not repeated.

        Args:
        - x (torch.Tensor): A tensor of shape [B, C, T, H, W].
        - threshold (int): The mean intensity threshold for considering
                a token as static.
        - tubelet_size (int): The temporal length of a token.
        Returns:
        - mask (torch.Tensor): A bool tensor of shape [B, T, H, W] 
            that selects tokens that are not repeated.

        """
        # Ensure input has the format [B, C, T, H, W]
        assert len(x.shape) == 5, "Input must be a 5D tensor"
        #ipdb.set_trace()
        # Convert to float32 if not already
        x = x.type(torch.float32)
        
        # Calculate differences between frames with a step of tubelet_size, ensuring 
        # batch dimension is preserved.
        # Compare "front" of first token to "back" of second token (for tubelet_size > 1)
        diffs = x[:, :, (2*tubelet_size-1)::tubelet_size] - x[:, :, :-tubelet_size:tubelet_size]
        # Ensure we track negative movement (L1 distance)
        diffs = torch.abs(diffs)
        
        # Apply average pooling over spatial dimensions while keeping the batch dimension intact
        avg_pool_blocks = F.avg_pool3d(diffs, (1, patch_size, patch_size))
        # Compute the mean along the channel dimension, preserving the batch dimension
        avg_pool_blocks = torch.mean(avg_pool_blocks, dim=1, keepdim=True)
        # Create a dummy first frame for each item in the batch
        first_frame = torch.ones_like(avg_pool_blocks[:, :, 0:1]) * 255
        # Concatenate the dummy first frame with the rest of the frames, preserving the batch dimension
        avg_pool_blocks = torch.cat([first_frame, avg_pool_blocks], dim=2)
        # Determine indices to keep based on the threshold, ensuring the operation is applied across the batch
        
        keep_idxs = avg_pool_blocks.squeeze(1) > threshold  
        # Flatten out everything but the batch dimension
        keep_idxs = keep_idxs.flatten(1)
        #ipdb.set_trace()
        return keep_idxs

    @torch.no_grad()
    def run_rlt(x, pos_embed):
        return x, pos_embed

    def forward(self, x):
        ##################################
        #ipdb.set_trace()
        assert self.patch_size[0] == self.patch_size[1]
        # Rearrange into tiles for Conv3D
        keep_mask = self.batched_find_idxs_to_keep(
            x,
            threshold=0.05,
            tubelet_size=self.tubelet_size,
            patch_size=self.patch_size[0])
        # Rearrange into tiles for Conv3D
        #ipdb.set_trace()
        B = x.shape[0]
        x = rearrange(
            x,
            "b c (t p0) (h p1) (w p2) -> (b t h w) c p0 p1 p2",
            p0=self.tubelet_size,
            p1=self.patch_size[0],
            p2=self.patch_size[1],
        )
        #ipdb.set_trace()

        ##################################
        x = self.proj(x)
        x = x.flatten(start_dim=1)  # N x D x 1 x 1 x 1 => N x D
        x = self.norm(x)
        # Add a dimension. We need to remove this later ot handle 
        # variable seqlens
        x = x.unsqueeze(0)
        #B, T, L, C = x.shape  # T: temporal; L: spatial
        #x = x.view([B,  C])
        # Output is now (sum_{i=1^B} N_i),  C
        return x, keep_mask

class RLTPretrainInternVideo2(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,
            patch_size: int = 14,
            img_size: int = 224,
            qkv_bias: bool = False,
            drop_path_rate: float = 0.25,
            embed_dim: int = 1408,
            num_heads: int = 16,
            mlp_ratio: float = 48/11,
            init_values: float = 1e-5,
            qk_normalization: bool = True,
            depth: int = 40,
            use_flash_attn: bool = True,
            use_fused_rmsnorm: bool = True,
            use_fused_mlp: bool = True,
            fused_mlp_heuristic: int = 1,
            attn_pool_num_heads: int = 16,
            clip_embed_dim: int = 768,
            layerscale_no_force_fp32: bool = False,
            num_frames: int = 8,
            tubelet_size: int = 1,
            sep_pos_embed: bool = False,
            sep_image_video_pos_embed: bool = False,
            use_checkpoint: bool = False,
            checkpoint_num: int = 0,
            # for unmasked teacher
            clip_teacher_embed_dim: int = 3200,
            clip_teacher_final_dim: int = 768, # if 0, not distill final features
            clip_norm_type: str = 'l2',
            clip_return_layer: int = 1,
            clip_student_return_interval: int = 1,
        ):
        super().__init__()

        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        assert use_flash_attn == use_fused_rmsnorm == use_fused_mlp, 'use_flash_attn, use_fused_rmsnorm and use_fused_mlp should be consistent'

        self.use_flash_attn = use_flash_attn
        self.embed_dim = embed_dim

        self.depth = depth
        self.clip_norm_type = clip_norm_type
        self.return_index = []
        for i in range(clip_return_layer):
            self.return_index.append(depth - int(i * clip_student_return_interval) - 1)
        logger.info(f'Normalization Type: {clip_norm_type}')
        logger.info(f'Strudent Return Index: {self.return_index}')

        if use_fused_rmsnorm:
            norm_layer_for_blocks = partial(DropoutAddRMSNorm, eps=1e-6, prenorm=True)
        else:
            norm_layer_for_blocks = partial(RMSNorm, eps=1e-6)
        self.norm_layer_for_blocks = norm_layer_for_blocks
        self.patch_embed = RLTPatchEmbed(
            img_size, patch_size, in_chans, embed_dim,
            num_frames=num_frames, tubelet_size=tubelet_size,
        )
        num_patches = self.patch_embed.num_patches
        num_img_patches = self.patch_embed.num_img_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # stolen from https://github.com/facebookresearch/mae_st/blob/dc072aaaf640d06892e23a33b42223a994efe272/models_vit.py#L65-L73C17
        self.sep_pos_embed = sep_pos_embed
        self.sep_image_video_pos_embed = sep_image_video_pos_embed
        if sep_pos_embed:
            raise NotImplementedError
        else:
            if sep_image_video_pos_embed:
                logger.info("Use joint position embedding, for image and video we use different pos_embed.")
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
                self.img_pos_embed = nn.Parameter(torch.zeros(1, num_img_patches + 1, embed_dim))
                # for CLIP decoder
                self.clip_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
                self.clip_img_pos_embed = nn.Parameter(torch.zeros(1, num_img_patches + 1, embed_dim))
            else:
                logger.info("Use joint position embedding, for image and video we use same pos_embed.")
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
                self.clip_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # choose which layer to use checkpoint
        with_cp_list = [False] * depth
        if use_checkpoint:
            for idx in range(depth):
                if idx < checkpoint_num:
                    with_cp_list[idx] = True
        logger.info(f"Droppath rate: {dpr}")
        logger.info(f"Checkpoint list: {with_cp_list}")

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias,
                  norm_layer=norm_layer_for_blocks,
                  drop_path=dpr[i], init_values=init_values, attn_drop=0.,
                  use_flash_attn=use_flash_attn, use_fused_mlp=use_fused_mlp,
                  fused_mlp_heuristic=fused_mlp_heuristic,
                  with_cp=with_cp_list[i],
                  qk_normalization=qk_normalization,
                  layerscale_no_force_fp32=layerscale_no_force_fp32,
                  use_fused_rmsnorm=use_fused_rmsnorm)
            for i in range(depth)])
        self.clip_projector = AttentionPoolingBlock(
            dim=embed_dim, num_heads=attn_pool_num_heads, qkv_bias=True, qk_scale=None,
            drop=0., attn_drop=0., norm_layer=partial(nn.LayerNorm, eps=1e-5), out_dim=clip_embed_dim)

        # CLIP decoder
        self.clip_decoder = nn.ModuleList([
            Linear_Decoder(
                in_channels=embed_dim,
                out_channels=clip_teacher_embed_dim,
                norm_layer=partial(nn.LayerNorm, eps=1e-5),
                clip_norm_type=clip_norm_type
            ) for _ in range(clip_return_layer)
        ])
        self.final_clip_decoder = nn.Identity()
        if clip_teacher_final_dim > 0:
            self.final_clip_decoder = Linear_Decoder(
                in_channels=clip_embed_dim,
                out_channels=clip_teacher_final_dim,
                norm_layer=partial(nn.LayerNorm, eps=1e-5),
                clip_norm_type=clip_norm_type
            )

        self.init_pos_embed()
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def init_pos_embed(self):
        logger.info("Init pos_embed from sincos pos_embed")
        if self.sep_pos_embed:
            raise NotImplementedError
        else:
            # trunc_normal_(self.pos_embed, std=.02)
            # trunc_normal_(self.clip_pos_embed, std=.02)
            pos_embed = get_3d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                self.patch_embed.grid_size[1], # height & weight
                self.patch_embed.grid_size[0], # t_size
                cls_token=True
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
            self.clip_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            if self.sep_image_video_pos_embed:
                img_pos_embed = get_3d_sincos_pos_embed(
                    self.pos_embed.shape[-1],
                    self.patch_embed.grid_size[1], # height & weight
                    1,
                    cls_token=True
                )
                self.img_pos_embed.data.copy_(torch.from_numpy(img_pos_embed).float().unsqueeze(0))
                self.clip_img_pos_embed.data.copy_(torch.from_numpy(img_pos_embed).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    @property
    def dtype(self):
        return self.patch_embed.proj.weight.dtype

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'pos_embed',
            'pos_embed_spatial',
            'pos_embed_temporal',
            'pos_embed_cls',
            'img_pos_embed',
            'cls_token',
            'clip_pos_embed',
            'clip_pos_embed_spatial',
            'clip_pos_embed_temporal',
            'clip_pos_embed_cls',
            'clip_img_pos_embed'
        }

    # @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x, mask=None, use_image=False, x_vis_return_idx=-1, x_vis_only=False):
        keep_mask = None
        x, keep_mask = self.patch_embed(x.type(self.dtype))
        #ipdb.set_trace()
        # print(f"x.shape: {x.shape} x.dtype: {x.dtype}, model.dtype: {self.dtype}")
        B, C = x.shape[0], x.shape[-1]

        # append cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # TEMPORARY: REMOVE LATER
        x = x[keep_mask].unsqueeze(0)
        x = torch.cat((cls_tokens, x), dim=1)

        # add pos_embed
        if self.sep_pos_embed:
            raise NotImplementedError
        else:
            if use_image:
                if self.sep_image_video_pos_embed:
                    pos_embed = self.img_pos_embed
                else:
                    # (1, num_img_patches + 1, embed_dim)
                    # print('origin pos_embed.shape:', self.pos_embed.shape)
                    cls_pos_embed = self.pos_embed[:, 0:1, :]
                    # print('cls_pos_embed.shape:', cls_pos_embed.shape)

                    img_pos_embed = self.pos_embed[:, 1:, :].view(1, self.num_frames, self.patch_embed.num_patches // self.num_frames, self.embed_dim).mean(dim=1)
                    if keep_mask is not None:
                        img_pos_embed = img_pos_embed[keep_mask]
                    # print('img_pos_embed.shape:', img_pos_embed.shape)
                    
                    pos_embed = torch.cat([cls_pos_embed, img_pos_embed], dim=1)
                    # print('final img_pos_embed.shape:', pos_embed.shape)
            else:
                pos_embed = self.pos_embed
        #ipdb.set_trace()
        # Select the RLT tokens, but keep CLS token.
        selected_embed = pos_embed[:, 1:][keep_mask].unsqueeze(0)
        cls_pos_embed = pos_embed[:, 0:1, :]
        pos_embed = torch.cat([cls_pos_embed, selected_embed], dim=1)
        x = x + pos_embed

        # mask tokens, ~mask means visible
        if mask is not None:
            x = x[~mask].reshape(B, -1, C)
        else:
            x = x.reshape(B, -1, C)

        residual = None
        x_clip = []
        for idx, blk in enumerate(self.blocks):
            if isinstance(x, tuple) and len(x) == 2:
                x, residual = x
            # print(f"\033[31m这是{idx}, {x.shape}\033[0m")
            x = blk(x, residual=residual)
            # return intermediate features
            if idx in self.return_index:
                if isinstance(x, tuple) and len(x) == 2:
                    tmp_x, tmp_residual = x
                    if residual is not None:
                        x_clip.append(tmp_x + tmp_residual)
                else:
                    x_clip.append(x)
            if idx == (self.depth + x_vis_return_idx):
                # print(f'idx = {idx} len(self.blocks)={len(self.blocks)}')
                break

        if isinstance(x, tuple) and len(x) == 2:
            x, residual = x
            if residual is not None:
                x = x + residual

        x_vis = x
        if x_vis_only:
            return x_vis

        x_pool_vis = self.clip_projector(x_vis)
        x_align = self.final_clip_decoder(x_pool_vis)

        # align CLIP
        x_clip = torch.stack(x_clip)
        K, B, _, C_CLIP = x_clip.shape
        # add pos_embed
        if self.sep_pos_embed:
            raise NotImplementedError
        else:
            if use_image:
                if self.sep_image_video_pos_embed:
                    clip_pos_embed = self.clip_img_pos_embed
                else:
                    # (1, num_img_patches + 1, embed_dim)
                    # print('origin pos_embed.shape:', self.pos_embed.shape)
                    clip_cls_pos_embed = self.clip_pos_embed[:, 0:1, :]
                    # print('cls_pos_embed.shape:', cls_pos_embed.shape)

                    clip_img_pos_embed = self.clip_pos_embed[:, 1:, :].view(1, self.num_frames, self.patch_embed.num_patches // self.num_frames, self.embed_dim).mean(dim=1)
                    # print('img_pos_embed.shape:', img_pos_embed.shape)

                    clip_pos_embed = torch.cat([clip_cls_pos_embed, clip_img_pos_embed], dim=1)
                    # print('final img_pos_embed.shape:', pos_embed.shape)

            else:
                clip_pos_embed = self.clip_pos_embed

        clip_pos_embed = clip_pos_embed.repeat(B, 1, 1)
        if mask is not None:
            x_clip = x_clip + clip_pos_embed[~mask].view(B, -1, C_CLIP).unsqueeze(0).repeat(K, 1, 1, 1)
        else:
            x_clip = x_clip + clip_pos_embed.view(B, -1, C_CLIP).unsqueeze(0).repeat(K, 1, 1, 1)

        # CLIP decoder
        x_clip_align = []
        for idx, clip_decoder in enumerate(self.clip_decoder):
            x_clip_align.append(clip_decoder(x_clip[idx]))
        x_clip_align = torch.stack(x_clip_align)


        return x_vis, x_pool_vis, x_clip_align, x_align

def rlt_pretrain_internvideo2_1b_patch14_224(config):
    model = RLTPretrainInternVideo2(
        in_chans=3, img_size=224, patch_size=14,
        embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11,
        clip_embed_dim=config.vision_encoder.clip_embed_dim,
        attn_pool_num_heads=16, qkv_bias=False,
        drop_path_rate=0.25,
        init_values=0.00001,
        qk_normalization=True,
        use_flash_attn=(config.vision_encoder.get('use_flash_attn', True) and use_flash_attn),
        use_fused_rmsnorm=(config.vision_encoder.get('use_fused_rmsnorm', True) and use_flash_attn),
        use_fused_mlp=(config.vision_encoder.get('use_fused_mlp', True) and use_flash_attn),
        fused_mlp_heuristic=1,
        layerscale_no_force_fp32=False,
        num_frames=config.vision_encoder.num_frames,
        tubelet_size=config.vision_encoder.tubelet_size,
        sep_pos_embed=False,
        sep_image_video_pos_embed=config.vision_encoder.sep_image_video_pos_embed,
        use_checkpoint=config.vision_encoder.use_checkpoint,
        checkpoint_num=config.vision_encoder.checkpoint_num,
        clip_teacher_embed_dim=config.vision_encoder.clip_teacher_embed_dim,
        clip_teacher_final_dim=config.vision_encoder.clip_teacher_final_dim,
        clip_norm_type=config.vision_encoder.clip_norm_type,
        clip_return_layer=config.vision_encoder.clip_return_layer,
        clip_student_return_interval=config.vision_encoder.clip_student_return_interval,
    )

    if config.vision_encoder.pretrained is not None:
        logger.info(f"Loading pretrained weights from {config.vision_encoder.pretrained}")
        state_dict = torch.load(config.vision_encoder.pretrained, map_location='cpu')
        interpolate_pos_embed_internvideo2(state_dict, model, orig_t_size=8)
        message = model.load_state_dict(state_dict, strict=False)
        logger.info(message)
    else:
        logger.info("No pretrained weights!!!")
    return model
