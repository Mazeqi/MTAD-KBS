import torch
import torch.nn as nn
import sys
sys.path.append("..")
sys.path.append(".")
from timm.models.registry import register_model
import math
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
from timm.models._builder import resolve_pretrained_cfg
try:
    from timm.models._builder import _update_default_kwargs as update_args
except:
    from timm.models._builder import _update_default_model_kwargs as update_args
from timm.models.vision_transformer import Mlp, PatchEmbed
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
import torch.nn.functional as F
# from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
from pathlib import Path
from timm.models.resnet import Bottleneck
import numpy as np
from hilbert import decode, encode
from pyzorder import ZOrderIndexer
from model import get_model
from model import MODEL
from util.utils_zigzag import reverse_permut_np, zigzag_path, hilbert_path
try:
    from dis_mamba.mamba_ssm.ops.selective_scan_interface import (
        selective_scan_fn,
        mamba_inner_fn,
        bimamba_inner_fn,
        mamba_inner_fn_no_out_proj,
    )
except ImportError:
    selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj = (
        None,
        None,
        None,
        None,
        None,
    )
from dis_mamba.mamba_ssm.modules.mamba_simple import Mamba

def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size: window size
        h_w: Height of window
        w_w: Width of window
    Returns:
        local window features (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B,windows.shape[2], H, W)
    return x

# @torch.compile
def forward_permutation(xz_main, _perm):
    return xz_main[:, :, _perm].contiguous()  # [B,C,T]


# @torch.compile
def backward_permutation(o_main, _perm_rev):
    return o_main[:, _perm_rev, :].contiguous()  # out is [B,T,C]

def _load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata
    
    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    
    if len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def _load_checkpoint(model,
                    filename,
                    map_location='cpu',
                    strict=False,
                    logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = torch.load(filename, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    if sorted(list(state_dict.keys()))[0].startswith('encoder'):
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}

    _load_state_dict(model, state_dict, strict, logger)
    return checkpoint


class Downsample(nn.Module):
    """
    Down-sampling block"
    """

    def __init__(self,
                 dim,
                 keep_dim=False,
                 ):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.reduction(x)
        return x

class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim*2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        x = self.expand(x)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x
    
class Upsample(nn.Module):
	"""
	Down-sampling block"
	"""

	def __init__(self,
					dim,
					keep_dim=False,
					):
		"""
		Args:
			dim: feature size dimension.
			norm_layer: normalization layer.
			keep_dim: bool argument for maintaining the resolution.
		"""

		super().__init__()
		if keep_dim:
			dim_in = dim
		else:
			dim_in = dim * 2

		self.upsample = nn.Sequential(
			nn.ConvTranspose2d(dim_in, dim, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(dim),
			nn.SiLU(),
		)

	def forward(self, x):
		x = self.upsample(x)
		return x

class PatchEmbed(nn.Module):
    """
    Patch embedding block"
    """

    def __init__(self, in_chans=3, in_dim=64, dim=96):
        """
        Args:
            in_chans: number of input channels.
            dim: feature size dimension.
        """
        # in_dim = 1
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU()
            )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x


class ConvBlock(nn.Module):

	def __init__(self, dim,
					drop_path=0.,
					layer_scale=None,
					kernel_size=3):
		super().__init__()

		self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
		self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
		self.act1 = nn.SiLU()

		self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
		self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
		self.layer_scale = layer_scale
		if layer_scale is not None and type(layer_scale) in [int, float]:
			self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
			self.layer_scale = True
		else:
			self.layer_scale = False
		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

	def forward(self, x):
		input = x
		x = self.conv1(x)
		x = self.norm1(x)
		x = self.act1(x)
		x = self.conv2(x)
		x = self.norm2(x)
		if self.layer_scale:
			x = x * self.gamma.view(1, -1, 1, 1)
		x = input + self.drop_path(x)
		return x
     
class MambaVisionMixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
        scan_type = "hilbertN8",
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner//2, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        self.scan_type = scan_type
        self.use_jit = False

        self.zigzag_paths = kwargs.get("zigzag_paths", None)
        self.zigzag_paths_reverse = kwargs.get("zigzag_paths_reverse", None)
        self.video_frames = kwargs.get("video_frames", None)
        self.st_order = kwargs.get("st_order", None)
        self.extras = kwargs.get("extras", None)
        self.use_jit = kwargs.get("use_jit", False)

        assert (
            (scan_type in ["v1", "v2"])
            or scan_type.startswith("video_")
            or scan_type.startswith("zigzagN")
            or scan_type.startswith("hilbertN")
            or scan_type.startswith("randomN")
            or scan_type.startswith("parallelN")
        ), f"Invalid scan_type: {scan_type}"

        if scan_type.startswith("parallelN"):
            self.parallel_num = int(scan_type.replace("parallelN", ""))
            self.A_b_log_list = []
            self.conv1d_b_list = []
            self.x_proj_b_list = []
            self.dt_proj_b_list = []
            self.D_b_list = []

            for _pn in range(self.parallel_num):
                A_b = repeat(
                    torch.arange(
                        1, self.d_state + 1, dtype=torch.float32, device=device
                    ),
                    "n -> d n",
                    d=self.d_inner,
                ).contiguous()
                A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
                self.A_b_log = nn.Parameter(A_b_log)
                self.A_b_log._no_weight_decay = True
                self.A_b_log_list.append(self.A_b_log)

                self.conv1d_b = nn.Conv1d(
                    in_channels=self.d_inner,
                    out_channels=self.d_inner,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    groups=self.d_inner,
                    padding=d_conv - 1,
                    **factory_kwargs,
                )
                self.conv1d_b_list.append(self.conv1d_b)

                self.x_proj_b = nn.Linear(
                    self.d_inner,
                    self.dt_rank + self.d_state * 2,
                    bias=False,
                    **factory_kwargs,
                )
                self.x_proj_b_list.append(self.x_proj_b)
                self.dt_proj_b = nn.Linear(
                    self.dt_rank, self.d_inner, bias=True, **factory_kwargs
                )
                self.dt_proj_b_list.append(self.dt_proj_b)

                self.D_b = nn.Parameter(
                    torch.ones(self.d_inner, device=device)
                )  # Keep in fp32
                self.D_b._no_weight_decay = True
                self.D_b_list.append(self.D_b)

            self.A_b_log_list = nn.ParameterList(self.A_b_log_list)
            self.conv1d_b_list = nn.ModuleList(self.conv1d_b_list)
            self.x_proj_b_list = nn.ModuleList(self.x_proj_b_list)
            self.dt_proj_b_list = nn.ModuleList(self.dt_proj_b_list)
            self.D_b_list = nn.ParameterList(self.D_b_list)

        elif scan_type == "v2":
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True

            #############################
            self.conv1d_b_z = nn.Conv1d(
                in_channels=self.d_inner//2,
                out_channels=self.d_inner//2,
                bias=conv_bias//2,
                kernel_size=d_conv,
                groups=self.d_inner//2,
                **factory_kwargs,
            )

            self.conv1d_b_x = nn.Conv1d(
                in_channels=self.d_inner//2,
                out_channels=self.d_inner//2,
                bias=conv_bias//2,
                kernel_size=d_conv,
                groups=self.d_inner//2,
                **factory_kwargs,
            )

            self.x_proj_b = nn.Linear(
                self.d_inner,
                self.dt_rank + self.d_state * 2,
                bias=False,
                **factory_kwargs,
            )
            self.dt_proj_b = nn.Linear(
                self.dt_rank, self.d_inner, bias=True, **factory_kwargs
            )

            self.D_b = nn.Parameter(
                torch.ones(self.d_inner, device=device)
            )  # Keep in fp32
            self.D_b._no_weight_decay = True

    def forward_ssm(self, xz, seqlen):
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x, 
                            dt, 
                            A, 
                            B, 
                            C, 
                            self.D.float(), 
                            z=None, 
                            delta_bias=self.dt_proj.bias.float(), 
                            delta_softplus=True, 
                            return_last_state=None)

        y = y*z
        y = rearrange(y, "b d l -> b l d")
        return y

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")

        if (
            self.scan_type.startswith("zigzagN")
            or self.scan_type.startswith("hilbertN")
            or self.scan_type.startswith("randomN")
        ):
            #### rearrange
            _perm = self.zigzag_paths[self.layer_idx]
            _perm_rev = self.zigzag_paths_reverse[self.layer_idx]
            _ex = self.extras
            xz_extra, xz_main = xz[:, :, :_ex], xz[:, :, _ex:]
            if self.use_jit:
                xz_main = forward_permutation(xz_main, _perm)
            else:
                xz_main = xz_main[:, :, _perm].contiguous()  # [B,C,T]
            xz = torch.cat([xz_extra, xz_main], dim=2)

            out = self.forward_ssm(xz, seqlen)
            out = self.out_proj(out)

            #### rearrange back
            o_ext, o_main = out[:, :_ex, :], out[:, _ex:, :]
            if self.use_jit:
                o_main = backward_permutation(o_main, _perm_rev)
            else:
                o_main = o_main[:, _perm_rev, :].contiguous()  # out is [B,T,C]
            out = torch.cat([o_ext, o_main], dim=1)
            return out
        
        elif self.scan_type == "v2":

            y1 = self.forward_ssm(xz, seqlen)
            xz = xz.flip([-1])
            x, z = xz.chunk(2, dim=1)
            A = -torch.exp(self.A_log.float())
            x = F.silu(F.conv1d(input=x, weight=self.conv1d_b_x.weight, bias=self.conv1d_b_x.bias, padding='same', groups=self.d_inner//2))
            z = F.silu(F.conv1d(input=z, weight=self.conv1d_b_z.weight, bias=self.conv1d_b_z.bias, padding='same', groups=self.d_inner//2))
            x_dbl = self.x_proj_b(rearrange(x, "b d l -> (b l) d"))
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = rearrange(self.dt_proj_b(dt), "(b l) d -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            y = selective_scan_fn(x, 
                                dt, 
                                A, 
                                B, 
                                C, 
                                self.D_b.float(), 
                                z=None, 
                                delta_bias=self.dt_proj_b.bias.float(), 
                                delta_softplus=True, 
                                return_last_state=None)
            y = y*z
            # y = torch.cat([y, z], dim=1)
            y2 = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y1+y2.flip([-1]))
        else:
            y = self.forward_ssm(xz, seqlen)
            out = self.out_proj(y)
            return out
    

class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
             q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = nn.functional.elu(q) + 1.
        k = nn.functional.elu(k) + 1.

        kv = torch.einsum('...sd,...se->...de', k, v)
        z = 1.0 / torch.einsum('...sd,...d->...s', q, k.sum(dim=-2))
        x = torch.einsum('...de,...sd,...s->...se', kv, q, z)
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class AgentAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 agent_num=49, window=14, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.agent_num = agent_num
        self.window = window

        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3),
                             padding=1, groups=dim)
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window, 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window, agent_num))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b, n, c = x.shape
        h = int(n ** 0.5)
        w = int(n ** 0.5)
        num_heads = self.num_heads
        head_dim = c // num_heads
        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # q, k, v: b, n, c

        agent_tokens = self.pool(q.reshape(b, h, w, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        position_bias1 = nn.functional.interpolate(self.an_bias, size=(self.window, self.window), mode='bilinear')
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        agent_bias1 = nn.functional.interpolate(self.na_bias, size=(self.window, self.window), mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)
        v_ = v[:, :, :, :].transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x[:, :, :] = x[:, :, :] + self.dwc(v_).permute(0, 2, 3, 1).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
        
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class Block(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads, 
                 window_size,
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=False, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 act_layer=nn.SiLU, 
                 norm_layer=nn.LayerNorm, 
                 Mlp_block=Mlp,
                 layer_scale=None,
                 size=8,
                 mam_att_depth=1,
                 layer_idx = 1,
                 scan_type = "hilbertN8",
                 att_mode = "linear",
                 **block_kwargs,
                 ):
        super().__init__()
        self.mamba_att_blocks = nn.ModuleList([
			MambaAttBlock(dim=dim, num_heads = num_heads, mlp_ratio = mlp_ratio, 
                          qkv_bias = qkv_bias, qk_scale = qk_scale, drop = drop, attn_drop = attn_drop,
                          drop_path=drop_path, act_layer= act_layer, norm_layer = norm_layer, Mlp_block = Mlp_block,
                          layer_scale = layer_scale,
                          size=size,
                          scan_type = scan_type,
                          layer_idx = layer_idx, 
                          att_mode = att_mode,
                          window_size = window_size,
                          **block_kwargs)
			for i in range(mam_att_depth)])
        self.window_size = window_size
        hidden_dim = dim
        self.finalconv11 = nn.Conv2d(in_channels=hidden_dim * 2, out_channels=hidden_dim, kernel_size=1, stride=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, window_size**2, hidden_dim), requires_grad=False)
        self.apply(self._init_weights)
        

    def _init_weights(self, m):
        """
        initialization
        """
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(np.sqrt(self.pos_embed.shape[-2])))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x):
        conv_in = x
        _, _, H, W = x.shape
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = torch.nn.functional.pad(x, (0, pad_r, 0, pad_b))
            _, _, Hp, Wp = x.shape
        else:
            Hp, Wp = H, W
        x = window_partition(x, self.window_size)
        
        x = x + self.pos_embed
        for i_blk in self.mamba_att_blocks:
            x = i_blk(x)

        x = window_reverse(x, self.window_size, Hp, Wp)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H, :W].contiguous()

        output = torch.cat((x, conv_in), dim=1)
        output = self.finalconv11(output).contiguous()
        return output


class MambaAttBlock(nn.Module):
    def __init__(self, 
                    dim, 
                    num_heads, 
                    mlp_ratio=4., 
                    qkv_bias=False, 
                    qk_scale=False, 
                    drop=0., 
                    attn_drop=0.,
                    drop_path=0., 
                    act_layer=nn.SiLU, 
                    norm_layer=nn.LayerNorm, 
                    Mlp_block=Mlp,
                    layer_scale=None,
                    size=8,
                    layer_idx=0,
                    scan_type = "hilbertN8",
                    window_size=16,
                    att_mode = "linear",
                    **block_kwargs
                    ):
        super().__init__()
        self.norm_att_1 = norm_layer(dim)
        att_mode = att_mode
        if att_mode == "linear":
            self.attn = LinearAttention(dim, num_heads=num_heads, qkv_bias=True)
        elif att_mode == "agent":
            self.attn = AgentAttention(dim, num_heads=num_heads, qkv_bias=True, window=window_size)
        elif att_mode == "no_attn":
            self.attn = ""
        else:
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True)

        self.norm_att_2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_att = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_att_1 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
        self.gamma_att_2 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1


        self.norm_mixer_1 = norm_layer(dim)

        ssm_cfg = {}
        factory_kwargs = {"device": "cuda:0", "dtype":torch.float32}
        self.mixer = Mamba(
            dim,
            d_state = 8,
            d_conv = 3,
            expand = 1,
            layer_idx=layer_idx,
            scan_type=scan_type,
            **ssm_cfg,
            **block_kwargs,
            **factory_kwargs,
        )

        self.norm_mixer_2 = norm_layer(dim)
        self.mlp_mixer = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_mixer_1 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
        self.gamma_mixer_2 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x):
        x = x + self.drop_path(self.gamma_mixer_1 * self.mixer(self.norm_mixer_1(x)))  
        x = x + self.drop_path(self.gamma_mixer_2 * self.mlp_mixer(self.norm_mixer_2(x)))
        x = x + self.drop_path(self.gamma_att_1   * self.attn(self.norm_att_1(x)))
        x = x + self.drop_path(self.gamma_att_2   * self.mlp_att(self.norm_att_2(x)))

        return x


class MambaRecLayer(nn.Module):
    """
    "MambaRec layer"
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 conv=False,
                 upsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 size=8,
                 scan_type = "hilbertN8",
                 att_mode = "linear"
    ):
        """
        Args:
            dim: feature size dimension.
            depth: number of layers in each stage.
            window_size: window size in each stage.
            conv: bool argument for conv stage flag.
            downsample: bool argument for down-sampling.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
            transformer_blocks: list of transformer blocks.
        """

        super().__init__()
        self.conv = conv
        if conv:
            self.blocks = nn.ModuleList([ConvBlock(dim=dim,
                                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                                   layer_scale=layer_scale_conv)
                                                   for i in range(depth)])
        else:
            self.extras = 0
            block_kwargs = {}
            device = "cuda:0"
            scan_type = scan_type
            print("scan_type", scan_type)
            patch_side_len = size

            if (
                scan_type.startswith("zigzagN")
                or scan_type.startswith("hilbertN")
                or scan_type.startswith("randomN")
                or scan_type.startswith("parallelN")
            ):
                if scan_type.startswith("zigzagN") or scan_type.startswith("parallelN"):
                    _zz_paths = zigzag_path(N=patch_side_len)
                    if scan_type.startswith("zigzagN"):
                        zigzag_num = int(scan_type.replace("zigzagN", ""))
                        zz_paths = _zz_paths[:zigzag_num]
                        assert (
                            len(zz_paths) == zigzag_num
                        ), f"{len(zz_paths)} != {zigzag_num}"
                    elif scan_type.startswith("parallelN"):
                        zz_paths = _zz_paths[:8]

                    else:
                        raise ValueError("scan_type should be xx")
                elif scan_type.startswith("hilbertN"):
                    _zz_paths = hilbert_path(N=patch_side_len)
                    if scan_type.startswith("hilbertN"):
                        zigzag_num = int(scan_type.replace("hilbertN", ""))
                        zz_paths = _zz_paths[:zigzag_num]
                        assert (
                            len(zz_paths) == zigzag_num
                        ), f"{len(zz_paths)} != {zigzag_num}"
                    else:
                        raise ValueError("scan_type should be xx")
                elif scan_type.startswith("randomN"):
                    zigzag_num = int(scan_type.replace("randomN", ""))
                    zz_paths = []
                    for _ddd in range(zigzag_num):
                        _tmp = np.array([_ for _ in range(patch_side_len**2)])
                        np.random.shuffle(_tmp)
                        print(_tmp)
                        zz_paths.append(_tmp)

                else:
                    raise ValueError(f"scan_type {scan_type} doenst match")
                print("zigzag_num", len(zz_paths))
                #############
                zz_paths_rev = [reverse_permut_np(_) for _ in zz_paths]
                zz_paths = zz_paths * depth
                zz_paths_rev = zz_paths_rev * depth
                zz_paths = [torch.from_numpy(_).to(device) for _ in zz_paths]
                zz_paths_rev = [torch.from_numpy(_).to(device) for _ in zz_paths_rev]
                assert len(zz_paths) == len(
                    zz_paths_rev
                ), f"{len(zz_paths)} != {len(zz_paths_rev)}"
                block_kwargs["zigzag_paths"] = zz_paths
                block_kwargs["zigzag_paths_reverse"] = zz_paths_rev
                block_kwargs["extras"] = self.extras
                print("zigzag_paths length", len(zz_paths))
                # for iii, _ in enumerate(zz_paths):
                #     print(f"zigzag_paths {iii}", _[:20])
            elif scan_type == "v2":
                pass  # no zigzag
            else:
                raise ValueError("scan_type doesn't match")
            
            self.blocks = nn.ModuleList([Block(dim=dim,
                                                window_size = window_size,
                                                num_heads=num_heads,
                                                mlp_ratio=mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                qk_scale=qk_scale,
                                                drop=drop,
                                                attn_drop=attn_drop,
                                                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                                layer_scale=layer_scale,
                                                size=size,
                                                layer_idx = i,
                                                mam_att_depth=1,
                                                scan_type=scan_type,
                                                att_mode = att_mode,
                                                **block_kwargs)
                                                for i in range(depth)])
   
                 

        self.upsample = None if not upsample else Upsample(dim=dim)
        self.do_gt = False


    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)

        for _, blk in enumerate(self.blocks):
            x = blk(x)
            
        return x

class MambaRec(nn.Module):
    """
    MambaRec,
    """

    def __init__(self,
                 dims = [512, 256, 128, 64],
                 depths = [3, 4, 6, 3],
                 window_size=[16, 16, 16, 16],
                 mlp_ratio=4.0,
                 num_heads=[8, 8, 8, 8],
                 drop_path_rate=0.2,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 scan_type = "hilbertN8",
                 att_mode = "linear",
                 **kwargs):
        """
        Args:
            dim: feature size dimension.
            depths: number of layers in each stage.
            window_size: window size in each stage.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            drop_path_rate: drop path rate.
            in_chans: number of input channels.
            num_classes: number of classes.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
        """
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(len(depths)):
            conv = False
            level = MambaRecLayer(dim=int(dims[i]),
                                     depth=depths[i],
                                     num_heads=num_heads[i],
                                     window_size=window_size[i],
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     conv=conv,
                                     drop=drop_rate,
                                     attn_drop=attn_drop_rate,
                                     drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])],
                                     upsample = True if i !=0 else False,
                                     layer_scale=layer_scale,
                                     layer_scale_conv=layer_scale_conv,
                                     size=window_size[i],
                                     scan_type = scan_type,
                                     att_mode = att_mode
                                     )
            self.levels.append(level)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'rpb'}
        

    def forward_features(self, x):
        out_list = []
        for i, level in enumerate(self.levels):
            x = level(x)
            if i != 0:
                out_list.insert(0, x)
        return out_list

    def forward(self, x):
        x = self.forward_features(x)
        return x

    def _load_state_dict(self, 
                         pretrained, 
                         strict: bool = False):
        _load_checkpoint(self, 
                         pretrained, 
                         strict=strict)


# ========== Decoder ==========
def conv3x3(in_planes, out_planes, stride = 1, groups = 1, dilation = 1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride = 1) -> nn.Conv2d:
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def deconv2x2(in_planes, out_planes, stride = 1, groups = 1, dilation = 1):
	return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride, groups=groups, bias=False, dilation=dilation)


class FPN(nn.Module):
	def __init__(self, block, layers, width_per_group = 64, norm_layer = None, ):
		super(FPN, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		# block.expansion = 16
		self._norm_layer = norm_layer
		self.base_width = width_per_group
		self.inplanes = 64 * block.expansion
		self.dilation = 1
		self.bn_layer = self._make_layer(block, 128, layers, stride=2)
		self.conv1 = conv3x3(16 * block.expansion, 32 * block.expansion, 2)
		self.bn1 = norm_layer(32 * block.expansion)

		self.conv2 = conv3x3(32 * block.expansion, 64 * block.expansion, 2)
		self.bn2 = norm_layer(64 * block.expansion)

		self.conv21 = nn.Conv2d(32 * block.expansion, 32 * block.expansion, 1)
		self.bn21 = norm_layer(32 * block.expansion)

		self.conv31 = nn.Conv2d(64 * block.expansion, 64 * block.expansion, 1)
		self.bn31 = norm_layer(64 * block.expansion)

		self.convf = nn.Conv2d(64 * block.expansion, 64 * block.expansion, 1)
		self.bnf = norm_layer(64 * block.expansion)
		self.relu = nn.ReLU(inplace=True)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def _make_layer(self, block, planes, blocks, stride = 1, dilate = False):
		norm_layer = self._norm_layer
		downsample = None
		previous_dilation = self.dilation
		if dilate:
			self.dilation *= stride
			stride = 1
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride),
										norm_layer(planes * block.expansion), )
		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
		return nn.Sequential(*layers)

	def forward(self, x):
		fpn0 = self.relu(self.bn1(self.conv1(x[0])))
		fpn1 = self.relu(self.bn21(self.conv21(x[1]))) + fpn0
		sv_features = self.relu(self.bn2(self.conv2(fpn1))) + self.relu(self.bn31(self.conv31(x[2])))
		sv_features = self.relu(self.bnf(self.convf(sv_features)))
		sv_features = self.bn_layer(sv_features)
		return sv_features.contiguous()


class TMAD(nn.Module):
	def __init__(self, model_t, model_s):
		super(TMAD, self).__init__()
		self.net_t = get_model(model_t)
		self.fpn = FPN(Bottleneck, 3)
		self.net_s = MambaRec(dims=model_s["dims"], 
                                depths=model_s['depths'], 
                                scan_type = model_s["scan_type"],
                                num_heads = model_s["num_heads"], 
                                window_size = model_s["window_size"],
                                mlp_ratio=model_s["mlp_ratio"],
                                att_mode = model_s["att_mode"])
		self.frozen_layers = ['net_t']

	def freeze_layer(self, module):
		module.eval()
		for param in module.parameters():
			param.requires_grad = False

	def train(self, mode=True):
		self.training = mode
		for mname, module in self.named_children():
			if mname in self.frozen_layers:
				self.freeze_layer(module)
			else:
				module.train(mode)
		return self

	def forward(self, imgs):
		feats_t = self.net_t(imgs)
		feats_t = [f.detach() for f in feats_t]
		fpn_f = self.fpn(feats_t)
		feats_s = self.net_s(fpn_f)
		return feats_t, feats_s

@MODEL.register_module
def tmad(pretrained=False, **kwargs):
	model = TMAD(**kwargs)
	return model

if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
    from util.util import get_timepc, get_net_params
    vmunet = TMAD()
    bs = 1
    reso = 8
    x = torch.randn(bs, 512, reso, reso).cuda()
    net = vmunet.cuda()
    net.eval()
    y = net(x)
    Flops = FlopCountAnalysis(net, x)
    print(flop_count_table(Flops, max_depth=5))
    flops = Flops.total() / bs / 1e9
    params = parameter_count(net)[''] / 1e6
    with torch.no_grad():
        pre_cnt, cnt = 5, 10
        for _ in range(pre_cnt):
            y = net(x)
        t_s = get_timepc()
        for _ in range(cnt):
            y = net(x)
        t_e = get_timepc()
    print('[GFLOPs: {:>6.3f}G]\t[Params: {:>6.3f}M]\t[Speed: {:>7.3f}]\n'.format(flops, params, bs * cnt / (t_e - t_s)))