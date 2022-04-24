"""
    Help functions
"""

import os
from typing import Optional
import functools
import torch
from torch import nn
from torch.nn import init
import matplotlib.pyplot as plt


def data_reg(images):
    """Regularization"""
    images_mean = images.mean()
    images_std = images.std()
    images = (images - images_mean) / images_std
    images_min = images.min()
    images = images - images_min
    return images


def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialization methods provided by CycleGAN."""

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    f"initialization method {init_type} is not implemented"
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print(f"initialize network with {init_type}")
    net.apply(init_func)


def get_embedding_function(
    num_encoding_functions=6, include_input=True, log_sampling=True
):
    r"""Returns a lambda function that internally calls positional_encoding."""
    return lambda x: positional_encoding(
        x, num_encoding_functions, include_input, log_sampling
    )


def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True, log_sampling=True
):
    r"""Apply positional encoding to the input.

    Args:
      tensor (torch.Tensor): Input tensor to be positionally encoded.
      num_encoding_functions (optional, int): Number of encoding functions used to
          compute a positional encoding (default: 6).
      include_input (optional, bool): Whether or not to include the input in the
          computed positional encoding (default: True).
      log_sampling (optional, bool): Sample logarithmically in frequency space, as
          opposed to linearly (default: True).

    Returns:
      (torch.Tensor): Positional encoding of the input tensor.
    """
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    # Now, encode the input using a set of high-frequency functions and append the
    # resulting values to the encoding.
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)


def get_minibatches(inputs: torch.Tensor, chunksize: Optional[int] = 1024 * 8):
    r"""Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
    Each element of the list (except possibly the last) has dimension `0` of length
    `chunksize`.
    """
    return [inputs[:, i : i + chunksize] for i in range(0, inputs.shape[1], chunksize)]


def meshgrid_xy(
    tensor1: torch.Tensor, tensor2: torch.Tensor
) -> (torch.Tensor, torch.Tensor):
    """Mimick np.meshgrid(..., indexing="xy") in pytorch. torch.meshgrid only allows "ij" indexing.
    Args:
      tensor1 (torch.Tensor): Tensor whose elements define the first dimension of the returned meshgrid.
      tensor2 (torch.Tensor): Tensor whose elements define the second dimension of the returned meshgrid.
    """
    ii, jj = torch.meshgrid(tensor1, tensor2)
    return ii.transpose(-1, -2), jj.transpose(-1, -2)


def get_ray_bundle(height, width, tform_cam2world):
    # Generate camera rays
    ii, jj = meshgrid_xy(
        torch.arange(width).to(tform_cam2world),
        torch.arange(height).to(tform_cam2world),
    )
    # return B,H,W,3
    scale_factor = height / width
    grid = torch.stack(
        [
            (ii - width * 0.5) / width,
            -((jj - height * 0.5) / height) * scale_factor,
            torch.zeros_like(ii),
        ],
        dim=0,
    )
    grid = grid.reshape([3, -1])
    ray_directions = torch.matmul(
        tform_cam2world[:, :3, :3], torch.tensor([0, 0, -1]).to(tform_cam2world)
    )
    ray_origins = (
        torch.matmul(tform_cam2world[:, :3, :3], grid) + tform_cam2world[:, :3, 3:]
    )
    ray_origins = ray_origins.reshape(*ray_origins.shape[:2], width, height).permute(
        0, 2, 3, 1
    )
    ray_directions = ray_directions[:, None, None, :].expand(ray_origins.shape)
    return ray_origins, ray_directions


def repeat_interleave(data, repeats):
    """
    Repeat interleave along axis 0
    torch.repeat_interleave is currently very slow
    https://github.com/pytorch/pytorch/issues/31980
    """
    output = data.unsqueeze(1).expand(-1, repeats, *data.shape[1:])
    return output.reshape(-1, *data.shape[1:])


def combine_interleaved(t, inner_dims=(1,), agg_type="average"):
    if len(inner_dims) == 1 and inner_dims[0] == 1:
        return t
    t = t.reshape(-1, *inner_dims, *t.shape[1:])
    if agg_type == "average":
        t = torch.mean(t, dim=1)
    elif agg_type == "max":
        t = torch.max(t, dim=1)[0]
    else:
        raise NotImplementedError("Unsupported combine type " + agg_type)
    return t


def get_norm_layer(norm_type="instance", group_norm_groups=32):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True
        )
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif norm_type == "group":
        norm_layer = functools.partial(nn.GroupNorm, group_norm_groups)
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError(f"Normalization layer {norm_type} is not found.")
    return norm_layer


def gather_cdf_util(cdf, inds):
    # A very contrived way of mimicking a version of the tf.gather()
    orig_inds_shape = inds.shape
    inds_flat = [inds[i].view(-1) for i in range(inds.shape[0])]
    valid_mask = [
        torch.where(ind >= cdf.shape[1], torch.zeros_like(ind), torch.ones_like(ind))
        for ind in inds_flat
    ]
    inds_flat = [
        torch.where(ind >= cdf.shape[1], (cdf.shape[1] - 1) * torch.ones_like(ind), ind)
        for ind in inds_flat
    ]
    cdf_flat = [cdf[i][ind] for i, ind in enumerate(inds_flat)]
    cdf_flat = [cdf_flat[i] * valid_mask[i] for i in range(len(cdf_flat))]
    cdf_flat = [
        cdf_chunk.reshape([1] + list(orig_inds_shape[1:])) for cdf_chunk in cdf_flat
    ]
    return torch.cat(cdf_flat, dim=0)


def sample_pdf(bins, weights, num_samples, det=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / weights.sum(-1).unsqueeze(-1)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat((torch.zeros_like(cdf[..., :1]), cdf), -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, num_samples).to(weights)
        u = u.expand(list(cdf.shape[:-1]) + [num_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [num_samples]).to(weights)
    # use pytorch 1.8 searchsorted
    inds = torch.searchsorted(cdf.contiguous(), u.contiguous(), right=True)
    below = torch.max(torch.zeros_like(inds), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack((below, above), -1)

    cdf_g = gather_cdf_util(cdf, inds_g)
    bins_g = gather_cdf_util(bins, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    r"""Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

    Args:
      tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
        is to be computed.

    Returns:
      cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
        tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
    """
    # Only works for the last dimension (dim=-1)
    dim = -1
    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, dim)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, dim)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.0

    return cumprod


def save_tensor_plot(data, save_folder=0, save_name=0):
    plt.rcParams.update({"font.size": 22})
    plt.figure(figsize=(20, 20))
    plt.imshow(data.squeeze().cpu().numpy())
    plt.axis("off")

    if save_folder != 0 and save_name != 0:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.savefig(f"{save_folder}/{save_name}.png")
        plt.cla()
        plt.close()
