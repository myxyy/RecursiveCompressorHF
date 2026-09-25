"""State-carrying extension of the official Mamba-2 unfused forward path.

The upstream module and parameters are used unchanged. The extension supplies
initial_states to the official SSD kernel and prepends the convolution history.
A PyTorch recurrence is retained as an independent, differentiable CPU reference.
State tensors are not mutated. No detach occurs at chunk boundaries.
"""
import torch
import torch.nn.functional as F


def rms_norm(x, weight, eps):
    dtype = x.dtype
    accum = x if dtype == torch.float64 else x.float()
    return (accum * torch.rsqrt(accum.square().mean(-1, keepdim=True) + eps)
            * weight).to(dtype)


def reference_scan(x, dt, A, B, C, D, dt_bias, initial=None):
    """Mamba-2's scalar-per-head transition and grouped input/output maps."""
    acc = torch.float64 if x.dtype == torch.float64 else torch.float32
    x, dt, A, B, C, D, dt_bias = [t.to(acc) for t in (x, dt, A, B, C, D, dt_bias)]
    heads, width = x.shape[-2:]
    B = B.repeat_interleave(heads // B.shape[2], dim=2)
    C = C.repeat_interleave(heads // C.shape[2], dim=2)
    state = x.new_zeros(x.shape[0], heads, width, B.shape[-1]) if initial is None else initial.to(acc)
    delta = F.softplus(dt + dt_bias)
    outputs = []
    for t in range(x.shape[1]):
        state = (torch.exp(delta[:, t] * A)[:, :, None, None] * state
                 + delta[:, t, :, None, None] * x[:, t, :, :, None] * B[:, t, :, None, :])
        outputs.append((state * C[:, t, :, None, :]).sum(-1) + D[:, None] * x[:, t])
    return torch.stack(outputs, dim=1), state


def mixer_step(mixer, u, state=None, *, reference=False):
    batch, length, _ = u.shape
    if length < 1:
        raise ValueError("A chunk must contain at least one token")
    projected = mixer.in_proj(u)
    z, xBC, dt = projected.split(
        [mixer.d_inner, mixer.d_ssm + 2 * mixer.ngroups * mixer.d_state, mixer.nheads], dim=-1)
    conv_input = xBC.transpose(1, 2)
    history_len = mixer.d_conv - 1
    if state is None:
        history = conv_input.new_zeros(batch, conv_input.shape[1], history_len)
        initial = None
    else:
        history, initial = state
        expected = (batch, mixer.nheads, mixer.headdim, mixer.d_state)
        if history.shape != (batch, conv_input.shape[1], history_len) or initial.shape != expected:
            raise ValueError("State batch/model dimensions do not match the input")
        if history.device != u.device or initial.device != u.device:
            raise ValueError("State and input must be on the same device")
        history = history.to(conv_input.dtype)
    joined = torch.cat([history, conv_input], dim=-1)
    # Clone avoids keeping a whole input chunk alive through a small slice.
    next_history = joined[:, :, -history_len:].clone() if history_len else joined[:, :, :0].clone()
    xBC = F.silu(F.conv1d(joined, mixer.conv1d.weight, mixer.conv1d.bias,
                         groups=mixer.conv1d.groups)).transpose(1, 2)
    x, B, C = xBC.split([mixer.d_ssm, mixer.ngroups * mixer.d_state,
                        mixer.ngroups * mixer.d_state], dim=-1)
    x = x.reshape(batch, length, mixer.nheads, mixer.headdim)
    B = B.reshape(batch, length, mixer.ngroups, mixer.d_state)
    C = C.reshape_as(B)
    A = -mixer.A_log.to(torch.float64 if u.dtype == torch.float64 else torch.float32).exp()
    if reference or not u.is_cuda:
        y, final = reference_scan(x, dt, A, B, C, mixer.D, mixer.dt_bias, initial)
        y = y.to(x.dtype)
    else:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
        y, final = mamba_chunk_scan_combined(
            x, dt, A, B, C, mixer.chunk_size, D=mixer.D,
            dt_bias=mixer.dt_bias, dt_softplus=True, initial_states=initial,
            return_final_states=True,
        )
    y = y.reshape(batch, length, mixer.d_inner)
    if u.is_cuda and not reference:
        y = mixer.norm(y, z)
    else:
        # Official Mamba2 defaults: RMSNorm(x * silu(z)), one norm per group.
        gated = y.to(torch.float64 if y.dtype == torch.float64 else torch.float32) * F.silu(z.to(torch.float64 if z.dtype == torch.float64 else torch.float32))
        grouped = gated.reshape(batch, length, mixer.ngroups, -1)
        weight = mixer.norm.weight.reshape(mixer.ngroups, -1)
        y = rms_norm(grouped, weight, mixer.norm.eps).reshape_as(y).to(y.dtype)
    return mixer.out_proj(y), (next_history, final)
