import torch
from torch import Tensor

from ai_osu_maps.config import FlowConfig

# Conditioning keys that can be dropped for CFG
_DROPPABLE_KEYS = ("difficulty", "cs", "ar", "od", "hp", "mapper", "year")


def _build_drop_mask(
    keys: tuple[str, ...],
    *,
    drop_all: bool,
    batch_size: int,
    device: torch.device,
) -> dict[str, Tensor]:
    """Build a drop_mask dict for the conditioning module.

    When drop_all is True, all conditions are masked (unconditional pass).
    When False, no conditions are masked (conditional pass).
    The timestep key is never dropped.
    """
    mask: dict[str, Tensor] = {}
    for key in keys:
        mask[key] = torch.full(
            (batch_size,), fill_value=drop_all, dtype=torch.bool, device=device
        )
    # Timestep is never dropped
    mask["timestep"] = torch.zeros(batch_size, dtype=torch.bool, device=device)
    return mask


def _call_model(
    model: torch.nn.Module,
    x_t: Tensor,
    t_tensor: Tensor,
    audio_features: Tensor,
    cond: dict[str, Tensor],
    drop_mask: dict[str, Tensor],
) -> Tensor:
    """Call FlowTransformer.forward and return only the velocity prediction."""
    velocity, _log_num_objects = model(
        x_t,
        t_tensor,
        audio_features,
        cond["difficulty"],
        cond["cs"],
        cond["ar"],
        cond["od"],
        cond["hp"],
        cond["mapper_id"],
        cond["year"],
        drop_mask,
    )
    return velocity


def euler_step(
    model: torch.nn.Module,
    x_t: Tensor,
    t: float,
    dt: float,
    audio_features: Tensor,
    cond: dict[str, Tensor],
    cfg_scale: float,
    device: torch.device,
) -> Tensor:
    """Single Euler step with classifier-free guidance.

    Computes x_{t+dt} = x_t + dt * v_cfg where
    v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond).

    Args:
        model: FlowTransformer model.
        x_t: Current noisy vectors of shape (B, N, D).
        t: Current timestep in [0, 1].
        dt: Step size.
        audio_features: Encoded audio of shape (B, T_audio, d_model).
        cond: Dict of conditioning tensors (difficulty, cs, ar, od, hp, mapper_id, year).
        cfg_scale: Classifier-free guidance scale.
        device: Torch device.

    Returns:
        Updated vectors x_{t+dt} of shape (B, N, D).
    """
    batch_size = x_t.shape[0]
    t_tensor = torch.full((batch_size,), t, dtype=x_t.dtype, device=device)

    cond_mask = _build_drop_mask(
        _DROPPABLE_KEYS, drop_all=False, batch_size=batch_size, device=device
    )
    v_cond = _call_model(model, x_t, t_tensor, audio_features, cond, cond_mask)

    uncond_mask = _build_drop_mask(
        _DROPPABLE_KEYS, drop_all=True, batch_size=batch_size, device=device
    )
    v_uncond = _call_model(model, x_t, t_tensor, audio_features, cond, uncond_mask)

    v = v_uncond + cfg_scale * (v_cond - v_uncond)
    return x_t + dt * v


def midpoint_step(
    model: torch.nn.Module,
    x_t: Tensor,
    t: float,
    dt: float,
    audio_features: Tensor,
    cond: dict[str, Tensor],
    cfg_scale: float,
    device: torch.device,
) -> Tensor:
    """Midpoint method step with classifier-free guidance.

    Computes v at t, steps to the midpoint, recomputes v there, then uses the
    midpoint velocity for the full step.

    Args:
        model: FlowTransformer model.
        x_t: Current noisy vectors of shape (B, N, D).
        t: Current timestep in [0, 1].
        dt: Step size.
        audio_features: Encoded audio of shape (B, T_audio, d_model).
        cond: Dict of conditioning tensors.
        cfg_scale: Classifier-free guidance scale.
        device: Torch device.

    Returns:
        Updated vectors x_{t+dt} of shape (B, N, D).
    """
    batch_size = x_t.shape[0]

    cond_mask = _build_drop_mask(
        _DROPPABLE_KEYS, drop_all=False, batch_size=batch_size, device=device
    )
    uncond_mask = _build_drop_mask(
        _DROPPABLE_KEYS, drop_all=True, batch_size=batch_size, device=device
    )

    # Velocity at t
    t_tensor = torch.full((batch_size,), t, dtype=x_t.dtype, device=device)
    v_cond_1 = _call_model(model, x_t, t_tensor, audio_features, cond, cond_mask)
    v_uncond_1 = _call_model(model, x_t, t_tensor, audio_features, cond, uncond_mask)
    v_1 = v_uncond_1 + cfg_scale * (v_cond_1 - v_uncond_1)

    # Step to midpoint
    x_mid = x_t + (dt / 2) * v_1
    t_mid = torch.full(
        (batch_size,), t + dt / 2, dtype=x_t.dtype, device=device
    )

    # Velocity at midpoint
    v_cond_2 = _call_model(model, x_mid, t_mid, audio_features, cond, cond_mask)
    v_uncond_2 = _call_model(model, x_mid, t_mid, audio_features, cond, uncond_mask)
    v_2 = v_uncond_2 + cfg_scale * (v_cond_2 - v_uncond_2)

    return x_t + dt * v_2


_SOLVERS = {
    "euler": euler_step,
    "midpoint": midpoint_step,
}


@torch.no_grad()
def sample(
    model: torch.nn.Module,
    audio_features: Tensor,
    num_objects: int,
    cond: dict[str, Tensor],
    config: FlowConfig,
    device: torch.device,
) -> Tensor:
    """Generate object vectors by sampling from the flow model.

    Integrates the learned velocity field from t=0 (noise) to t=1 (data)
    using the configured ODE solver with classifier-free guidance.

    Args:
        model: FlowTransformer model.
        audio_features: Encoded audio of shape (1, T_audio, d_model).
        num_objects: Number of hit objects to generate.
        cond: Dict of conditioning tensors with keys: difficulty, cs, ar, od,
            hp, mapper_id, year (each of shape (1,)).
        config: Flow sampling configuration.
        device: Torch device.

    Returns:
        Final generated vectors of shape (1, num_objects, obj_dim).
    """
    solver_fn = _SOLVERS[config.solver]

    # Initialize from standard normal noise
    x_t = torch.randn(1, num_objects, 32, device=device)

    # Time schedule: uniform steps from 0 to 1
    schedule = torch.linspace(0.0, 1.0, config.n_steps + 1)

    model.eval()
    for i in range(config.n_steps):
        t = schedule[i].item()
        dt = (schedule[i + 1] - schedule[i]).item()

        x_t = solver_fn(
            model=model,
            x_t=x_t,
            t=t,
            dt=dt,
            audio_features=audio_features,
            cond=cond,
            cfg_scale=config.cfg_scale,
            device=device,
        )

    return x_t
