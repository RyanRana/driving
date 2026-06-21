"""Shared-parameter actor + centralized critic (CTDE).

All agents share one policy (homogeneous cars). The critic is centralized: it
sees each agent's local obs plus a pooled summary of the whole scene, which is
what lets agents learn to anticipate/avoid each other (the multi-agent point).
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class ActorCritic(nn.Module):
    act_dim: int
    hidden: int = 128

    @nn.compact
    def __call__(self, obs, global_feat):
        # --- actor: decentralized, local obs only ---
        x = obs
        x = nn.tanh(nn.Dense(self.hidden)(x))
        x = nn.tanh(nn.Dense(self.hidden)(x))
        mean = nn.Dense(self.act_dim,
                        kernel_init=nn.initializers.orthogonal(0.01))(x)
        log_std = self.param("log_std",
                             nn.initializers.constant(-0.5), (self.act_dim,))

        # --- critic: centralized, local obs + pooled scene summary ---
        c = jnp.concatenate([obs, global_feat], axis=-1)
        c = nn.tanh(nn.Dense(self.hidden)(c))
        c = nn.tanh(nn.Dense(self.hidden)(c))
        value = nn.Dense(1)(c)[..., 0]
        return mean, log_std, value


class DeepSets(nn.Module):
    """Permutation-invariant set encoder: per-element MLP phi, then masked
    mean+max pool. Empty set -> zeros. Density-agnostic, handles padded slots."""

    feat_dim: int
    hidden: int = 64

    @nn.compact
    def __call__(self, entities: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        """Encode a set of entities with validity mask.

        Args:
            entities: (..., C, feat_dim) — one row per slot.
            mask: (..., C) bool — True for valid slots, False for padding.

        Returns:
            (..., 2*hidden) — concat of masked mean-pool and max-pool.
        """
        h = nn.relu(nn.Dense(self.hidden)(entities))
        h = nn.relu(nn.Dense(self.hidden)(h))          # (..., C, hidden)
        m = mask[..., None].astype(h.dtype)            # (..., C, 1) float
        h = h * m                                      # zero invalid slots
        summed = jnp.sum(h, axis=-2)                   # (..., hidden)
        count = jnp.clip(jnp.sum(m, axis=-2), 1.0)    # live count >= 1
        mean = summed / count
        # masked max: push invalid slots to large-negative so they never win
        neg = jnp.where(m > 0, h, -1e9)
        mx = jnp.max(neg, axis=-2)                     # (..., hidden)
        any_valid = jnp.sum(m, axis=-2) > 0            # (..., 1) bool
        mx = jnp.where(any_valid, mx, 0.0)             # guard empty set
        return jnp.concatenate([mean, mx], axis=-1)    # (..., 2*hidden)


def gaussian_logp(actions, mean, log_std):
    std = jnp.exp(log_std)
    pre = -0.5 * (((actions - mean) / std) ** 2) - log_std - 0.5 * jnp.log(2 * jnp.pi)
    return pre.sum(-1)


def gaussian_entropy(log_std):
    return (log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e)).sum(-1)
