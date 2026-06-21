import jax
import jax.numpy as jnp
import numpy as np

from smoothride.rl.networks import DeepSets


def _apply(mod, ents, mask):
    p = mod.init(jax.random.PRNGKey(0), ents, mask)
    return mod.apply(p, ents, mask), p


def test_output_shape_and_empty_set_is_zero():
    mod = DeepSets(feat_dim=4, hidden=8)
    ents = jnp.ones((3, 5, 4))                      # batch 3, cap 5, feat 4
    mask = jnp.zeros((3, 5), bool)                  # all empty
    out, p = _apply(mod, ents, mask)
    assert out.shape == (3, 16)                     # 2*hidden
    np.testing.assert_allclose(np.asarray(out), 0.0, atol=1e-6)


def test_permutation_invariance():
    mod = DeepSets(feat_dim=4, hidden=8)
    ents = jax.random.normal(jax.random.PRNGKey(1), (1, 5, 4))
    mask = jnp.array([[True, True, True, False, False]])
    out_a, p = _apply(mod, ents, mask)
    perm = jnp.array([2, 0, 1, 3, 4])              # shuffle only valid+padding consistently
    out_b = mod.apply(p, ents[:, perm], mask[:, perm])
    np.testing.assert_allclose(np.asarray(out_a), np.asarray(out_b), atol=1e-5)


def test_masked_slots_do_not_affect_output():
    mod = DeepSets(feat_dim=4, hidden=8)
    ents = jax.random.normal(jax.random.PRNGKey(2), (1, 5, 4))
    mask = jnp.array([[True, True, False, False, False]])
    out_a, p = _apply(mod, ents, mask)
    # garbage in the masked slots must not change the result
    ents2 = ents.at[:, 2:].set(999.0)
    out_b = mod.apply(p, ents2, mask)
    np.testing.assert_allclose(np.asarray(out_a), np.asarray(out_b), atol=1e-5)
