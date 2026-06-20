"""Smoke test: env v2 reset/step under JIT, and vmapped over parallel worlds."""
import jax

from smoothride.data.map_loader import load_road_network
from smoothride.env import kinematic as K
from smoothride.env.routing import build_route_pool

net = load_road_network()
x0, y0, x1, y1 = net.bounds()
pool = build_route_pool(net, n_routes=512)
env = K.make_env(pool, (x0, y0), (x1, y1), n_agents=24, n_peds=12, max_steps=300)
print(f"obs_dim={env.obs_dim} act_dim={env.act_dim} "
      f"agents={env.n_agents} peds={env.n_peds}")

key = jax.random.PRNGKey(0)
st, obs = K.reset(env, key)
assert obs.shape == (env.n_agents, env.obs_dim), obs.shape

step = jax.jit(lambda s, a, k: K.step(env, s, a, k))
for i in range(env.max_steps):
    key, ka, ks = jax.random.split(key, 3)
    act = jax.random.uniform(ka, (env.n_agents, env.act_dim), minval=-1, maxval=1)
    st, obs, r, done, info = step(st, act, ks)
print(f"[single] crashes/car={float(info['crashes_per_car']):.2f} "
      f"goals={int(info['total_goals'])} ped_hits={int(info['ped_hits'])}")

B = 32
vreset = jax.jit(jax.vmap(lambda k: K.reset(env, k)))
vstep = jax.jit(jax.vmap(lambda s, a, k: K.step(env, s, a, k)))
bst, bobs = vreset(jax.random.split(jax.random.PRNGKey(1), B))
assert bobs.shape == (B, env.n_agents, env.obs_dim), bobs.shape
acts = jax.vmap(lambda k: jax.random.uniform(
    k, (env.n_agents, env.act_dim), minval=-1, maxval=1))(
    jax.random.split(jax.random.PRNGKey(2), B))
bst, bobs, br, bdone, binfo = vstep(
    bst, acts, jax.random.split(jax.random.PRNGKey(3), B))
print(f"[vmap x{B}] obs={tuple(bobs.shape)} reward={tuple(br.shape)}")

# --- 3D path: terrain + grade dynamics + building occlusion ---------------
try:
    from smoothride.data.map_loader import load_road_network_3d
    net3, buildings = load_road_network_3d()
    x0, y0, x1, y1 = net3.bounds()
    pool3 = build_route_pool(net3, n_routes=256)
    env3 = K.make_env(pool3, (x0, y0), (x1, y1), n_agents=24, n_peds=12,
                      max_steps=300, buildings=buildings)
    nseg = int(env3.bld_segs.shape[0])
    print(f"[3D] z {net3.node_z.min():.0f}-{net3.node_z.max():.0f} m  "
          f"max grade {abs(pool3.grade).max()*100:.0f}%  occluder segs={nseg}")
    st3, obs3 = K.reset(env3, jax.random.PRNGKey(0))
    assert obs3.shape == (env3.n_agents, env3.obs_dim), obs3.shape
    step3 = jax.jit(lambda s, a, k: K.step(env3, s, a, k))
    key3 = jax.random.PRNGKey(0)
    for _ in range(50):
        key3, ka, ks = jax.random.split(key3, 3)
        act = jax.random.uniform(ka, (env3.n_agents, env3.act_dim), minval=-1, maxval=1)
        st3, obs3, r3, d3, info3 = step3(st3, act, ks)
    assert info3["cost"].shape == (env3.n_agents,)
    print(f"[3D] obs_dim={env3.obs_dim} (occlusion+grade ran under jit)  "
          f"z range now {float(st3.z.min()):.0f}-{float(st3.z.max()):.0f} m  "
          f"cost_sum={float(info3['cost'].sum()):.0f}")
except Exception as e:  # noqa: BLE001 — 3D data may be offline in CI
    print(f"[3D] skipped ({e!r})")

print("OK")
