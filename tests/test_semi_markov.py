"""Prueba de la formulación semi-Markov en los buffers de DQN y PPO (sección 3.3).

Usa un entorno de juguete que reporta en info una duración conocida, para no depender de env.py.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import DQN, PPO

from train import CallbackDuraciones, ReplayBufferSemiMarkov, RolloutBufferSemiMarkov

GAMMA = 0.9


class EntornoDuraciones(gym.Env):
    """Cada acción a dura a + 1 barras; el episodio se trunca a los 20 pasos."""

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pasos = 0
        return self.observation_space.sample(), {}

    def step(self, action):
        self.pasos += 1
        obs = self.np_random.uniform(-1, 1, size=2).astype(np.float32)
        return obs, float(action), False, self.pasos >= 20, {"duracion": int(action) + 1}


def test_dqn_descuenta_con_la_duracion_real():
    modelo = DQN("MlpPolicy", EntornoDuraciones(), gamma=GAMMA, learning_starts=50, buffer_size=1_000,
                 replay_buffer_class=ReplayBufferSemiMarkov, replay_buffer_kwargs={"gamma": GAMMA}, seed=0)
    modelo.learn(200)
    buffer = modelo.replay_buffer
    n = buffer.pos
    # La duración guardada corresponde a la acción de esa misma transición
    assert np.array_equal(buffer.duraciones[:n, 0], buffer.actions[:n, 0, 0] + 1)
    muestra = buffer.sample(64)
    acciones = muestra.actions.cpu().numpy().ravel()
    assert np.allclose(muestra.discounts.cpu().numpy().ravel(), GAMMA ** (acciones + 1))


def test_ppo_calcula_gae_con_la_duracion_real():
    # Con gae_lambda = 0 la ventaja es el error TD de un paso: r + gamma^tau * V(s') - V(s)
    modelo = PPO("MlpPolicy", EntornoDuraciones(), gamma=GAMMA, gae_lambda=0.0, n_steps=64, batch_size=32,
                 n_epochs=1, rollout_buffer_class=RolloutBufferSemiMarkov, seed=0)
    # Solo se juntan los datos, como en el primer paso de learn(): al entrenar, SB3 aplana el buffer
    _, callback = modelo._setup_learn(64, callback=CallbackDuraciones())
    callback.on_training_start(locals(), globals())
    modelo.collect_rollouts(modelo.env, callback, modelo.rollout_buffer, n_rollout_steps=64)
    buffer = modelo.rollout_buffer
    assert np.array_equal(buffer.duraciones[:, 0], buffer.actions[:, 0, 0] + 1)
    revisados = 0
    for t in range(buffer.buffer_size - 1):
        if buffer.episode_starts[t + 1, 0] == 0:  # el paso siguiente es del mismo episodio
            esperado = (buffer.rewards[t, 0] + GAMMA ** buffer.duraciones[t, 0] * buffer.values[t + 1, 0]
                        - buffer.values[t, 0])
            assert np.isclose(buffer.advantages[t, 0], esperado, atol=1e-5)
            revisados += 1
    assert revisados > 0
