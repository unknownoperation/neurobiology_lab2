import numpy as np


class KuramotoModel:
    def __init__(self, t_k, dt, omega_t, K=1, N=500, sigma=1, freq=10):
        self.K = K
        self.N = N
        self.dt = dt
        self.sigma = sigma
        self.timestamps = np.arange(0, t_k, dt)
        self.phi_0 = np.random.uniform(-np.pi, np.pi, size=self.N)
        self.omega_t = omega_t
        self.freq = freq
        self.phases = np.zeros((len(self.timestamps), self.N))

        phases = np.zeros((len(self.timestamps), self.N))
        phases[0, :] = self.phi_0

        for i_t, t in list(enumerate(self.timestamps))[1:]:
            internal = 2 * np.pi * self.omega_t(t, self.freq, self.N)
            delta = np.sin(phases[i_t - 1, :] - phases[i_t - 1, :][np.newaxis, :])
            external = self.K / self.N * np.sum(delta, axis=1)
            phases[i_t, :] = phases[i_t - 1, :] + self.dt * (internal + external + np.random.normal(0, self.sigma))

        self.phases = phases
        self.signal = np.exp(1j * self.phases).mean(axis=1).real

    def get_phases(self):
        return self.phases

    def get_signal(self):
        return self.signal

