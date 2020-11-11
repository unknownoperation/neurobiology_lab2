import numpy as np
import ray

from scipy.signal import welch

from model import KuramotoModel
from plots import *


@ray.remote
def run_simulation(freq, omega, return_phases=False):
    dt = 0.01
    model = KuramotoModel(t_k=60, dt=dt, omega_t=omega, K=1.5, N=500, sigma=1, freq=freq)

    signal = model.get_signal()
    freqs, psd = welch(signal, fs=int(1/dt), scaling="spectrum")
    if return_phases:
        return psd, freqs, signal, model.get_phases()
    return psd, freqs, signal


def experiment1():
    def omega1(t, f, n):
        return np.random.uniform(0.75 * f, 1.25 * f, size=n)

    freqs = [10] * 10 + [15] * 5 + [25] * 5
    result_ids = []
    for freq in freqs:
        result_ids.append(run_simulation.remote(freq, omega1, return_phases=True))
    results = np.array(ray.get(result_ids))

    plot_PSD(np.mean(results[:, 1], axis=0), np.mean(results[:, 0], axis=0))
    plot_PLV(results[:, 3])


def experiment2():
    def omega2(t, f, n):
        assert t >= 0
        main_freq = 0
        if t <= 15:
            main_freq = 10
        elif t <= 30:
            main_freq = 20
        elif t <= 45:
            main_freq = 25
        elif t <= 60:
            main_freq = 10

        return np.random.uniform(0.75 * main_freq, 1.25 * main_freq, size=n)

    freq = 10
    result_ids = []
    for i in range(100):
        result_ids.append(run_simulation.remote(freq, omega2))
    results = np.array(ray.get(result_ids))

    plot_PSD(np.mean(results[:, 1], axis=0), np.mean(results[:, 0], axis=0))
    plot_PSD_heathmap(np.mean(results[:, 2], axis=0))


if __name__ == '__main__':
    ray.init()

    experiment1()
    experiment2()





