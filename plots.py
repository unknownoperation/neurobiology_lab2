import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import welch


def plot_PSD(freqs, psds):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(freqs, psds)
    ax.set_yscale('log')
    plt.title('Power Spectral Density', fontsize=20)
    plt.xlabel('Frequency', fontsize=20)
    plt.ylabel('Power', fontsize=20)
    plt.show()


def plot_PLV(phases):
    x = np.exp(1j * np.stack(phases, axis=0))
    x = np.mean(x, axis=-1)
    y = np.conj(x)
    x /= np.abs(x)
    y /= np.abs(y)
    signals = np.abs((x @ y.T) / x.shape[1])

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1, 1, 1)
    sns.heatmap(signals, cmap="rocket", ax=ax)
    ax.set_xlabel('Signals', fontsize=20)
    ax.set_ylabel('Signals', fontsize=20)
    ax.set_title('PLV heatmap', fontsize=20)
    plt.show()


def plot_PSD_heathmap(signals):
    table = []
    for signal in np.split(signals, 60):
        freq, psd = welch(signal, fs=int(1 / 0.01), scaling="spectrum", nperseg=len(signal), nfft=len(signal))
        table.append(psd)

    fig = plt.figure(figsize=(10, 16))
    ax = fig.add_subplot(1, 1, 1)
    sns.heatmap(table, cmap="rocket", ax=ax)
    ax.set_xlabel('Frequency', fontsize=20)
    ax.set_ylabel('Time', fontsize=20)
    ax.set_title('PSD heatmap', fontsize=20)
    plt.show()
