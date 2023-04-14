import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import scipy.signal as sig

MIDIFROM = 24
MIDITO = 108
SKIP_SEC = 0.25
HOWMUCH_SEC = 0.5
WHOLETONE_SEC = 2
howmanytones = MIDITO - MIDIFROM + 1
tones = np.arange(MIDIFROM, MIDITO + 1)
s, Fs = sf.read("audio/klavir.wav")
N = int(Fs * HOWMUCH_SEC)
Nwholetone = int(Fs * WHOLETONE_SEC)
xall = np.zeros((MIDITO + 1, N))  # matrix with all tones - first signals empty,
# but we have plenty of memory ...
samplefrom = int(SKIP_SEC * Fs)
sampleto = samplefrom + N
for tone in tones:
    x = s[samplefrom:sampleto]
    x = x - np.mean(x)  # safer to center ...
    xall[tone, :] = x
    samplefrom += Nwholetone
    sampleto += Nwholetone

tone_a = 24
freq_a = 32.7
tone_b = 48
freq_b = 130.81
tone_c = 101
freq_c = 2793.83
tone_samples = Fs * 2


sf.write("audio/a_orig.wav", xall[tone_a], Fs)
sf.write("audio/b_orig.wav", xall[tone_b], Fs)
sf.write("audio/c_orig.wav", xall[tone_c], Fs)


def plot_audio(t, s, fs, tone, ax=None, name='', plot_kwargs={}):
    """  """
    if ax == None:
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
    if name:
        name = ' [' + name + ']'
    ax.plot(t, s, **plot_kwargs)
    ax.set_title(f'MIDI {tone}' + name, pad=25)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude')

    ax.set_xlim(min(t), max(t))
    if tone == 101:
        ylim = (-0.1, 0.1)
        yticks = np.array([-0.1, -0.05, 0, 0.05, 0.1])
    else:
        ylim = (-1., 1.)
        yticks = np.array([-1, -0.5, 0, 0.5, 1])
    for h in [0.5, 0.25, 0.125]:
        if np.max(abs(s)) <= h:
            yticks /= 2.
            ylim = (ylim[0] / 2., ylim[1] / 2.)
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.grid()
    xticks = ax.get_xticks()

    # Second x-axis (samples)
    ax2 = ax.twiny()
    ax2.set_xlabel('Samples', labelpad=10)
    ax2.set_xticks(np.round(xticks * fs).astype(int))
    ax2.set_xlim(min(t) * fs, max(t) * fs)
    plt.tight_layout()

def plot_spectrum(f, g, tone, stop=None, ax=None, plot_kwargs={}):
    if ax is None:
        fig = plt.figure(f"spectrum {tone}", figsize=(12, 4))
        ax = fig.add_subplot(111)
    if stop is None:
        stop = f.size
    ax.plot(f[:stop], g[:stop], **plot_kwargs)  # <= spectrum
    ax.set_xlim(f[:stop].min(), f[:stop].max())
    ax.set_title(f"Power spectral density (PSD), MIDI {tone}")
    ax.set_xlabel('Frequency $[Hz]$', labelpad=15)
    ax.set_ylabel('Mag.\n$[dB]$', labelpad=10, rotation=0)
    plt.xlim(0, 24000)
    plt.xticks(np.arange(24001, step=2000))
    plt.ylim((-140, 30))
    ax.grid()
    plt.tight_layout()

def plot_spectrum_2(f, g, tone, stop=None, ax=None, plot_kwargs={}):
    if ax is None:
        fig = plt.figure(f"spectrum 2, 4.4 {tone}", figsize=(12, 4))
        ax = fig.add_subplot(111)
    if stop is None:
        stop = f.size
    ax.plot(f[:stop], g[:stop], **plot_kwargs, alpha=0.7)  # <= spectrum
    ax.set_xlim(f[:stop].min(), f[:stop].max())
    ax.set_title(f"Power spectral density (PSD), MIDI {tone}")
    ax.set_xlabel('Frequency $[Hz]$', labelpad=15)
    ax.set_ylabel('Mag.\n$[dB]$', labelpad=10, rotation=0)
    plt.xlim(0, 11*precisefmax[tone])
    ymax = np.max(g[int(precisefmax[tone])-3:int(precisefmax[tone])+3])
    ax.scatter(precisefmax[tone], g[round(precisefmax[tone]/2)], color='red', alpha=1)
    ax.scatter(precisefmax[tone] * 2, g[round(precisefmax[tone] * 2/2)], color='red', alpha=1)
    ax.scatter(precisefmax[tone] * 3, g[round(precisefmax[tone] * 3/2)], color='red', alpha=1)
    ax.scatter(precisefmax[tone] * 4, g[round(precisefmax[tone] * 4/2)], color='red', alpha=1)
    ax.scatter(precisefmax[tone] * 5, g[round(precisefmax[tone] * 5/2)], color='red', alpha=1)
    plt.ylim((-140, 30))
    ax.grid()
    plt.tight_layout()

seg_ta = np.arange((3 / freq_a * Fs)) / Fs
seg_tb = np.arange((3 / freq_b * Fs)) / Fs
seg_tc = np.arange((3 / freq_c * Fs)) / Fs
plot_audio(seg_ta, xall[tone_a, :round(3 / freq_a * Fs)], Fs, 24)
plot_audio(seg_tb, xall[tone_b, :round(3 / freq_b * Fs)], Fs, 48)
plot_audio(seg_tc, xall[tone_c, :round(3 / freq_c * Fs)], Fs, 101)

# _, tones = plt.subplots(3, 1, figsize=(10, 6))
# tones[0].plot(xall[tone_a], label="MIDI %d" % tone_a)
# tones[0].legend(loc='upper right')
# tones[0].set_xlabel("samples")
# tones[1].plot(xall[tone_b], label="MIDI %d" % tone_b)
# tones[1].legend(loc='upper right')
# tones[1].set_xlabel("samples")
# tones[2].plot(xall[tone_c], label="MIDI %d" % tone_c)
# tones[2].legend(loc='upper right')
# tones[2].set_xlabel("samples")

N_size = xall[tone_a].size
spec_a = np.fft.fft(xall[tone_a])
spec_b = np.fft.fft(xall[tone_b])
spec_c = np.fft.fft(xall[tone_c])
G_a = 10 * np.log10(1e-10 + 1 / N * np.abs(spec_a) ** 2)
G_b = 10 * np.log10(1e-10 + 1 / N * np.abs(spec_b) ** 2)
G_c = 10 * np.log10(1e-10 + 1 / N * np.abs(spec_c) ** 2)
freq = np.arange(G_a.size) * Fs / N_size
half = freq.size // 2
plot_spectrum(freq, G_a, tone_a, stop=half)
plot_spectrum(freq, G_b, tone_b, stop=half)
plot_spectrum(freq, G_c, tone_c, stop=half)

fund_cor = np.zeros((MIDITO + 1, N))
fund_dft = np.zeros((MIDITO + 1, N))
fund_a = 0
fund_b = 0
fund_c = 0
fundamental = [0] * (MIDITO + 1)
for tone in tones[0: 38]:
    corr = sig.correlate(xall[tone], xall[tone], mode='full')
    fund_cor[tone] = corr[corr.size // 2:]
    i = 0
    while fund_cor[tone, i] > 0:
        fund_cor[tone, i] = 0
        i += 1
    peak = np.argmax(fund_cor[tone])
    fundamental[tone] = Fs / peak
    if tone == tone_a:
        fund_a = peak
    elif tone == tone_b:
        fund_b = peak
for tone in tones[38:]:
    fund_dft[tone] = np.real(np.abs(np.fft.fft(xall[tone])))
    index = np.argmax(fund_dft[tone])
    value = fund_dft[tone, index]
    peaks, _ = sig.find_peaks(fund_dft[tone], height=value * 0.8)
    fundamental[tone] = peaks[0] * Fs / fund_dft[tone].size
    if tone == tone_c:
        fund_c = peaks[0]
    print(f"DFT {tone} {fundamental[tone]}")
print("\n")

a = plt.figure("Fundamental a", figsize=(16, 4))
suba = a.add_subplot(111)
plt.plot(fund_cor[tone_a])
suba.set_title("Autocorrelation of MIDI 24")
plt.xlim(0, 24000)
plt.xticks(np.arange(24001, step=2000))
plt.ylim((-120, 210))
plt.hlines(y=100, xmin=0, xmax=fund_a, color='r', label="Fundamental of MIDI 24")
plt.axvline(x=fund_a, color='r', linestyle='--')
plt.legend(loc='upper right')
suba.set_xlabel("samples")
suba.set_ylabel("Correlation", labelpad=10, rotation=90)
plt.grid()

b = plt.figure("Fundamental b", figsize=(16, 4))
subb = b.add_subplot(111)
plt.plot(fund_cor[tone_b])
subb.set_title("Autocorrelation of MIDI 48")
plt.xlim(0, 24000)
plt.xticks(np.arange(24001, step=2000))
plt.ylim((-160, 180))
plt.axvline(x=fund_b, color='r', linestyle='--')
plt.hlines(y=100, xmin=0, xmax=fund_b, color='r', label="Fundamental of MIDI 48")
plt.legend(loc='upper right')
subb.set_xlabel("samples")
subb.set_ylabel("Correlation", labelpad=10, rotation=90)
plt.grid()

c = plt.figure("Fundamental c", figsize=(16, 4))
subc = c.add_subplot(111)
plt.plot(freq[:half], fund_dft[tone_c, :half])
subc.set_title("Spectrum of MIDI 101")
plt.xlim(0, 24000)
plt.xticks(np.arange(24001, step=2000))
plt.ylim((-10, 20))
plt.axvline(x=fund_c*2, color='r', linewidth=0.75, linestyle='-', label="Fundamental of MIDI 101")
plt.legend(loc='upper right')
subc.set_xlabel("frequency [Hz]")
subc.set_ylabel('Mag.\n$[dB]$', labelpad=10, rotation=0)
plt.grid()

################################# 4.3 #################################

FREQPOINTS = 200
precisefmax = [0] * (MIDITO + 1)
for tone in tones:
    FREQRANGE = fundamental[tone] * 2 ** (5 / 1200) - fundamental[tone]  # 5 cents to both sides
    fsweep = np.linspace(fundamental[tone] - FREQRANGE, fundamental[tone] + FREQRANGE, FREQPOINTS)
    n = np.arange(N)
    A = np.zeros([FREQPOINTS, N], dtype=complex)
    for k in np.arange(0, FREQPOINTS):
        A[k, :] = np.exp(-1j * 2 * np.pi * fsweep[k] / Fs * n)  # norm. omega = 2 * pi * f / Fs ...
    Xdtft = np.matmul(A, xall[tone].T)
    precisefmax[tone] = fsweep[np.argmax(np.abs(Xdtft))]
    print(f"DTFT {tone} {precisefmax[tone]}")


################################# 4.4 #################################

xall_dft = np.zeros((MIDITO + 1, N))
Xdtft = []
float_rep = [0] * 10
RANGE = 50
xall_float_rep = []
kall = np.arange(0, int(N / 2) + 1)
for tone in tones:
    xall_dft[tone] = np.real(np.fft.fft(xall[tone]))
    fsweep = [precisefmax[tone], precisefmax[tone] * 2, precisefmax[tone] * 3, precisefmax[tone] * 4, precisefmax[tone] * 5]
    n = np.arange(N)
    A = np.zeros([5, N], dtype=complex)
    for k in np.arange(0, 5):
        A[k, :] = np.exp(-1j * 2 * np.pi * fsweep[k] / Fs * n)
    Xdtft.append((np.matmul(A, xall[tone].T)))
    for i in range(5):
        float_rep[i] = np.abs(Xdtft[tone-24][i])
    for i in range(5, 10):
        float_rep[i] = np.angle(Xdtft[tone-24][i-5])
    xall_float_rep.append(float_rep)

plot_spectrum_2(freq, G_a, tone_a, stop=half)
plot_spectrum_2(freq, G_b, tone_b, stop=half)
plot_spectrum_2(freq, G_c, tone_c, stop=half)

################################# 4.5 doesn't work#################################
amplitudes = xall_float_rep[tone_a-24][0:5]
phases = xall_float_rep[tone_a-24][5:10]
NN = len(amplitudes)
x = np.zeros(NN)
for n in range(NN):
    for k in range(NN):
        x[n] += np.real(amplitudes[k] * np.exp(1j * phases[k] * 2 * np.pi * k * n / NN))
    x[n] /= NN

#plt.figure("Synthesized signal", figsize=(16, 4))
#plt.plot(x)
#plt.title("Synthesized signal")
#plt.xlim(0, N)
#plt.xticks(np.arange(N + 1, step=2000))
#plt.ylim((-1.5, 1.5))
#plt.xlabel("samples")
#plt.ylabel("Amplitude", labelpad=10, rotation=0)
#plt.grid()

plt.show()
