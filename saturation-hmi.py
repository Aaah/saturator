import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import saturation as sat
from scipy.signal import welch
import sounddevice as sd
import scipy.io.wavfile as sio

def psd(arr, fs):
    N = 1024
    f, Pxx = welch(arr, fs=fs, window='hanning', nperseg=N, noverlap=N/2, nfft=N, return_onesided=True, scaling='spectrum')
    return Pxx, f

# -- init
f0 = 400.0
fs = 16000.0
tdisplay = 1. / f0
ndisplay = int(tdisplay * fs)
amp = 1.0
nsamples = int(1.0 * fs)
sig = amp * np.sin(2.0 * 3.1415 * f0 / fs * np.arange(0, nsamples))

s = sat.Saturator()
res = s.process(sig)

# -- plot
fig, ax = plt.subplots(figsize=(6,8))
plt.subplot(311)
l1, = plt.plot(res[:ndisplay], lw=1)
lthre1, = plt.plot(s.threshold * np.ones(ndisplay), 'r--', lw=1)
lthre2, = plt.plot(-s.threshold * np.ones(ndisplay), 'r--', lw=1)
plt.ylabel("Amplitude")
plt.xlabel("Time (samples)")
ax.axis('tight')
plt.ylim([-1.1, +1.1])

p2 = plt.subplot(312)
nrj, w = psd(res, fs)
l2, = plt.semilogy(w, nrj, lw=1)
l3 = plt.fill_between(w, nrj, -10000.0 * np.ones(len(nrj)), color=(0.0, 0.0, 1.0, 0.15))

plt.grid(linestyle='--', color='gray')
plt.ylabel("Magnitude (dB)")
plt.xlabel("Frequencies (Hz)")
plt.ylim([1e-15, 1.0])
fig.tight_layout()

# -- commands
axcolor = 'white'
ax_thre = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor=axcolor)
ax_ratio = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_symm = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_listen = plt.axes([0.4, 0.2, 0.3, 0.03], facecolor=axcolor)

s_thre = Slider(ax_thre, 'Threshold', 0.0, 1.0, valinit=1.0, valstep=0.001)
s_ratio = Slider(ax_ratio, 'Ratio', 1.0, 25.0, valinit=1.0, valstep=0.1)
s_symm = Slider(ax_symm, 'Symmetry', 0.0, 1.0, valinit=1.0, valstep=0.001)
b_listen = Button(ax_listen, 'Export', color=axcolor, hovercolor='0.8')

def update(val):
    s.threshold = s_thre.val
    s.ratio = s_ratio.val
    s.symmetry = s_symm.val

    s.update_params()

    global res
    res = s.process(sig)
    l1.set_ydata(res[:ndisplay])
    lthre1.set_ydata(s.threshold * np.ones(ndisplay))
    lthre2.set_ydata(-s.threshold * np.ones(ndisplay))

    nrj, w = psd(res, fs)
    l2.set_ydata(nrj)
    p2.collections.clear()
    p2.fill_between(w, nrj, -10000.0 * np.ones(len(nrj)), color=(0.0, 0.0, 1.0, 0.15))

    fig.canvas.draw_idle()

def export(val):
    sio.write("out.wav", int(fs), res)
    # sd.play(res, fs, blocking=True)
    return

# -- tie widgets to actions
s_thre.on_changed(update)
s_ratio.on_changed(update)
s_symm.on_changed(update)
b_listen.on_clicked(export)

plt.show()