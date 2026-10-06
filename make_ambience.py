"""Synthesize a realistic, non-distracting clinic reception ambience loop.

Layers: room tone + HVAC, ceiling fan, unintelligible walla (from TTS clips),
aperiodic micro-sounds (typing, mouse, paper, chair, distant horn).
Everything is band-limited to telephone range (300-3400 Hz).
"""
import glob
import numpy as np
from scipy import signal
from scipy.io import wavfile

SR = 22050
DUR = 180          # loop length in seconds
XFADE = 3          # seconds crossfaded for a seamless loop
N = (DUR + XFADE) * SR
rng = np.random.default_rng(7)


def bp(x, lo, hi, order=4):
    sos = signal.butter(order, [lo, hi], btype="band", fs=SR, output="sos")
    return signal.sosfilt(sos, x)


def lp(x, hi, order=4):
    sos = signal.butter(order, hi, btype="low", fs=SR, output="sos")
    return signal.sosfilt(sos, x)


def norm(x):
    m = np.max(np.abs(x))
    return x / m if m > 0 else x


def brown(n):
    b = np.cumsum(rng.standard_normal(n))
    b = signal.detrend(b)
    return norm(lp(b, 800))


def add(out, clip, t, gain):
    i = int(t * SR)
    j = min(len(out), i + len(clip))
    if i < len(out):
        out[i:j] += gain * clip[: j - i]


t = np.arange(N) / SR

# 1. Room tone + HVAC: soft broadband hiss and low structural rumble
room = 0.25 * norm(bp(rng.standard_normal(N), 300, 3400)) + 0.6 * brown(N)

# 2. Ceiling fan: brown noise with gentle blade modulation (~3 Hz, slightly drifting)
fan_mod = 1 + 0.18 * np.sin(2 * np.pi * (3.1 + 0.05 * np.sin(2 * np.pi * t / 23)) * t)
fan = norm(lp(rng.standard_normal(N), 650)) * fan_mod

# 3. Walla: overlapping, muffled, reverberant, partly reversed speech -> unintelligible
ir_len = int(0.7 * SR)
ir = rng.standard_normal(ir_len) * np.exp(-np.linspace(0, 7, ir_len))
clips = []
for f in sorted(glob.glob("l*.wav")):
    sr, d = wavfile.read(f)
    d = d.astype(np.float64)
    if d.ndim > 1:
        d = d.mean(axis=1)
    d = signal.resample_poly(d, SR, sr)
    clips.append(norm(d))
walla = np.zeros(N)
tcur = 1.0
while tcur < DUR + XFADE - 2:
    c = clips[rng.integers(len(clips))]
    if rng.random() < 0.5:
        c = c[::-1]                       # reversed speech keeps cadence, loses words
    c = lp(c, 900)
    c = signal.fftconvolve(c, ir)[: len(c) + ir_len // 2]
    add(walla, norm(c), tcur, rng.uniform(0.35, 1.0))
    # sometimes a second voice overlaps, sometimes a quiet gap
    tcur += rng.choice([rng.uniform(0.5, 2.0), rng.uniform(2.5, 6.0), rng.uniform(8, 15)],
                       p=[0.45, 0.4, 0.15])
walla = norm(walla)

# 4. Aperiodic micro-sounds
fx = np.zeros(N)


def keystroke():
    n = int(rng.uniform(0.006, 0.014) * SR)
    k = rng.standard_normal(n) * np.exp(-np.linspace(0, 6, n))
    return norm(bp(k, 1500, 3300, 2))


def typing_burst(t0):
    tt = t0
    for _ in range(rng.integers(5, 26)):
        add(fx, keystroke(), tt, rng.uniform(0.25, 0.6))
        tt += rng.uniform(0.07, 0.24)
        if rng.random() < 0.08:
            tt += rng.uniform(0.3, 0.8)   # thinking pause mid-typing


def mouse(t0):
    for k in range(rng.integers(1, 3)):
        add(fx, keystroke() * 0.8, t0 + k * rng.uniform(0.09, 0.15), 0.35)


def paper(t0):
    n = int(rng.uniform(0.4, 1.2) * SR)
    env = np.abs(lp(rng.standard_normal(n), 25, 2))
    env = norm(env) * np.hanning(n)
    add(fx, norm(bp(rng.standard_normal(n), 1200, 3300)) * env, t0, 0.25)


def chair(t0):
    n = int(rng.uniform(0.25, 0.5) * SR)
    tt = np.arange(n) / SR
    f = np.linspace(rng.uniform(550, 700), rng.uniform(800, 1000), n) * (1 + 0.02 * np.sin(2 * np.pi * 18 * tt))
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = (np.sin(ph) + 0.4 * np.sin(2 * ph)) * np.hanning(n)
    add(fx, norm(s), t0, 0.12)


def horn(t0):
    for k in range(rng.integers(1, 3)):
        n = int(rng.uniform(0.15, 0.35) * SR)
        tt = np.arange(n) / SR
        s = np.sign(np.sin(2 * np.pi * 420 * tt)) + np.sign(np.sin(2 * np.pi * 510 * tt))
        s = lp(s, 1200) * np.hanning(n)
        add(fx, norm(s), t0 + k * 0.45, 0.07)


def desk_thud(t0):
    n = int(0.08 * SR)
    s = rng.standard_normal(n) * np.exp(-np.linspace(0, 8, n))
    add(fx, norm(bp(s, 300, 700, 2)), t0, 0.3)


events = [(typing_burst, 0.30), (mouse, 0.22), (paper, 0.15), (desk_thud, 0.13),
          (chair, 0.12), (horn, 0.08)]
fns, probs = zip(*events)
tcur = rng.uniform(2, 5)
while tcur < DUR + XFADE - 3:
    fn = fns[rng.choice(len(fns), p=np.array(probs) / sum(probs))]
    fn(tcur)
    tcur += rng.exponential(7.0) + 1.5   # irregular, never metronomic

# Mix: walla and fx sit well under the steady bed
mix = 0.55 * norm(room) + 0.8 * norm(fan) + 0.45 * walla + 0.5 * norm(fx)

# Telephone band + gentle "AGC" style soft-limit
mix = bp(mix, 300, 3400, 6)
mix = np.tanh(2.0 * norm(mix)) / np.tanh(2.0)

# Seamless loop: crossfade the tail into the head
xs = XFADE * SR
fade = np.linspace(0, 1, xs)
loop = mix[: DUR * SR].copy()
loop[:xs] = loop[:xs] * fade + mix[DUR * SR: DUR * SR + xs] * (1 - fade)

wavfile.write("ambience_v3.wav", SR, (norm(loop) * 0.9 * 32767).astype(np.int16))
print("wrote ambience_v3.wav", len(loop) / SR, "s")
