#!/usr/bin/env python3
"""
wave_lab.py  --  Play a song, record it through the mic, and pull it apart
                 into its sine-wave ingredients with the Fourier transform.
                 Also shows the Laplace transform (the s-plane) so you can
                 see how Fourier is just one slice of it.

Modes
-----
  demo        Synthesize a chord (known sines), analyze it. No song/mic needed.
              Great first test -- the analysis should recover the exact notes.

  record      Record from the microphone for N seconds, save to a WAV.

  play        Play a song file AND record the mic at the same time, save the
              recording (this is the "play the song and record it" mode).

  analyze     Take any WAV file and do the full Fourier breakdown:
                - time-domain waveform
                - magnitude spectrum (every sine wave, with peaks labelled)
                - the top-N dominant sine waves drawn individually
                - a reconstruction = sum of those sines, vs the original
                - a spectrogram (how the sines change over time)

  laplace     Laplace s-plane view of a short clip: |X(s)| over s = sigma + j*omega.
              The vertical sigma = 0 line is exactly the Fourier transform.

Examples
--------
  python3 wave_lab.py demo
  python3 wave_lab.py record --seconds 5 --out mic.wav
  python3 wave_lab.py play --song mysong.wav --out recorded.wav
  python3 wave_lab.py analyze --in recorded.wav --peaks 8
  python3 wave_lab.py laplace --in recorded.wav

Notes
-----
* On macOS the first time you record, the OS will ask the *terminal app*
  (Terminal / iTerm / VS Code) for microphone permission. Allow it, then rerun.
* Song files: WAV works out of the box. MP3/M4A also work if ffmpeg is
  installed (brew install ffmpeg) -- otherwise convert to WAV first.
"""

import argparse
import sys
import numpy as np

# Audio I/O libs are only needed for the record/play modes, so import lazily
# and give a friendly message if missing.
def _need_audio():
    try:
        import sounddevice as sd  # noqa
        import soundfile as sf    # noqa
        return sd, sf
    except ImportError:
        sys.exit("Missing audio libs. Install with:\n"
                 "    python3 -m pip install sounddevice soundfile")


# ----------------------------------------------------------------------------
# Loading / saving audio
# ----------------------------------------------------------------------------
def load_audio(path):
    """Return (signal, samplerate). Signal is mono float in [-1, 1]."""
    _, sf = _need_audio()
    data, sr = sf.read(path, always_2d=True)
    mono = data.mean(axis=1)            # mix stereo down to mono
    return mono.astype(np.float64), sr


def save_audio(path, signal, sr):
    _, sf = _need_audio()
    sf.write(path, signal, sr)
    print(f"  saved -> {path}  ({len(signal)/sr:.1f}s @ {sr} Hz)")


# ----------------------------------------------------------------------------
# The Fourier breakdown -- the heart of the tool
# ----------------------------------------------------------------------------
def trim_silence(signal, sr, thresh=0.05, pad=0.03):
    """Cut leading/trailing quiet (e.g. the countdown before the song starts)."""
    env = np.abs(signal)
    loud = np.where(env > thresh * env.max())[0]
    if len(loud) == 0:
        return signal
    a = max(0, loud[0] - int(pad * sr))
    b = min(len(signal), loud[-1] + int(pad * sr))
    return signal[a:b]


def loudest_window(signal, sr, dur):
    """Index of the start of the loudest `dur`-second window (the busy bit)."""
    w = int(sr * dur)
    if len(signal) <= w:
        return 0
    best_i, best_rms = 0, -1.0
    for s in range(0, len(signal) - w, max(1, w // 2)):
        r = np.sqrt(np.mean(signal[s:s + w] ** 2))
        if r > best_rms:
            best_rms, best_i = r, s
    return best_i


def fourier_components(signal, sr, n_peaks=6, fmin=20.0, fmax=None,
                       min_sep_hz=25.0, rel_floor=0.05):
    """
    Run an FFT and pick out the strongest sine waves.

    `min_sep_hz` forces the chosen peaks to be spread out in frequency, so we
    don't return eight near-identical bins from one bass note (which is what
    made the labels overlap).

    Returns a list of dicts: {freq, amp, phase} sorted strongest-first.
    Each one means:  amp * cos(2*pi*freq*t + phase)
    Add them all up and you approximate the original signal.
    """
    n = len(signal)
    # A window tapers the clip's edges so the FFT doesn't smear energy
    # across frequencies ("spectral leakage"). Hann is the friendly default.
    window = np.hanning(n)
    windowed = signal * window

    # rfft: real-input FFT -> only the non-negative frequencies (the rest are
    # mirror images for a real signal, so we'd be double counting).
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    # Convert complex FFT bins to real amplitude & phase of cosine waves.
    # The window removed energy, and rfft splits each real cosine across the
    # bin -- the (2/sum(window)) factor undoes both so amp is the true height.
    amps = np.abs(spectrum) * (2.0 / np.sum(window))
    phases = np.angle(spectrum)

    if fmax is None:
        fmax = sr / 2.0  # Nyquist: the highest frequency we can represent

    band = (freqs >= fmin) & (freqs <= fmax)

    # Find local peaks in the spectrum (a bin taller than both neighbours),
    # so two adjacent bins of one note don't both get reported.
    peaks = []
    a = amps
    for i in range(1, len(a) - 1):
        if band[i] and a[i] > a[i - 1] and a[i] >= a[i + 1]:
            peaks.append(i)
    if not peaks:                      # fallback: just take the loudest bins
        peaks = list(np.where(band)[0])

    peaks.sort(key=lambda i: a[i], reverse=True)

    # ignore tiny peaks (noise bins) far below the loudest one -- this stops a
    # stray high-frequency hiss from being reported and stretching the x-axis.
    floor = a[peaks[0]] * rel_floor if peaks else 0.0

    # greedily take the loudest peaks, but skip any too close to one already
    # chosen -> the reported sines are spread across the spectrum.
    chosen = []
    for i in peaks:
        if a[i] < floor:
            continue
        if all(abs(freqs[i] - freqs[j]) >= min_sep_hz for j in chosen):
            chosen.append(i)
        if len(chosen) >= n_peaks:
            break

    return [{"freq": float(freqs[i]),
             "amp": float(amps[i]),
             "phase": float(phases[i])} for i in chosen], freqs, amps


def pick_peaks(amps, freqs, n, min_sep_hz=40.0, rel_floor=0.15,
               fmin=40.0, fmax=None):
    """Indices of up to `n` strongest, well-separated spectral peaks.

    Shared by the static analysis and the live display so both stay clean:
    drops quiet noise bins (rel_floor) and forces frequency spacing (min_sep_hz).
    """
    fmax = fmax or freqs[-1]
    band = (freqs >= fmin) & (freqs <= fmax)
    cand = [i for i in range(1, len(amps) - 1)
            if band[i] and amps[i] > amps[i - 1] and amps[i] >= amps[i + 1]]
    if not cand:
        return []
    cand.sort(key=lambda i: amps[i], reverse=True)
    floor = amps[cand[0]] * rel_floor
    chosen = []
    for i in cand:
        if amps[i] < floor:
            break
        if all(abs(freqs[i] - freqs[j]) >= min_sep_hz for j in chosen):
            chosen.append(i)
        if len(chosen) >= n:
            break
    return chosen


def reconstruct(components, t):
    """Add the chosen sine waves back together over time axis t."""
    out = np.zeros_like(t)
    for c in components:
        out += c["amp"] * np.cos(2 * np.pi * c["freq"] * t + c["phase"])
    return out


def note_name(freq):
    """Nearest musical note for a frequency (just for nice labels)."""
    if freq <= 0:
        return ""
    names = ["C", "C#", "D", "D#", "E", "F",
             "F#", "G", "G#", "A", "A#", "B"]
    midi = round(69 + 12 * np.log2(freq / 440.0))
    return f"{names[midi % 12]}{midi // 12 - 1}"


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def analyze_and_plot(signal, sr, n_peaks=6, title="audio"):
    import matplotlib.pyplot as plt

    full = np.arange(len(signal)) / sr            # time axis of the full clip

    # Drop the silent countdown, then analyze the loudest 200 ms window so the
    # spectrum is clean and the sines actually match the sound there.
    trimmed = trim_silence(signal, sr)
    seg_dur = min(0.2, len(trimmed) / sr)
    s0 = loudest_window(trimmed, sr, seg_dur)
    snippet = trimmed[s0:s0 + int(sr * seg_dur)]

    comps, freqs, amps = fourier_components(snippet, sr, n_peaks=n_peaks)
    tsnip = np.arange(len(snippet)) / sr          # snippet-relative time

    print(f"\nTop {len(comps)} sine waves (loudest 200 ms of '{title}'):")
    print(f"  {'#':>2}  {'freq (Hz)':>10}  {'note':>5}  {'amplitude':>10}")
    for k, c in enumerate(comps, 1):
        print(f"  {k:>2}  {c['freq']:>10.1f}  {note_name(c['freq']):>5}  "
              f"{c['amp']:>10.4f}")

    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Fourier breakdown of '{title}'", fontsize=14, weight="bold")

    # (1) full waveform, with the analyzed snippet highlighted
    ax[0, 0].plot(full, signal, lw=0.5, color="#1f77b4")
    off = (len(signal) - len(trimmed))            # samples trimmed at the front
    a_t = (off + s0) / sr
    ax[0, 0].axvspan(a_t, a_t + seg_dur, color="orange", alpha=0.35,
                     label="analyzed snippet")
    ax[0, 0].set(title="1. Recorded waveform (orange = the bit we analyze)",
                 xlabel="time (s)", ylabel="amplitude")
    ax[0, 0].legend(fontsize=8, loc="upper right")

    # (2) spectrum of the snippet, peaks labelled with staggered, non-overlapping text
    fmax_show = min(sr / 2, max((c["freq"] for c in comps), default=1000) * 2.5 + 200)
    ax[0, 1].plot(freqs, amps, lw=0.8, color="#888")
    ymax = max((c["amp"] for c in comps), default=1.0)
    for rank, c in enumerate(sorted(comps, key=lambda c: c["freq"])):
        ax[0, 1].plot(c["freq"], c["amp"], "o", color="#d62728", ms=5)
        ytext = ymax * (1.18 + 0.12 * (rank % 3))   # stagger height to avoid overlap
        ax[0, 1].annotate(f"{c['freq']:.0f} Hz\n{note_name(c['freq'])}",
                          (c["freq"], c["amp"]), xytext=(c["freq"], ytext),
                          ha="center", fontsize=8,
                          arrowprops=dict(arrowstyle="-", lw=0.5, color="#d62728"))
    ax[0, 1].set(title="2. Spectrum of that snippet (labelled peaks = the sine waves)",
                 xlabel="frequency (Hz)", ylabel="amplitude",
                 xlim=(0, fmax_show), ylim=(0, ymax * 1.7))

    # (3) the individual dominant sine waves, stacked with vertical offsets
    show = tsnip[tsnip <= min(tsnip[-1], 0.05)]   # ~50 ms so waves are visible
    step = (ymax * 2.5) if comps else 1.0
    for k, c in enumerate(comps):
        y = c["amp"] * np.cos(2 * np.pi * c["freq"] * show + c["phase"])
        ax[1, 0].plot(show * 1000, y - k * step, lw=1.2,
                      label=f"{c['freq']:.0f} Hz ({note_name(c['freq'])})")
        ax[1, 0].text(0, -k * step + step * 0.25, f"{c['freq']:.0f} Hz",
                      fontsize=7, color="gray")
    ax[1, 0].set(title="3. The dominant sine waves, one by one",
                 xlabel="time (ms)", yticks=[])
    ax[1, 0].legend(fontsize=7, loc="upper right", ncol=1)

    # (4) reconstruction vs the original snippet (same window -> amplitudes match)
    recon = reconstruct(comps, show)
    ax[1, 1].plot(show * 1000, snippet[:len(show)], lw=1.4,
                  color="#1f77b4", label="original snippet")
    ax[1, 1].plot(show * 1000, recon, lw=1.4, ls="--",
                  color="#d62728", label=f"sum of {len(comps)} sines")
    ax[1, 1].set(title="4. Those sines added back up vs the real sound",
                 xlabel="time (ms)", ylabel="amplitude")
    ax[1, 1].legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # spectrogram in its own window: how the sine content evolves over time
    fig2, axs = plt.subplots(figsize=(11, 5))
    axs.specgram(trimmed, NFFT=2048, Fs=sr, noverlap=1024, cmap="magma")
    axs.set(title=f"Spectrogram of '{title}'  (sines over time)",
            xlabel="time (s)", ylabel="frequency (Hz)",
            ylim=(0, min(sr / 2, 5000)))

    print("\nShowing plots -- close the windows to exit.")
    plt.show()


def laplace_plot(signal, sr, max_ms=40.0):
    """
    Numerically evaluate the (unilateral) Laplace transform
        X(s) = integral_0^T x(t) e^{-s t} dt,   s = sigma + j*omega
    over a grid of s and show |X(s)| as a surface in the s-plane.

    The line sigma = 0 is exactly the Fourier transform -- this makes the
    "Fourier is a slice of Laplace" idea visible.
    """
    import matplotlib.pyplot as plt

    # Use only a short clip -- the Laplace integral over a long sustained
    # signal blows up and isn't illuminating. A transient is the point.
    clip = signal[:int(sr * max_ms / 1000.0)]
    t = np.arange(len(clip)) / sr

    # Frequency axis (omega) and a small range of damping (sigma).
    f = np.linspace(1, min(sr / 2, 3000), 240)
    omega = 2 * np.pi * f
    sigma = np.linspace(-300, 300, 200)     # sigma=0 in the middle = Fourier

    S, W = np.meshgrid(sigma, omega)
    s = S + 1j * W
    # X(s) = sum_n x[n] e^{-s t_n} dt   (rectangular-rule integral)
    dt = 1.0 / sr
    # shape: (len(omega), len(sigma)); broadcast over time with einsum
    expo = np.exp(-np.einsum("ij,k->ijk", s, t))
    X = np.einsum("ijk,k->ij", expo, clip) * dt
    mag = np.log10(np.abs(X) + 1e-9)

    fig, ax = plt.subplots(figsize=(11, 6))
    pcm = ax.pcolormesh(sigma, f, mag, shading="auto", cmap="viridis")
    ax.axvline(0, color="white", ls="--", lw=1.5)
    ax.text(5, f[-1] * 0.95, "sigma = 0  ->  this line IS the Fourier transform",
            color="white", fontsize=9, va="top")
    ax.set(title="Laplace transform  |X(s)|  in the s-plane  (log magnitude)",
           xlabel="sigma  (real part of s -- damping)",
           ylabel="frequency  f = omega / 2pi  (Hz)")
    fig.colorbar(pcm, ax=ax, label="log10 |X(s)|")
    fig.tight_layout()
    print("Showing s-plane. The dashed line is the Fourier slice. Close to exit.")
    plt.show()


# ----------------------------------------------------------------------------
# Recording / playback
# ----------------------------------------------------------------------------
def list_devices():
    sd, _ = _need_audio()
    print("Microphones this computer can record from:")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            mark = "  <- current default" if i == sd.default.device[0] else ""
            print(f"  [{i}] {d['name']}  ({d['max_input_channels']} ch){mark}")
    print("\nTip: to record the song from your phone's speaker, pick the "
          "'MacBook Pro Microphone', e.g.  --device 3")


def record(seconds, sr, out, device=None):
    sd, _ = _need_audio()
    if device is not None:
        name = sd.query_devices(device)["name"]
        print(f"Using mic [{device}] {name}")
    print(f"Recording {seconds:.0f}s at {sr} Hz... start the song on your phone now!")
    # countdown so you have time to hit play on the phone
    import time
    for n in (3, 2, 1):
        print(f"  ...{n}")
        time.sleep(1)
    print("  RECORDING")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1,
                   dtype="float64", device=device)
    sd.wait()
    print("  done.")
    audio = audio.flatten()
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-3:
        print("  ⚠ Recording is almost silent — check mic permission "
              "(System Settings → Privacy & Security → Microphone) "
              "and that the phone is loud / close to the mic.")
    else:
        audio = audio / peak       # normalize so the plots/analysis are clear
    save_audio(out, audio, sr)
    return audio, sr


def live(sr=44100, device=None, fmax=5000.0, n_fft=8192):
    """
    Real-time display: opens a window that updates ~30x/second straight from
    the mic. Start a song on your phone and watch its sine waves move.

    Top    : rolling waveform
    Middle : live spectrum, with the strongest frequency + musical note labelled
    Bottom : scrolling spectrogram (how the sines evolve)

    Close the window (or Ctrl-C in the terminal) to stop.
    """
    import queue
    import matplotlib.pyplot as plt
    from matplotlib import animation
    sd, _ = _need_audio()

    block = 1024
    ring = np.zeros(n_fft)                  # newest audio at the right end
    q = queue.Queue()

    def cb(indata, frames, t_info, status):
        q.put(indata[:, 0].copy())

    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    kmax = int(np.searchsorted(freqs, fmax))
    win = np.hanning(n_fft)

    spec_cols = 200                         # spectrogram history width
    spec = np.zeros((kmax, spec_cols))

    fig, (axW, axS, axG) = plt.subplots(
        3, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [1, 1.4, 1.4]})
    fig.suptitle("wave_lab — LIVE  (play a song into the mic; close window to stop)",
                 weight="bold")

    tw = np.arange(n_fft) / sr
    (wave_line,) = axW.plot(tw, ring, lw=0.5, color="#1f77b4")
    axW.set(ylim=(-1, 1), xlim=(0, n_fft / sr), yticks=[],
            title="waveform", xlabel="time (s)")

    (spec_line,) = axS.plot(freqs[:kmax], np.zeros(kmax), color="#d62728", lw=1)
    (peak_dots,) = axS.plot([], [], "o", color="black", ms=6)
    N_LABEL = 5
    peak_texts = [axS.text(0, 0, "", ha="center", va="bottom", fontsize=9,
                           weight="bold", color="#222", visible=False)
                  for _ in range(N_LABEL)]
    note_txt = axS.text(0.98, 0.92, "", transform=axS.transAxes, ha="right",
                        fontsize=13, weight="bold", color="#1a7f37")
    axS.set(xlim=(0, fmax), ylim=(0, 0.05), xlabel="frequency (Hz)",
            ylabel="amplitude", title="live spectrum (top sine waves, labelled)")

    img = axG.imshow(spec, origin="lower", aspect="auto", cmap="magma",
                     extent=[0, spec_cols, 0, fmax], vmin=-80, vmax=0)
    axG.set(xlabel="time →", ylabel="frequency (Hz)",
            title="scrolling spectrogram", xticks=[])
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    def update(_):
        nonlocal ring, spec
        got = False
        while not q.empty():
            b = q.get()
            m = len(b)
            ring = np.roll(ring, -m)
            ring[-m:] = b
            got = True
        if not got:
            return wave_line, spec_line, peak_dot, note_txt, img

        wave_line.set_ydata(ring)
        sp = np.abs(np.fft.rfft(ring * win))[:kmax] * 2 / np.sum(win)
        spec_line.set_ydata(sp)
        top = max(0.02, float(sp.max()) * 1.45)   # headroom for the labels
        axS.set_ylim(0, top)

        # de-cluttered peaks: spread out + above a noise floor (same as static)
        idx = pick_peaks(sp, freqs[:kmax], N_LABEL, min_sep_hz=40,
                         rel_floor=0.2, fmin=40, fmax=fmax)
        if idx:
            peak_dots.set_data(freqs[idx], sp[idx])
            note_txt.set_text(f"{freqs[idx[0]]:.0f} Hz  ({note_name(freqs[idx[0]])})")
        else:
            peak_dots.set_data([], [])
            note_txt.set_text("")
        for j, txt in enumerate(peak_texts):
            if j < len(idx):
                i = idx[j]
                txt.set_position((freqs[i], sp[i] + top * 0.03))
                txt.set_text(f"{freqs[i]:.0f}\n{note_name(freqs[i])}")
                txt.set_visible(True)
            else:
                txt.set_visible(False)

        col = 20 * np.log10(sp + 1e-6)       # dB for the spectrogram
        spec = np.roll(spec, -1, axis=1)
        spec[:, -1] = col
        img.set_data(spec)
        return (wave_line, spec_line, peak_dots, note_txt, img, *peak_texts)

    if device is not None:
        print(f"Using mic [{device}] {sd.query_devices(device)['name']}")
    print("Opening live window... play a song on your phone now. "
          "Close the window to stop.")
    stream = sd.InputStream(channels=1, samplerate=sr, blocksize=block,
                            device=device, callback=cb)
    with stream:
        _ani = animation.FuncAnimation(fig, update, interval=33,
                                       blit=False, cache_frame_data=False)
        plt.show()
    print("stopped.")


def play_and_record(song_path, out):
    sd, _ = _need_audio()
    song, sr = load_audio(song_path)
    dur = len(song) / sr
    print(f"Playing '{song_path}' ({dur:.1f}s) and recording the mic together...")

    rec = sd.rec(len(song), samplerate=sr, channels=1, dtype="float64")
    sd.play(song, sr)
    sd.wait()
    rec = rec.flatten()
    save_audio(out, rec, sr)
    return rec, sr


# ----------------------------------------------------------------------------
# Demo: a synthetic chord so you can test with no song/mic
# ----------------------------------------------------------------------------
def make_demo(sr=44100, seconds=2.0):
    """A-major-ish chord: A4 440, C#5 554.37, E5 659.25 + a little noise."""
    t = np.arange(int(sr * seconds)) / sr
    sig = (0.6 * np.cos(2 * np.pi * 440.00 * t) +
           0.4 * np.cos(2 * np.pi * 554.37 * t + 0.5) +
           0.3 * np.cos(2 * np.pi * 659.25 * t + 1.0))
    sig += 0.02 * np.random.randn(len(t))
    sig /= np.max(np.abs(sig))
    return sig, sr


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Play a song, record it, and break it into sine waves "
                    "(Fourier) + show the Laplace s-plane.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    d = sub.add_parser("demo", help="synthetic chord -> analyze (no mic needed)")
    d.add_argument("--peaks", type=int, default=8)

    sub.add_parser("devices", help="list microphones you can record from")

    lv = sub.add_parser("live", help="real-time spectrum window from the mic")
    lv.add_argument("--device", type=int, default=None,
                    help="mic index from 'devices' (e.g. 3 = MacBook mic)")
    lv.add_argument("--sr", type=int, default=44100)
    lv.add_argument("--fmax", type=float, default=5000.0,
                    help="top frequency to display (Hz)")

    r = sub.add_parser("record", help="record from mic to a WAV")
    r.add_argument("--seconds", type=float, default=15.0)
    r.add_argument("--sr", type=int, default=44100)
    r.add_argument("--out", default="mic.wav")
    r.add_argument("--device", type=int, default=None,
                   help="mic index from 'devices' (e.g. 3 = MacBook mic)")
    r.add_argument("--peaks", type=int, default=8)
    r.add_argument("--no-analyze", action="store_true")

    pl = sub.add_parser("play", help="play a song + record the mic together")
    pl.add_argument("--song", required=True, help="path to song (WAV/MP3/...)")
    pl.add_argument("--out", default="recorded.wav")
    pl.add_argument("--peaks", type=int, default=8)
    pl.add_argument("--no-analyze", action="store_true")

    a = sub.add_parser("analyze", help="Fourier breakdown of a WAV file")
    a.add_argument("--in", dest="infile", required=True)
    a.add_argument("--peaks", type=int, default=8)

    lp = sub.add_parser("laplace", help="Laplace s-plane view of a WAV file")
    lp.add_argument("--in", dest="infile", required=True)

    args = p.parse_args()

    if args.mode == "demo":
        sig, sr = make_demo()
        analyze_and_plot(sig, sr, n_peaks=args.peaks, title="demo chord")

    elif args.mode == "devices":
        list_devices()

    elif args.mode == "live":
        live(sr=args.sr, device=args.device, fmax=args.fmax)

    elif args.mode == "record":
        sig, sr = record(args.seconds, args.sr, args.out, device=args.device)
        if not args.no_analyze:
            analyze_and_plot(sig, sr, n_peaks=args.peaks, title=args.out)

    elif args.mode == "play":
        sig, sr = play_and_record(args.song, args.out)
        if not args.no_analyze:
            analyze_and_plot(sig, sr, n_peaks=args.peaks, title=args.out)

    elif args.mode == "analyze":
        sig, sr = load_audio(args.infile)
        analyze_and_plot(sig, sr, n_peaks=args.peaks, title=args.infile)

    elif args.mode == "laplace":
        sig, sr = load_audio(args.infile)
        laplace_plot(sig, sr)


if __name__ == "__main__":
    main()
