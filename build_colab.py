#!/usr/bin/env python3
"""Generate wave_lab_colab.ipynb -- a self-contained Google Colab notebook:
play a song, live real-time spectrum, filters, and ML/AI on the audio."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(src):  cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))
def code(src): cells.append(nbf.v4.new_code_cell(src.strip("\n")))

# ---------------------------------------------------------------- title
md(r"""
# 🎵 Wave Lab — Fourier, Live Spectrum, Filters & ML on a Song

Upload a song and this notebook will:

1. **Break it into sine waves** with the Fourier transform (which notes is it made of?)
2. Show a **live, real-time scrolling spectrum** that plays *in sync* with the audio
3. Let you **filter** it (low/high/band-pass) and hear the difference
4. Run **machine learning / AI** on the sound:
   - *Unsupervised*: K-means auto-segments the song into sections
   - *Deep learning*: a PyTorch CNN (trained on synthesized notes) detects the pitch over time

Runtime → **GPU** is nice for the CNN but not required. Run the cells top to bottom.
""")

# ---------------------------------------------------------------- setup
md("## 1 · Setup")
code(r"""
# librosa + soundfile for audio; torch is preinstalled on Colab.
!pip -q install librosa soundfile

import numpy as np, IPython.display as ipd
import matplotlib.pyplot as plt
import librosa, librosa.display
from scipy import signal as sps
print("ready ✓")
""")

# ---------------------------------------------------------------- load
md(r"""
## 2 · Load a song

Run the cell and **upload a file** (WAV / MP3 / M4A / FLAC all work — librosa
decodes them). If you skip the upload, the notebook synthesizes a little melody
so everything still runs.
""")
code(r"""
from google.colab import files

SR = 22050          # sample rate we resample everything to
MAX_SECONDS = 30    # keep it short so animations render quickly

def synth_melody(sr=SR, seconds=8):
    # C-E-G-C major arpeggio so the ML/Fourier parts have clear notes
    notes = [261.63, 329.63, 392.00, 523.25] * 2
    y = np.array([], dtype=np.float32)
    for f in notes:
        t = np.arange(int(sr*0.5))/sr
        env = np.exp(-3*t)                      # plucky decay
        tone = sum((0.6**k)*np.sin(2*np.pi*f*(k+1)*t) for k in range(4))
        y = np.concatenate([y, (env*tone).astype(np.float32)])
    return y/np.max(np.abs(y)), sr

try:
    up = files.upload()
    path = list(up.keys())[0]
    y, sr = librosa.load(path, sr=SR, mono=True, duration=MAX_SECONDS)
    name = path
except Exception as e:
    print("No upload — using a synthesized melody.", e)
    y, sr = synth_melody(); name = "synth_melody"

y = y/ (np.max(np.abs(y)) + 1e-9)
print(f"Loaded '{name}'  →  {len(y)/sr:.1f}s @ {sr} Hz")
ipd.display(ipd.Audio(y, rate=sr))
""")

# ---------------------------------------------------------------- fourier
md(r"""
## 3 · The sine-wave ingredients (Fourier transform)

The Fourier transform rewrites the sound as a sum of pure sine waves. The tallest
peaks are the dominant notes. We label each with its nearest musical note.
""")
code(r"""
def note_name(f):
    if f <= 0: return ""
    names=["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    m=int(round(69+12*np.log2(f/440.0))); return f"{names[m%12]}{m//12-1}"

def top_sines(y, sr, n=8, fmin=20, fmax=None):
    w=np.hanning(len(y)); Y=np.fft.rfft(y*w)
    f=np.fft.rfftfreq(len(y),1/sr); amp=np.abs(Y)*2/np.sum(w)
    fmax=fmax or sr/2; band=(f>=fmin)&(f<=fmax)
    pk=[i for i in range(1,len(amp)-1) if band[i] and amp[i]>amp[i-1] and amp[i]>=amp[i+1]]
    pk.sort(key=lambda i:amp[i], reverse=True)
    return f, amp, [(f[i],amp[i]) for i in pk[:n]]

f, amp, peaks = top_sines(y, sr, n=8)
print(f"{'freq(Hz)':>9} {'note':>5} {'amp':>8}")
for fr,a in peaks: print(f"{fr:>9.1f} {note_name(fr):>5} {a:>8.4f}")

plt.figure(figsize=(12,4))
plt.plot(f, amp, lw=.7, color="#444")
for fr,a in peaks:
    plt.plot(fr,a,"o",color="#d62728"); plt.annotate(f"{fr:.0f}",(fr,a),
             textcoords="offset points",xytext=(3,3),fontsize=8)
plt.xlim(0, min(sr/2, peaks[0][0]*4+300) if peaks else sr/2)
plt.title("Spectrum — every sine wave (red = strongest)")
plt.xlabel("frequency (Hz)"); plt.ylabel("amplitude"); plt.show()
""")

# ---------------------------------------------------------------- live spectrum
md(r"""
## 4 · Live, real-time scrolling spectrum 🎬

This renders a video where a window slides through the song; for each moment we
compute the FFT and draw the instantaneous spectrum + a scrolling spectrogram.
We then **mux the original audio onto the video with ffmpeg**, so the spectrum
plays *in sync* with the sound. (First render takes ~30–60 s.)
""")
code(r"""
from matplotlib import animation
import soundfile as sf, subprocess, os

FPS  = 20
WIN  = 2048                      # FFT window length (samples)
hop  = max(1, int(sr/FPS))       # advance per video frame
frames = max(1, (len(y)-WIN)//hop)
freqs = np.fft.rfftfreq(WIN, 1/sr)
fmax_show = min(sr/2, 5000); kmax = np.searchsorted(freqs, fmax_show)

# pre-compute the spectrogram strip the running line scrolls over
S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=WIN, hop_length=hop))+1e-6)
S = S[:kmax]

fig,(axT,axS)=plt.subplots(2,1,figsize=(10,6),gridspec_kw={"height_ratios":[1,1.3]})
win=np.hanning(WIN)
(line,)=axT.plot(freqs[:kmax], np.zeros(kmax), color="#1f77b4")
axT.set(xlim=(0,fmax_show), ylim=(0,0.05), xlabel="frequency (Hz)",
        ylabel="amplitude", title="Instantaneous spectrum")
axS.imshow(S, origin="lower", aspect="auto", cmap="magma",
           extent=[0,len(y)/sr,0,fmax_show])
cursor=axS.axvline(0,color="cyan",lw=1.5)
axS.set(xlabel="time (s)", ylabel="frequency (Hz)", title="Spectrogram (cyan = now)")
fig.tight_layout()

def upd(i):
    s=i*hop; seg=y[s:s+WIN]*win
    sp=np.abs(np.fft.rfft(seg))[:kmax]*2/np.sum(win)
    line.set_ydata(sp); axT.set_ylim(0, max(0.02, sp.max()*1.2))
    cursor.set_xdata([s/sr,s/sr]); return line,cursor

anim=animation.FuncAnimation(fig, upd, frames=frames, interval=1000/FPS, blit=False)
anim.save("spectrum.mp4", writer="ffmpeg", fps=FPS, dpi=110)
plt.close(fig)

# mux the song's audio onto the silent animation -> synced playback
sf.write("audio.wav", y, sr)
subprocess.run(["ffmpeg","-y","-i","spectrum.mp4","-i","audio.wav",
                "-c:v","copy","-c:a","aac","-shortest","synced.mp4"],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("done — press play (spectrum is synced to the audio):")
ipd.display(ipd.Video("synced.mp4", embed=True, width=720))
""")

# ---------------------------------------------------------------- filters
md(r"""
## 5 · Filters — keep only the frequencies you want

A filter removes part of the spectrum. Use the slider-style arguments below:
- **low-pass**: keep bass, cut treble (muffled)
- **high-pass**: keep treble, cut bass (tinny)
- **band-pass**: keep a middle band (telephone-like)

We use a zero-phase Butterworth filter (`sosfiltfilt`) so nothing gets time-shifted.
""")
code(r"""
def make_filter(kind, cutoff, sr, order=6):
    nyq=sr/2
    if kind=="band":
        sos=sps.butter(order,[cutoff[0]/nyq,cutoff[1]/nyq],btype="band",output="sos")
    else:
        sos=sps.butter(order,cutoff/nyq,btype=kind,output="sos")
    return sos

def apply_and_show(kind, cutoff):
    sos=make_filter(kind,cutoff,sr)
    yf=sps.sosfiltfilt(sos,y).astype(np.float32)
    yf=yf/(np.max(np.abs(yf))+1e-9)
    f,a0,_=top_sines(y,sr,n=1); _,a1,_=top_sines(yf,sr,n=1)
    plt.figure(figsize=(12,3.5))
    plt.semilogy(f,a0+1e-6,lw=.7,label="original",color="#888")
    plt.semilogy(f,a1+1e-6,lw=.9,label=f"{kind}-pass {cutoff}",color="#d62728")
    plt.xlim(0,min(sr/2,8000)); plt.legend(); plt.xlabel("frequency (Hz)")
    plt.title(f"{kind}-pass filter @ {cutoff} Hz"); plt.show()
    print("original:"); ipd.display(ipd.Audio(y,rate=sr))
    print(f"{kind}-pass {cutoff} Hz:"); ipd.display(ipd.Audio(yf,rate=sr))
    return yf

# ▼▼ edit these and re-run ▼▼
_=apply_and_show("low",  np.array(800))          # low-pass at 800 Hz
_=apply_and_show("high", np.array(2000))         # high-pass at 2 kHz
_=apply_and_show("band", (300,3000))             # 300–3000 Hz band
""")

# ---------------------------------------------------------------- ML intro
md(r"""
## 6 · Machine Learning / AI on the audio

The front-end every audio-ML model uses is the **mel-spectrogram** (a Fourier
spectrogram warped to how humans hear pitch) and its compressed cousin the
**MFCCs**. We build those, then do two ML tasks.
""")
code(r"""
n_fft, hop = 2048, 512
mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=64)
mel_db = librosa.power_to_db(mel, ref=np.max)
mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)

fig,ax=plt.subplots(1,2,figsize=(13,4))
librosa.display.specshow(mel_db, sr=sr, hop_length=hop, x_axis="time",
                         y_axis="mel", ax=ax[0], cmap="magma")
ax[0].set_title("Mel-spectrogram (the ML input)")
librosa.display.specshow(mfcc, sr=sr, hop_length=hop, x_axis="time", ax=ax[1])
ax[1].set_title("MFCCs (compressed timbre features)")
plt.tight_layout(); plt.show()
print("feature shapes:", mel_db.shape, mfcc.shape)
""")

# ---------------------------------------------------------------- unsupervised
md(r"""
### 6a · Unsupervised — auto-segment the song with K-means

No labels needed. We cluster each time-frame by its MFCC fingerprint; frames that
*sound alike* land in the same cluster. The colored timeline reveals structure
(intro / verse / chorus-style sections) the algorithm found on its own.
""")
code(r"""
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

K = 4
X = StandardScaler().fit_transform(mfcc.T)        # frames × features
labels = KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(X)
times = librosa.frames_to_time(np.arange(len(labels)), sr=sr, hop_length=hop)

fig,ax=plt.subplots(2,1,figsize=(12,5),gridspec_kw={"height_ratios":[1,2]})
ax[0].scatter(times, np.zeros_like(times), c=labels, cmap="tab10", marker="|", s=400)
ax[0].set(title=f"Song auto-segmented into {K} sections (K-means on MFCCs)",
          yticks=[], xlabel="time (s)")
p2 = PCA(2).fit_transform(X)
sc=ax[1].scatter(p2[:,0],p2[:,1],c=labels,cmap="tab10",s=8)
ax[1].set(title="Frames in 2-D (PCA) — clusters = similar-sounding moments",
          xlabel="PC1", ylabel="PC2")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- supervised CNN
md(r"""
### 6b · Deep learning — a CNN that hears pitch 🧠

We **synthesize a labelled training set** of musical notes (each = fundamental +
harmonics + decay + noise), turn each into a tiny mel-spectrogram patch, and train
a small **PyTorch CNN** to classify the pitch class (C, C#, … B — 12 classes).
No external dataset required. Then we slide the trained network across the uploaded
song to read out a **predicted-note timeline**.
""")
code(r"""
import torch, torch.nn as nn
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

NOTES, PATCH = 12, 32          # 12 pitch classes, 32-frame mel patches
def synth_note(midi, sr=SR, dur=0.4):
    f=440*2**((midi-69)/12); t=np.arange(int(sr*dur))/sr
    env=np.exp(-np.random.uniform(2,6)*t)
    y=sum((np.random.uniform(.3,.8)**k)*np.sin(2*np.pi*f*(k+1)*t) for k in range(np.random.randint(2,6)))
    y=env*y + 0.01*np.random.randn(len(t))
    return (y/ (np.max(np.abs(y))+1e-9)).astype(np.float32)

def to_patch(sig):
    m=librosa.feature.melspectrogram(y=sig,sr=SR,n_fft=1024,hop_length=256,n_mels=32)
    m=librosa.power_to_db(m,ref=np.max)
    if m.shape[1]<PATCH: m=np.pad(m,((0,0),(0,PATCH-m.shape[1])))
    return ((m[:, :PATCH]+80)/80).astype(np.float32)   # → roughly 0..1

def build_set(n_per=120, midi_lo=48, midi_hi=83):
    Xs,ys=[],[]
    for _ in range(n_per):
        for midi in range(midi_lo,midi_hi+1):
            Xs.append(to_patch(synth_note(midi))); ys.append(midi%12)
    return np.stack(Xs)[:,None], np.array(ys)

print("synthesizing training data…")
Xtr,ytr=build_set()
Xtr=torch.tensor(Xtr); ytr=torch.tensor(ytr)
perm=torch.randperm(len(Xtr)); Xtr,ytr=Xtr[perm],ytr[perm]
n_val=len(Xtr)//5
Xv,yv=Xtr[:n_val].to(device),ytr[:n_val].to(device)
Xt,yt=Xtr[n_val:].to(device),ytr[n_val:].to(device)

class NoteCNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.c=nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1))
        s.f=nn.Sequential(nn.Flatten(),nn.Linear(32,NOTES))
    def forward(s,x): return s.f(s.c(x))

net=NoteCNN().to(device)
opt=torch.optim.Adam(net.parameters(),1e-3); lossf=nn.CrossEntropyLoss()
for ep in range(12):
    net.train(); idx=torch.randperm(len(Xt))
    for b in range(0,len(Xt),128):
        j=idx[b:b+128]; opt.zero_grad()
        l=lossf(net(Xt[j]),yt[j]); l.backward(); opt.step()
    net.eval()
    with torch.no_grad(): acc=(net(Xv).argmax(1)==yv).float().mean().item()
    print(f"epoch {ep+1:2d}  val-acc {acc*100:5.1f}%")
""")

md(r"""
Now slide the trained CNN over the **uploaded song** and plot what note it thinks
it hears at each moment, against the energy of the audio.
""")
code(r"""
names=["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
hop_s=0.1; wlen=int(SR*0.4); step=int(SR*hop_s)
preds, tt = [], []
net.eval()
for s in range(0, len(y)-wlen, step):
    patch=to_patch(y[s:s+wlen])
    with torch.no_grad():
        p=net(torch.tensor(patch)[None,None].to(device)).argmax(1).item()
    preds.append(p); tt.append(s/SR)

rms=librosa.feature.rms(y=y, hop_length=hop)[0]
rt=librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

fig,ax=plt.subplots(2,1,figsize=(12,5),sharex=True,gridspec_kw={"height_ratios":[2,1]})
ax[0].scatter(tt,preds,c=preds,cmap="hsv",s=18)
ax[0].set(yticks=range(12),yticklabels=names,ylabel="predicted pitch class",
          title="CNN's predicted note over time (trained only on synthetic notes)")
ax[1].plot(rt,rms,color="#1f77b4"); ax[1].set(xlabel="time (s)",ylabel="loudness (RMS)")
plt.tight_layout(); plt.show()
print("Tip: the CNN is most confident during clear, sustained notes.")
""")

# ---------------------------------------------------------------- recap
md(r"""
## 7 · Recap

You went from raw audio → **Fourier** sine-wave decomposition → a **synced live
spectrum** → **filtering** → **ML**: unsupervised K-means structure discovery and a
**CNN** pitch detector trained on synthetic notes.

**Things to try next**
- Upload different genres and compare the K-means segmentation.
- Train the CNN on the full 88-key range, or predict the *full* note (not just pitch class).
- Swap the synthetic training notes for real instrument samples for a big accuracy jump.
- Add a high-pass filter *before* the CNN to see how preprocessing changes its predictions.
""")

nb["cells"]=cells
nb["metadata"]={"colab":{"provenance":[],"toc_visible":True},
                "kernelspec":{"name":"python3","display_name":"Python 3"},
                "accelerator":"GPU"}
with open("wave_lab_colab.ipynb","w") as fh:
    nbf.write(nb,fh)
print("wrote wave_lab_colab.ipynb with", len(cells), "cells")
