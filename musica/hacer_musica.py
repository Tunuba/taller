# Musica de fondo de la presentacion. Alegre y comica, para que este ahi sin estorbar.
# 32 compases a 120 por minuto (64 s). Lo que suena despues del final se suma al principio,
# asi el bucle no tiene costura y no se nota en que punto va.
# Uso: python hacer_musica.py   (deja ../musica.wav)
import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter

SR = 22050
NEGRA = 0.5                      # 120 por minuto
OCT = NEGRA / 2                  # una corchea
COMPAS = 4 * NEGRA
COMPASES = 32
LARGO = COMPASES * COMPAS
pista = np.zeros(int((LARGO + 4) * SR))
rng = np.random.default_rng(7)

NOTAS = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
def midi(n):
    return 12 * (int(n[-1]) + 1) + NOTAS[n[0]] + (1 if '#' in n else 0)
def hz(m):
    return 440 * 2 ** ((m - 69) / 12)

def poner(t0, x, vol):
    i = int(t0 * SR)
    pista[i:i + len(x)] += vol * x[:len(pista) - i]

def tiempo(seg):
    return np.arange(int(seg * SR)) / SR

# ---- instrumentos ----
def ukulele(f, seg=0.6):
    t = tiempo(seg)
    x = sum(np.sin(2 * np.pi * f * h * t) * np.exp(-t * (6 + 5 * h)) / h for h in range(1, 7))
    return x * np.minimum(1, t / 0.004)

def bajo(f, seg=0.32):
    # un bajo redondo y corto, el oom del oom-pa
    t = tiempo(seg)
    x = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) + 0.12 * np.sin(6 * np.pi * f * t)
    return x * np.exp(-t * 7) * np.minimum(1, t / 0.006)

def campanita(f, seg=1.2):
    t = tiempo(seg)
    x = np.sin(2 * np.pi * f * t) * np.exp(-t * 3.5) + 0.25 * np.sin(2 * np.pi * f * 4.07 * t) * np.exp(-t * 9)
    return x * np.minimum(1, t / 0.002)

def silbido(f, seg):
    # silbato con vibrato y una subidita al empezar, el toque comico
    t = tiempo(seg)
    sube = 1 - 0.06 * np.exp(-t * 30)
    vib = 1 + 0.008 * np.sin(2 * np.pi * 5.5 * t) * np.minimum(1, t / 0.15)
    fase = 2 * np.cumsum(np.pi * f * sube * vib) / SR
    env = np.minimum(1, t / 0.03) * np.minimum(1, (seg - t) / 0.05).clip(0)
    return (np.sin(fase) + 0.05 * rng.standard_normal(len(t)) * 0.3) * env

def ruido(seg, dec, alto=True):
    t = tiempo(seg)
    x = rng.standard_normal(len(t))
    b, a = butter(2, 5000 / (SR / 2), 'high') if alto else butter(2, 1200 / (SR / 2), 'low')
    return lfilter(b, a, x) * np.exp(-t * dec)

def bombo(seg=0.25):
    t = tiempo(seg)
    return np.sin(2 * np.pi * np.cumsum(55 + 90 * np.exp(-t * 30)) / SR) * np.exp(-t * 14)

def madera(f=900, seg=0.12):
    t = tiempo(seg)
    return np.sin(2 * np.pi * f * t) * np.exp(-t * 45)

def deslizar(f0, f1, seg):
    # el silbato que sube, para cerrar las vueltas
    t = tiempo(seg)
    f = f0 * (f1 / f0) ** (t / seg)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.minimum(1, t / 0.02) * np.minimum(1, (seg - t) / 0.04)

# ---- armonia ----
ACORDES = {'C': ['C4', 'E4', 'G4'], 'G': ['G3', 'B3', 'D4'], 'Am': ['A3', 'C4', 'E4'],
           'F': ['F3', 'A3', 'C4'], 'Em': ['E3', 'G3', 'B3']}
RAIZ = {'C': ('C2', 'G2'), 'G': ('G2', 'D2'), 'Am': ('A2', 'E2'), 'F': ('F2', 'C3'), 'Em': ('E2', 'B2')}
PARTE_A = ['C', 'G', 'Am', 'F', 'C', 'G', 'F', 'G']
PARTE_B = ['F', 'G', 'Em', 'Am', 'F', 'G', 'C', 'G']

# melodias en corcheas: (nota o None, cuantas corcheas)
MEL_A = [('G4', 1), ('C5', 1), ('E5', 2), ('G5', 2), ('E5', 2),
         ('D5', 2), ('B4', 1), ('D5', 1), ('G5', 4),
         ('A4', 1), ('C5', 1), ('E5', 2), ('A5', 2), ('G5', 2),
         ('F5', 2), ('E5', 2), ('D5', 2), ('C5', 2),
         ('E5', 1), ('E5', 1), (None, 1), ('E5', 1), ('G5', 2), ('E5', 2),
         ('D5', 1), ('D5', 1), (None, 1), ('D5', 1), ('B4', 2), ('G4', 2),
         ('A4', 2), ('C5', 2), ('F5', 2), ('A5', 2),
         ('G5', 3), (None, 1), ('F5', 1), ('E5', 1), ('D5', 2)]
MEL_B = [('C6', 2), ('A5', 2), ('F5', 2), ('A5', 2),
         ('B5', 2), ('G5', 2), ('D5', 2), ('G5', 2),
         ('E5', 1), ('G5', 1), ('B5', 2), ('G5', 2), ('E5', 2),
         ('A5', 4), (None, 2), ('E5', 1), ('G5', 1),
         ('A5', 1), ('G5', 1), ('F5', 1), ('E5', 1), ('F5', 2), ('A5', 2),
         ('G5', 1), ('F5', 1), ('E5', 1), ('D5', 1), ('B4', 2), ('D5', 2),
         ('C5', 2), ('E5', 2), ('G5', 2), ('C6', 2),
         ('B5', 2), (None, 2), ('G5', 1), ('A5', 1), ('B5', 2)]
assert sum(d for _, d in MEL_A) == 64 and sum(d for _, d in MEL_B) == 64

def melodia(t0, mel, voz):
    t = t0
    for n, d in mel:
        if n:
            f = hz(midi(n))
            if voz == 'campanita':
                poner(t, campanita(f), 0.30)
            elif voz == 'silbido':
                poner(t, silbido(f, d * OCT * 0.85), 0.20)
            else:                                   # las dos juntas, la campanita una octava arriba
                poner(t, silbido(f, d * OCT * 0.85), 0.16)
                poner(t, campanita(f * 2), 0.12)
        t += d * OCT

def acompanar(t0, acordes, lleno):
    for c, nombre in enumerate(acordes):
        tb = t0 + c * COMPAS
        r, q = RAIZ[nombre]
        for b in range(4):                          # oom-pa, raiz y quinta, con el rasgueo a contratiempo
            tt = tb + b * NEGRA
            if b % 2 == 0:
                poner(tt, bajo(hz(midi(r if b == 0 else q))), 0.50)
                if lleno: poner(tt, bombo(), 0.35)
            for k, n in enumerate(ACORDES[nombre]):
                poner(tt + OCT + k * 0.012, ukulele(hz(midi(n))), 0.11)
            if lleno:
                poner(tt, ruido(0.06, 60), 0.05)    # maraca en cada corchea
                poner(tt + OCT, ruido(0.06, 60), 0.035)
            if b == 3 and c % 2 == 1:
                poner(tt + OCT, madera(), 0.18)    # el toc de madera, comico, cada dos compases
        if b == 3 and c == 3 and lleno:
            poner(tb + 3 * NEGRA, ruido(0.12, 30, alto=False), 0.12)   # aplauso a mitad de frase

# A, B, A, B, cada vuelta con otra voz para que no canse
vueltas = [(PARTE_A, MEL_A, 'campanita', False), (PARTE_B, MEL_B, 'campanita', True),
           (PARTE_A, MEL_A, 'silbido', True), (PARTE_B, MEL_B, 'las dos', True)]
for v, (acordes, mel, voz, lleno) in enumerate(vueltas):
    t0 = v * 8 * COMPAS
    acompanar(t0, acordes, lleno)
    melodia(t0, mel, voz)
    if v in (1, 3):                                 # silbato que sube al cerrar cada A-B
        poner(t0 + 8 * COMPAS - NEGRA, deslizar(hz(midi('G5')), hz(midi('G6')), 0.4), 0.07)

# un eco corto y suave para que no suene seco
eco = np.zeros_like(pista)
d = int(0.375 * SR)
eco[d:] = pista[:-d] * 0.18
pista += eco

# el bucle, lo que pasa del final se suma al principio
n = int(LARGO * SR)
salida = pista[:n].copy()
salida[:len(pista) - n] += pista[n:]
salida = salida / np.max(np.abs(salida)) * 0.8

aqui = os.path.dirname(os.path.abspath(__file__))
wavfile.write(os.path.join(aqui, '..', 'musica.wav'), SR, (salida * 32767).astype(np.int16))
print('musica.wav', round(LARGO, 1), 's')
