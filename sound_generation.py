import numpy as np
import sounddevice as sd

sd.default.samplerate = 44100

time = 2.0
samplerate = 44100
frequency = 261  # C
frequency2 = 329  # E
frequency3 = 392  # G

wave_size = (samplerate / frequency)
wave_size2 = (samplerate / frequency2)
wave_size3 = (samplerate / frequency3)

sound = np.zeros(int(samplerate * time))
for i in range(len(sound)):
   sound[i] = (np.sin(i / wave_size * 2 * np.pi) + np.sin(i / wave_size2 * 2 * np.pi) + np.sin(i / wave_size3 * 2 * np.pi)) * 32767/3

wav_wave = np.array(sound, dtype=np.int16)
sd.play(wav_wave, blocking=True)