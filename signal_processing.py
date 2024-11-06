import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import scipy as sp

#Loading audio files

bebop = "DroneAudioDataset/Multiclass_Drone_Audio/bebop_1/B_S2_D1_067-bebop_000_.wav"
membo = "DroneAudioDataset/Multiclass_Drone_Audio/membo_1/extra_membo_D2_2000.wav"
unknown = "DroneAudioDataset/Multiclass_Drone_Audio/unknown/1-137-A-320.wav"

audio_bebop , framerate_bebop = librosa.load(bebop,sr=None) #sr=None conserve le framerate d'origine
audio_membo , framerate_membo = librosa.load(membo,sr=None)
audio_unknown , framerate_unknown = librosa.load(unknown,sr=None)

print("BEBOP : fréquence d'échantillonage:", framerate_bebop)
print("MEMBO: fréquence d'échantillonage:", framerate_membo)
print("UNKNOWN : fréquence d'échantillonage:", framerate_unknown)

print("BEBOP : données audio:", audio_bebop)
print("MEMBO : données audio:", audio_membo)
print("UNKNOWN : données audio:", audio_unknown)

print("taille BEBOP:", len(audio_bebop))
print("taille membo:", len(audio_membo))
print("taille unknown:", len(audio_unknown))

plt.figure(figsize=(13,15))

plt.subplot(3,1,1)
librosa.display.waveshow(audio_bebop, alpha=0.5)
plt.title("BEBOP")

plt.subplot(3,1,2)
librosa.display.waveshow(audio_membo, alpha=0.5)
plt.title("MEMBO")

plt.subplot(3,1,3)
librosa.display.waveshow(audio_unknown, alpha=0.5)
plt.title("UNKNOWN")

plt.show()


#Fourier Transform

ft_bebop = sp.fft.fft(audio_bebop)
magnitude_ftt_bebop = np.abs(ft_bebop)
frequency_fft_bebop = np.linspace(0, framerate_bebop, len(audio_bebop))

ft_membo = sp.fft.fft(audio_membo)
magnitude_ft_membo = np.abs(ft_membo)
frequency_ft_membo = np.linspace(0, framerate_membo, len(audio_membo))

ft_unknown = sp.fft.fft(audio_unknown)
magnitude_ft_unknown = np.abs(ft_unknown)
frequency_ft_unknown = np.linspace(0, framerate_unknown, len(audio_unknown))

plt.figure(figsize=(13,15))

plt.subplot(3,1,1)
plt.plot(frequency_fft_bebop, magnitude_ftt_bebop)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")

plt.subplot(3,1,2)
plt.plot(frequency_ft_membo, magnitude_ft_membo)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")

plt.subplot(3,1,3)
plt.plot(frequency_ft_unknown, magnitude_ft_unknown)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")

plt.show()

#With Fourier Transforms we know what but don't know when
#Small segment of the signal and apply fft locally : STFT (short time fourier transform)
#Fenêtre si taille de la fenêtre = frame_size alors pas d'overlap
#Si tailles différentes alors overlap et différence des tailles = hop_size

#Extracting Short-Time Fourier Transform:

frame_size = 1024
hop_size = 512

S_scale_bebop = librosa.stft(audio_bebop)
print("STFT bebop:", S_scale_bebop.shape)
S_scale_membo = librosa.stft(audio_membo)
print("STFT membo:", S_scale_membo.shape)
S_scale_unknown = librosa.stft(audio_unknown)
print("STFT unknown:", S_scale_unknown.shape)

#Spectrum
Y_scale_bebop = np.abs(S_scale_bebop)**2
Y_scale_membo = np.abs(S_scale_membo)**2
Y_scale_unknown = np.abs(S_scale_unknown)**2

"""
Spectogramme issu d'une STFT est une représentation visuelle qui montre comment 
le contenu fréquentiel d'un signal audio évolue au cours du temps.

Transformée de Fourier à court terme (STFT) :

    La transformée de Fourier classique analyse un signal pour en extraire ses composantes fréquentielles, mais elle ne permet pas de savoir quand ces fréquences apparaissent.
    La STFT, en revanche, segmente le signal en petites fenêtres de temps (d'où le terme "court terme") puis applique la transformée de Fourier à chaque segment.
    En découpant le signal en intervalles, on peut observer les fréquences présentes à chaque instant (approximatif) du signal.

Fonctionnement :

    On prend une fenêtre du signal d’une durée fixe (en millisecondes par exemple).
    On applique la transformée de Fourier sur cette fenêtre pour extraire le spectre fréquentiel de ce court segment.
    Ensuite, on décale la fenêtre dans le temps, de manière à répéter l’opération sur toute la longueur du signal.
    Chaque segment produit un spectre des fréquences pour un intervalle spécifique de temps.
    L'ensemble de ces spectres, mis bout à bout, forme une représentation en deux dimensions du signal.

Spectrogramme :

    Une fois la STFT calculée, on obtient un spectrogramme en représentant les magnitudes des fréquences (ou puissances) pour chaque fenêtre temporelle.
    Le spectrogramme est une matrice où chaque point (ou pixel) représente une fréquence donnée à un moment précis, avec une couleur ou une intensité indiquant l’amplitude ou la puissance de cette fréquence.
    
aille de la fenêtre :

    Une petite fenêtre temporelle permet une bonne précision temporelle (on peut mieux savoir quand une fréquence apparaît) mais diminue la précision en fréquence.
    Une grande fenêtre améliore la résolution en fréquence mais diminue la précision temporelle.

Fenêtrage et chevauchement :

    Pour éviter les discontinuités, chaque fenêtre est souvent pondérée par une fonction de fenêtrage (ex : fenêtre de Hamming).
    Le chevauchement entre les fenêtres successives permet de lisser la transition entre les segments et de réduire la perte d’information temporelle.
"""

def plot_spectrogram(Y, sr, hop_length,title ,y_axis="linear"):
    plt.figure(figsize=(7,5))
    librosa.display.specshow(Y,
                             sr=sr,
                             hop_length=hop_length,
                             x_axis="time",
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")
    plt.title(title)
    plt.show()

plot_spectrogram(Y_scale_bebop, framerate_bebop,hop_size, "Bebop linéaire")


#Log-Amplitude Spectrogram

#L'amplitude est exprimée en échalle log
Y_log_bebop = librosa.power_to_db(Y_scale_bebop)
plot_spectrogram(Y_log_bebop, framerate_bebop,hop_size, "Bebop log")
Y_log_membo = librosa.power_to_db(Y_scale_membo)
plot_spectrogram(Y_log_membo, framerate_membo, hop_size, "Membo log")
Y_log_unknown = librosa.power_to_db(Y_scale_unknown)
plot_spectrogram(Y_log_unknown, framerate_unknown,hop_size, "Unknown log")
#Pas très bonne séparation des fréquences donc intéressant de tester représentation log en fréquence
#Log-Frequency spectogram

plot_spectrogram(Y_log_bebop, framerate_bebop,hop_size, "Bebop log-freq", y_axis="log")
plot_spectrogram(Y_log_membo, framerate_membo, hop_size, "Membo log-freq", y_axis="log")
plot_spectrogram(Y_log_unknown, framerate_unknown,hop_size, "Unknown log-freq", y_axis="log")

#Mel-Spectrogram

"""
Mel-spectogram est une représentation spectrale d'un signal audio qui utilise une échelle
fréquentielle appelée échelle Mel qui est plus adaptée à la perception humaine des fréquences.

Pourquoi l'échelle Mel : oreille humaine plus sensible aux variations de fréquence dans les basses
fréquences qu'aux variations dans les hautes fréquences

L'échelle Mel compresse donc les hautes fréquences tout en offrant plus de détails dans les basses
fréquences ce qui rend les Mel-spectrogrammes particulièrement adaptés pour les tâches de reconnaissance 
vocale.

Calcul du Mel-spctrogramme : 
-calcul de la STFT pour obtenir un spectogramme classique
-appliquée une banque de filtres Mel aux fréquences issues de la STFT:
    Cette banque de filtres est une série de filtres triangulaires, chacun centré sur une fréquence Mel spécifique.
    Chaque filtre capte une bande de fréquences et leur applique un poids en fonction de la réponse en fréquence de l’échelle Mel.
    Ce filtrage regroupe et compresse les fréquences, réduisant ainsi la résolution en fréquence tout en augmentant la pertinence perceptuelle.
-on prend ensuite le logarithme de l'amplitude ou de la puissance 
"""

# EXTRACTION MEL SPECTOGRAMS

#mel filter banks

"""

filter_banks = librosa.filters.mel(n_fft=1024, sr=framerate_bebop, n_mels=10)

plt.figure(figsize=(7,5))
librosa.display.specshow(filter_banks, sr = framerate_bebop, x_axis="linear")
plt.colorbar(format="%+2.f")
plt.title("Filtres mel bebop")
plt.show()

# Extracting Mel Spectrogram

mel_spectrogram_bebop = librosa.feature.melspectrogram(audio_bebop, sr=framerate_bebop, n_fft=1024,hop_length=512)
mel_spectrogram_bebop_log = librosa.power_to_db(mel_spectrogram_bebop)

mel_spectrogram_membo = librosa.feature.melspectrogram(audio_membo, sr=framerate_membo, n_fft = 1024, hop_length=512)
mel_spectrogram_membo_log = librosa.power_to_db(mel_spectrogram_membo)

mel_spectrogram_unknown = librosa.feature.melspectrogram(audio_unknown, sr= framerate_unknown, n_fft = 1024 ,hop_length=512)
mel_spectrogram_unknown_log = librosa.power_to_db(mel_spectrogram_unknown)

plt.figure(figsize=(25,15))
plt.subplot(311)
librosa.display.specshow(mel_spectrogram_bebop_log,
                         x_axis="time",
                         y_axis="mel",
                         sr=framerate_bebop)
plt.colorbar(format="%+2.f")
plt.subplot(312)
librosa.display.specshow(mel_spectrogram_membo_log,
                         x_axis="time",
                         y_axis="mel",
                         sr=framerate_membo)
plt.colorbar(format="%+2.f")
plt.subplot(313)
librosa.display.specshow(mel_spectrogram_unknown_log,
                         x_axis="time",
                         y_axis="mel",
                         sr=framerate_unknown)
plt.colorbar(format="%+2.f")
plt.show()

"""

"""
                ADVANCED TIME-FREQUENCY ANALYSIS
#-----------------The Wigner-Ville Distribution-----------------#

La WVD peut être interprétée comme la densité d'énergie dans le temps et la fréquence.
Cependant cette tranformation fait apparaître des termes croisés.
Il est intéréssant de conservé la WVD (energie et moment) mais en conservant ...
Une approche pour créer une représentation temps fréquence qui respecte les critères et d'utiliser un algorithme itératif 
pour transformer la WVD en une distribution positive.


Limitations de la WVD :

    Présence de termes croisés : La WVD souffre de la présence de « termes croisés », qui apparaissent lorsque le signal contient plusieurs composants fréquentiels proches les uns des autres. Ces termes introduisent du bruit dans la représentation, ce qui complique l'analyse.
    Positivité et Interprétation : Bien que la WVD représente une densité d'énergie, elle peut prendre des valeurs positives et négatives, ce qui rend difficile son interprétation en tant qu'énergie positive pure.

#-----------------Positive Transformed Wigner Distribution-----------------#

Pour calculer la PTWD, un algorithme itératif est utilisé, basé sur l'algorithme LMS (Least Mean Squares). Cet algorithme est adapté pour minimiser les termes croisés tout en respectant les moments requis, ce qui permet d'obtenir une distribution temps-fréquence transformée 
avec des termes croisés réduits et des valeurs positives.



The Wigner-Ville distribution was chosen for three reasons.
First, it provides more details in the time-frequency domain
than the mainstream techniques like the spectrogram.
Second, it was interesting to investigate its potential use for
sound classification
"""


