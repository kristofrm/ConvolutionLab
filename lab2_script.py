#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 21:12:11 2024

lab2_script.py

This lab illustrates convolution and its properties through various mediums and strategies 

Authors: Kristof Rohaly-Medved, Jordan Tyler

sources - numpy.org, ChatGPT - used for reference and function assistance

"""
#%% Part One: Convolution Properties - Plotting

# import needed functions
import numpy as np
from matplotlib import pyplot as plt 
import lab2_module as l2m
import sounddevice as sd
from scipy.io import wavfile

# create time vector
dt = 0.01 # sampling frequency
time = np.arange(0, 5 + dt, dt)

# create original and scaled input signal variable
input_signal = np.sin(6 * np.pi * time)
input_signal_scaled = 2 * input_signal

# create impulse variable
system_impulse = np.zeros(len(time))
system_impulse[(time >= 0.5) & (time <= 2)] = 1 # set rectangular pulse = 1 from 0.5-2 sec

# create figure
plt.figure(1, clear=True, figsize=(12,6))
plt.suptitle('Visualization of Convolution Properties')

# row 0, column 0: input_signal
plt.subplot(3, 3, 1)
plt.plot(time, input_signal)
plt.title('input_signal x(t)')
plt.xlabel('time (sec)')
plt.ylabel('amplitude (A.U.)')

# row 0, column 1: system_impulse
plt.subplot(3, 3, 2)
plt.plot(time, system_impulse)
plt.title('system_impulse h(t)')
plt.xlabel('time (sec)')
plt.ylabel('amplitude (A.U.)')

# row 0, column 2: convolution of input_signal and system_impulse
conv_one = np.convolve(input_signal, system_impulse)
conv_one_time = np.arange(0, len(conv_one) * dt, dt)
plt.subplot(3, 3, 3)
plt.plot(conv_one_time, conv_one)
plt.title('Convolution: x(t) * h(t)')
plt.xlabel('time (sec)')
plt.ylabel('amplitude (A.U.)')

# row 1, column 0: system_impulse
plt.subplot(3, 3, 4)
plt.plot(time, system_impulse)
plt.title('system_impulse h(t)')
plt.xlabel('time (sec)')
plt.ylabel('amplitude (A.U.)')

# row 1, column 1: input_signal
plt.subplot(3, 3, 5)
plt.plot(time, input_signal)
plt.title('input_signal x(t)')
plt.xlabel('time (sec)')
plt.ylabel('amplitude (A.U.)')

# row 1, column 2: convolution of system_impulse and input_signal
conv_two = np.convolve(system_impulse, input_signal)
conv_two_time = np.arange(0, len(conv_two) * dt, dt)
plt.subplot(3, 3, 6)
plt.plot(conv_two_time, conv_two)
plt.title('Convolution: h(t) * x(t)')
plt.xlabel('time (sec)')
plt.ylabel('amplitude (A.U.)')

# row 2, column 0: input_signal_scaled
plt.subplot(3, 3, 7)
plt.plot(time, input_signal_scaled)
plt.title('input_signal_scaled 2x(t)')
plt.xlabel('time (sec)')
plt.ylabel('amplitude (A.U.)')

# row 2, column 1: system_impulse
plt.subplot(3, 3, 8)
plt.plot(time, system_impulse)
plt.title('system_impulse h(t)')
plt.xlabel('time (sec)')
plt.ylabel('amplitude (A.U.)')

# row 2, column 2: convolution of input_signal_scaled and system_impulse
conv_three = np.convolve(input_signal_scaled, system_impulse)
conv_three_time = np.arange(0, len(conv_three) * dt, dt)
plt.subplot(3, 3, 9)
plt.plot(conv_three_time, conv_three)
plt.title('Convolution: 2x(t) * h(t)')
plt.xlabel('time (sec)')
plt.ylabel('amplitude (A.U.)')

# save figure as pdf
plt.tight_layout()
plt.savefig('convolution_properties_subplot.pdf')
plt.show()

# Step 6: Print statement
print("Two convolution properties that can be observed:")
print("1. Commutativity: Convolving x(t) * h(t) yields an equivalent result to convolving h(t) * x(t).")
print("2. Linear and Time-Invariant: Scaling the either input signal scales the convolution result by the same value.")
print()

#%% Part Two: Build a Convolution Function 

# create built signal from get_convolved_signal function
my_convolved_signal = l2m.get_convolved_signal(input_signal, system_impulse)

# create time vector for the signal
my_conv_time = np.arange(0, len(my_convolved_signal) * dt, dt)

# create new convolved figure
plt.figure(2, clear=True)
plt.plot(my_conv_time, my_convolved_signal)
plt.title('Iterated Discrete Convolution')
plt.xlabel('time (sec)')
plt.ylabel('amplitude (A.U.)')

# save the figure as pdf
plt.tight_layout()
plt.savefig('my_convolved_signal.pdf')
plt.show()

# compare both convolution results
conv_comparison = np.allclose(my_convolved_signal, conv_one) #np.allclose adapted from numpy.org
if conv_comparison:
    print("Both convolutions yield the same result. This makes sense as the same discrete convolution operation is being done in both scenarios, just in a different way.")
    print()
else:
    print("The results are different. There is some sort of error in calculation")
    print()
    
#%% Part Three: Simplify a Cascade System 

# create time vector for cascade
drug_time = np.arange(0, 50 + dt, dt)

# define input signal drug_dosage
drug_dosage = 1 - np.cos((np.pi / 4) * drug_time)

# define gut input signal h1(t)
h1 = ((1 / 4) * np.exp(-drug_time / 0.4)) * drug_dosage

# define blood input signal h2(t)
h2 = 1 - np.cos((np.pi / 4) * -drug_time)

# define kidney input signal h3(t)
h3 = np.exp(-2 * (drug_time - 1) ** 2)

# define whole body response convolving all 3 systems
body_impulse = np.convolve(h1, h2)
body_impulse = np.convolve(body_impulse, h3)

# create whole body system figure
plt.figure(3, clear=True)
plt.title('Whole Body Drug Delivery Cascade (Gut, Blood, & Kidney)')
plt.xlabel('time (sec)')
plt.ylabel('drug concentration (A.U.)')

# outer loop variation in denominator of x(t)
# loop format adapted from ChatGPT
for denominator_index in range(2, 7, 2):
    # inner loop variation in amplitude of x(t)
    for amplitude_index in range(0, 4):
        # define x(t) with altered variables
        drug_dosage_altered = amplitude_index - np.cos((np.pi / denominator_index) * drug_time)
        
        # call the run_drug_simulations function for each iteration of variables
        l2m.run_drug_simulations(drug_dosage_altered, body_impulse, dt, label=f"Amplitude: {amplitude_index}, Denominator: {denominator_index}")
        
# add legend and save as pdf
plt.legend()
plt.tight_layout()
plt.savefig('drug_simulation.pdf')
plt.show()

# print resulting maximum drug concentration parameters 
print("The parameters that result in the maximum drug concentration to the power of 10 with minimal fluctuation is an amplitude of 2 and denominator of 6.")
print()

#%% Part Four: Audio Convolution Pt 1

# a) download audio file

# b) load audio data and sampling rate
filename = 'example.wav'

sampling_rate, audio_data = wavfile.read(filename)

# c) combine right and left channels
audio_data = np.mean(audio_data, axis=1)

# d) clip song to 3 seconds
clip_duration = 3 #seconds
num_samples = clip_duration * sampling_rate
audio_data = (audio_data[num_samples:2*num_samples]) / 1000.0 # to reduce amplitude (volume)

# e) play clipped song
sd.play(audio_data, sampling_rate)
#%% Part Four: Audio Convolution Pt 2

# f) create corresponding time vector and plot audio file as a function of time
time_vector = np.arange(0, clip_duration, 1/sampling_rate)

plt.figure(4, clear=True)
plt.plot(time_vector, audio_data)
plt.title('Clipped Audio Signal')
plt.xlabel('time (seconds)')
plt.ylabel('amplitude (A.U.)')
plt.grid(True)
plt.show()
plt.tight_layout()


plt.savefig('clipped_audio_signal_plotted.pdf')

#Describe in a print statement: what does it look like?
print('The plot of the clipped audio signal looks like a typical sound wave with spikes in amplitude and width where the guitar is strummed')
print()

#%% Part Four: Audio Convolution Pt 3

# g) audio file with doubled sample rate
sd.play(audio_data, 2*sampling_rate)

#Describe in a print statement: how did you think this is going to affect the sound? How did it affect the sound?
print('When the sampling rate is doubled, the song sounds faster and the sound quality becomes a bit more clear, but the melody isnt as clear')
print()

#%% Part Four: Audio Convolution Pt 4

# h) audio file with halved sample rate
sd.play(audio_data, 0.5*sampling_rate)

#Describe in a print statement: how did you think this is going to affect the sound? How did it affect the sound?
print('When the sampling rate is halved, the song sounds slower but the sound quality consequently becomes a bit worse and more grainy')
print()

#%% Part Four: Audio Convolution Pt 5

# i) audio convolved with highpass filter
HPF = np.loadtxt('HPF_1000Hz_fs44100_n10001.txt')
audio_HPF = np.convolve(HPF, audio_data)

sd.play(audio_HPF, sampling_rate)

#Describe in a print statement: What did this do to your song?
print('When convolved with the highpass filter, the higher pitched notes/sounds of my song were emphasized and made louder while the lower tones were more muted')
print()

#%% Part Four: Audio Convolution Pt 6

# j) audio convolved with lowpass filter
LPF = np.loadtxt('LPF_1000Hz_fs44100_n10001.txt')
audio_LPF = np.convolve(LPF, audio_data)

sd.play(audio_LPF, sampling_rate)

#Describe in a print statement: What did this do to your song?
print('When convolved with the lowpass filter, the bass and lower tones of my song became more pronounced while the higher pitched sounds were more muted.')
print()

#%% Part Four: Audio Convolution Pt 7

# k) convolve with created filter
filter_impulse_rise = np.arange(0,0.02+0.0004,0.0004)
filter_impulse_fall = np.arange(0.02, -0.0004, -0.0004)
filter_impulse = np.concatenate([filter_impulse_rise, filter_impulse_fall])

audio_rising_falling_impulse = np.convolve(filter_impulse, audio_data)

sd.play(audio_rising_falling_impulse, sampling_rate)

#Describe in a print statement: How did it affect your result? What could this be called in non-technical terms?
print('The gradual rising and falling filter "smoothed" out the sound and made transitions between high and low pitches a bit more gradual and less sharp. It also added a bit of a fading in and out effect to the song, and almost sounded muffled.')
print()

#%% Part Four: Audio Convolution Pt 8

# l) array of 1, 10000 0s, then 1 convolved with audio signal
h = np.zeros(10002)
h[0] = 1
h[-1] = 1

audio_h = np.convolve(h, audio_data)

sd.play(audio_h, sampling_rate)

#Describe in a print statement: How did it affect your result? What could this be called in non-technical terms?
print('This convolution added an echo to the song')
print('In non-technical terms, this convolution plays the sound and then after a short delay plays the sound again, as dictated by the 1 then 10,000 0s then 1')
print()

#%% Part Four: Audio Convolution Pt 9

# m) impulse function from environment convolved with audio signal
filename = 'impulse.wav'
sampling_rate_impulse, audio_data_environment_impulse = wavfile.read(filename)

audio_environment_impulse = (np.convolve(audio_data_environment_impulse, audio_data)) / 100000.0 # to reduce amplitude (volume)

sd.play(audio_environment_impulse, sampling_rate_impulse)

#Describe in a print statement: how did you record the impulse response function of the room? How did it change the sound?
print('I recorded the impulse by recording a clap in the new environment. However, it sounds extremely loud, sharp, and quite different from the original sound')
print('Because there is not much echo or unique sound coming from the environment, the convolved sound does not sound much like the original sound played in the environment, but rather a loud mess of noise')

# n) Save the results of each of the convolutions as separate clearly labeled text files (e.g., highpassed.txt, lowpassed.txt).
np.savetxt('audio_HPF.txt', audio_HPF, delimiter=',')
np.savetxt('audio_LPF.txt', audio_LPF, delimiter=',')
np.savetxt('audio_rising_falling_impulse.txt', audio_rising_falling_impulse, delimiter=',')
np.savetxt('audio_h.txt', audio_h, delimiter=',')
np.savetxt('audio_environment_impulse.txt', audio_environment_impulse, delimiter=',')
