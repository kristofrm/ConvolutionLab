#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 21:12:11 2024

lab2_module.py

This module contains functions needed to convolve signals using manual inputs and numpy functions

Authors: Kristof Rohaly-Medved, Jordan Tyler

sources - ChatGPT - used for reference and function assistance

"""
#%% import needed functions

import numpy as np
from matplotlib import pyplot as plt

#%% Part Two: Build a Convolution Function

def get_convolved_signal(input_signal, system_impulse):
    '''
    Function that manually convolves two signals and returns a singular convolved signal

    Parameters
    ----------
    input_signal : Array of floats (time samples,)
        Array of signals at given time-stamp based on input signal equation.
    system_impulse : Arrays of floats (time samples,)
        Array of signals at given time-stamp based on impulse equation.

    Returns
    -------
    my_convolved_signal : Array of floats (time samples,)
        Array of signals at given time-stamp based on convolution of 2 input parameters.

    '''
    # create manual convolution my_convolved_signal
    my_convolved_signal = np.zeros(len(input_signal) + len(system_impulse) - 1)

    # calculate discrete convolution in nested loops 
    # method adapted from ChatGPT
    for input_index in range(len(input_signal)):
        for impulse_index in range(len(system_impulse)):
            my_convolved_signal[input_index + impulse_index] += input_signal[input_index] * system_impulse[impulse_index]

    return my_convolved_signal

#%% Part Three: Simplify a Cascade System 

def run_drug_simulations(input_signal, system_impulse, dt, label):
    '''
    Function that convolves two input signals and plots that convolution result

    Parameters
    ----------
    input_signal : Array of floats (time samples,)
        Array of signals at given time-stamp based on input signal equation.
    system_impulse : Array of floats (time samples,)
        Array of signals at given time-stamp based on impulse equation.
    dt : float
        Given sampling frequency.
    label : string, optional 
        Label to be displayed in legend on graph. The default is ''

    Returns
    -------
    None.

    '''
    # convolve input and impulse signals
    system_output = np.convolve(input_signal, system_impulse)
    # create time vector for drug simulation 
    conv_time_drug = np.arange(0, len(system_output) * dt, dt)
    
    # plot the drug simulation convolution
    plt.plot(conv_time_drug, system_output, label=label)
