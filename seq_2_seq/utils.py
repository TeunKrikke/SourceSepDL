import numpy as np

from librosa.core import load
from librosa.core import stft as rose_stft
from librosa.core import istft as rose_istft
from librosa.output import write_wav as rose_write_wav

import glob
import os


def read_wav(filename, sr=16000):
    '''
     load the data from a file

     parameters
     filename - what the file to load is called
     sr - sample rate of the file (default 16000)
    '''
    data, _ = load(filename,sr)
    return data

def stft(x, windowsize=1024, overlap=4, N=None):  # Window 480 fft 512 overlap 60%
    hop = int(windowsize * overlap)
    # X =  np.abs(rose_stft(x, win_length=windowsize, hop_length=hop, n_fft=1024))
    X =  rose_stft(x, win_length=windowsize, hop_length=hop, n_fft=1024).real
    return X.T
  
def istft(X, windowsize=1024, overlap=4,):   
    X = X.T
    hop = int(windowsize * overlap)
    return rose_istft(X, win_length=windowsize, hop_length=hop)

def write_wav(x, filename='signal.wav', sr=16000):
    '''
     write the data to a file

     parameters
     filename - what the file to write to  is called
     sr - sample rate of the file (default 16000)
    '''
    rose_write_wav(filename, x, sr)


def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))


def create_mixture(corpus):
    '''
    create a mixture of 2 files

    parameters:
    corpus - the directory to load the files from
    win - the length of the stft window (default 255)
    overlap - the amounth the windows will overlap (default 0.5)

    returns:
    mix - the raw version of the mixture
    speaker_1_padded the padded version of the speaker 1 file
    speaker_2_padded the padded version of the speaker 2 file
    '''
    speaker_1, speaker_2 = map(read_wav, corpus()) # load the audio files
    nsamp = max(len(speaker_1), len(speaker_2))+1 # determine which file is longest and save that lenght

    # make both files even length by zero padding
    speaker_1_padded = np.pad(speaker_1, (0,nsamp-len(speaker_1)), mode='constant', constant_values=0)
    speaker_2_padded = np.pad(speaker_2, (0,nsamp-len(speaker_2)), mode='constant', constant_values=0)

    # mix both files
    mix = speaker_1_padded + speaker_2_padded

    return mix, speaker_1_padded, speaker_2_padded

def do_STFT_on_data(mix, speaker_1_padded, speaker_2_padded, win=255, overlap=0.5):
    '''
        applies the STFT to all the input files

        parameters:
        mix - the raw mixture
        speaker_1_padded - the raw padded signal of speaker 1
        speaker_2_padded - the raw padded signal of speaker 2
        win - window size of the STFT (default 255)
        overlape - the overlap of each window (default 0.5)

    '''
    #get the STFT of the different files so that we have an direct input
    Mix_STFT = stft(mix, windowsize=win, overlap=overlap)
    speaker_1_STFT = stft(speaker_1_padded, windowsize=win, overlap=overlap)
    speaker_2_STFT = stft(speaker_2_padded, windowsize=win, overlap=overlap)

    return Mix_STFT, speaker_1_STFT, speaker_2_STFT

def create_batches(Mix_STFT, speaker_1_STFT, speaker_2_STFT, batch_hop=500, batch_size=855):
    '''
    create a batched version of the input

    parameters:
    Mix_STFT - the STFT of the mixture of the different speakers
    speaker_1_STFT - the STFT of speaker 1
    speaker_2_STFT - the STFT of speaker 2
    batch_hop - the number of items to skip (default 500)
    batch_size - the number of items in a batch (default 855)
    '''
    if Mix_STFT.shape[0] > batch_size: 
        
        mix_batched = create_batch(Mix_STFT, batch_hop, batch_size)
        speaker_1_batched = create_batch(speaker_1_STFT, batch_hop, batch_size)
        speaker_2_batched = create_batch(speaker_2_STFT, batch_hop, batch_size) 

    else:
        mix_batched = Mix_STFT
        speaker_1_batched = speaker_1_STFT
        speaker_2_batched = speaker_2_STFT
 
    return mix_batched, speaker_1_batched, speaker_2_batched

def create_batch(input, batch_hop, batch_size):
    '''
    create a batched version of the input

    parameters:
    input - the input to batch
    batch_hop - the number of items to skip
    batch_size - the size of a single batch

    output
    the batched version of the input
    '''

    output = []

    for i in range(0,len(input)-(batch_hop),batch_hop): 
        temp = np.array(input[i:i+batch_size]) 
        if i == 0 or temp.shape[0] == output[0].shape[0]: 
            output.append(temp) 
        
    output = np.array(output)

    return output
        
def util_slice(self, x, dim, index):
    return x[:, index * dim: dim * (index + 1)]

def get_slices(x, n):
    dim = int(K.int_shape(x)[1] / n)
    return [Lambda(util_slice, arguments={'dim': dim, 'index': i}, output_shape=lambda s: (s[0], dim))(x) for i in range(n)]

def to_list(x):
    if type(x) is not list:
        x = [x]
    return x