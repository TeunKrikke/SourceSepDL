import numpy as np

from librosa.core import load
from librosa.core import stft as rose_stft
from librosa.core import istft as rose_istft
from librosa.output import write_wav as rose_write_wav

from librosa.feature import mfcc

import glob
import os


def read_wav(filename, sr=16000):
    '''
     load the data from a file

     parameters
     filename - what the file to load is called
     sr - sample rate of the file (default 16000)
    '''
    data, _ = load(filename, sr)
    return data


def stft(x, windowsize=1024, overlap=4, N=None, do_log=True, do_mag=True):
    # Window 480 fft 512 overlap 60%
    hop = int(windowsize * overlap)
    X = rose_stft(x, win_length=windowsize, hop_length=hop, n_fft=1024).real
    X = X.T
    if do_log and not do_mag:
        return np.log10(np.absolute(X) + 1e-7)
    if do_mag and not do_log:
        return np.power(np.absolute(X), 2)
    return X


def istft(X, windowsize=1024, overlap=4):
    X = X.T
    hop = int(windowsize * overlap)
    return rose_istft(X, win_length=windowsize, hop_length=hop)


def get_MFCC(mix, speaker_1_padded, speaker_2_padded, sr=16000, nr=13):

    mix_MFCC = mfcc(y=mix, sr=sr, n_mfcc=nr)
    speaker_1_MFCC = mfcc(y=mix, sr=sr, n_mfcc=nr)
    speaker_2_MFCC = mfcc(y=mix, sr=sr, n_mfcc=nr)

    return mix_MFCC, speaker_1_MFCC, speaker_2_MFCC


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
    nsamp = max(len(speaker_1), len(speaker_2)) + 1 # determine which file is longest and save that lenght

    # make both files even length by zero padding
    speaker_1_padded = np.pad(speaker_1, (0, nsamp - len(speaker_1)),
                              mode='constant', constant_values=0)
    speaker_2_padded = np.pad(speaker_2, (0, nsamp - len(speaker_2)),
                              mode='constant', constant_values=0)

    # mix both files
    mix = speaker_1_padded + speaker_2_padded

    return mix, speaker_1_padded, speaker_2_padded


def create_mixture_no_gt(corpus):
    '''
    load a mixture for which we do not have a ground truth (gt)

    parameters:
    corpus - the directory to load the files from
    win - the length of the stft window (default 255)
    overlap - the amounth the windows will overlap (default 0.5)

    returns:
    mix - the raw version of the mixture
    '''
    mix = map(read_wav, corpus()) # load the audio file

    return np.array(mix)


def do_STFT_on_data(mix, speaker_1_padded=None, speaker_2_padded=None, win=255,
                    overlap=0.5, do_log=True, do_mag=True):
    '''
        applies the STFT to all the input files

        parameters:
        mix - the raw mixture
        speaker_1_padded - the raw padded signal of speaker 1
        speaker_2_padded - the raw padded signal of speaker 2
        win - window size of the STFT (default 255)
        overlape - the overlap of each window (default 0.5)

    '''
    # get the STFT of the different files so that we have an direct input
    Mix_STFT = stft(mix, windowsize=win, overlap=overlap, do_log=do_log,
                    do_mag=do_mag)
    if speaker_1_padded is None and speaker_2_padded is None:
        return Mix_STFT
    elif speaker_1_padded is not None and speaker_2_padded is not None:
        speaker_1_STFT = stft(speaker_1_padded, windowsize=win,
                              overlap=overlap,
                              do_log=do_log, do_mag=do_mag)

        speaker_2_STFT = stft(speaker_2_padded, windowsize=win,
                              overlap=overlap,
                              do_log=do_log, do_mag=do_mag)

        return Mix_STFT, speaker_1_STFT, speaker_2_STFT


def create_batches(Mix_STFT, speaker_1_STFT, speaker_2_STFT, batch_hop=500,
                   batch_size=855):
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

        mix_batched = []
        for i in range(0, len(Mix_STFT) - (batch_hop), batch_hop):
            temp = np.array(Mix_STFT[i:i + batch_size])
            if i == 0 or temp.shape[0] == mix_batched[0].shape[0]:
                mix_batched.append(temp)

        mix_batched = np.array(mix_batched)

    return mix_batched

def create_clust_output(X, gt, DB_THRESHOLD=40):
    # Get dominant spectra indexes, create one-hot outputs
    Y = np.zeros(X.shape + (2,))

    vals = np.argmax(gt, axis=0)
    for i in range(2):
        t = np.zeros(2)
        t[i] = 1
        Y[vals == i] = t

    # Create mask for zeroing out gradients from silence components
    m = np.max(X) - DB_THRESHOLD/20.  # From dB to log10 power
    z = np.zeros(2)
    Y[X < m] = z

    return Y

def create_stft_output(X, gt, DB_THRESHOLD=40):
    # Get dominant spectra indexes, create one-hot outputs
    Y = gt.reshape(gt.shape[1],gt.shape[2], gt.shape[0])

    return Y

def normalize(x, axis=None):
    return x / (np.sum(x, axis, keepdims=True) + 1e-12)

def generate_stft_examples(corpus, output_gen, batch_size=128, do_log=False, do_mag=False, do_pobs=False, do_mfcc=False):
    TIMESTEPS = 100
    batch_x = []
    batch_pobs = []
    batch_y1 = []
    batch_y2 = []
    batch_count = 0

    while True:
        mix, speaker_1_padded, speaker_2_padded = create_mixture(corpus)
        if not do_mfcc:
            Mix_STFT, speaker_1_STFT, speaker_2_STFT = do_STFT_on_data(mix, speaker_1_padded, speaker_2_padded, do_log=do_log, do_mag=do_mag)
            if do_pobs:
                pobs = normalize(np.abs(Mix_STFT))
        else:
            Mix_STFT, speaker_1_STFT, speaker_2_STFT = get_MFCC(mix, speaker_1_padded, speaker_2_padded)
        specs = np.array([speaker_1_STFT, speaker_2_STFT])
        Y = output_gen(Mix_STFT, specs)
# (8192, 100, 513)
        # Generating sequences
        i = 0
        while i + TIMESTEPS < len(Mix_STFT):
            batch_x.append(Mix_STFT[i:i+TIMESTEPS].reshape((TIMESTEPS, -1)))
            batch_pobs.append(Mix_STFT[i:i+TIMESTEPS].reshape((TIMESTEPS, -1)))
            batch_y1.append(speaker_1_STFT[i:i+TIMESTEPS].reshape((TIMESTEPS, -1)))
            batch_y2.append(speaker_2_STFT[i:i+TIMESTEPS].reshape((TIMESTEPS, -1)))
            i += TIMESTEPS // 2

            batch_count = batch_count + 1

            if batch_count == batch_size:
                inp = np.array(batch_x).reshape((batch_size,
                                                 TIMESTEPS, -1))
                pobs_inp = np.array(batch_pobs).reshape((batch_size,
                                                 TIMESTEPS, -1))
                out1 = np.array(batch_y1).reshape((batch_size,
                                                 TIMESTEPS, -1))
                out2 = np.array(batch_y2).reshape((batch_size,
                                                 TIMESTEPS, -1))
                if do_pobs:
                    yield({'input': pobs_inp, 'true_input': inp},
                          {'speaker_1': out1, 'speaker_2' : out2})
                else:
                    yield({'input': inp},
                          {'speaker_1': out1, 'speaker_2' : out2})
                batch_x = []
                batch_pobs = []
                batch_y1 = []
                batch_y2 = []
                batch_count = 0



if __name__ == "__main__":
    # x, y = next(generate_stft_examples(corpus.experiment_files_voc, create_clust_output, batch_size=50))
    # print(x['input'].shape)
    # print(y['kmeans_o'].shape)
    import corpus
    x, y = next(generate_stft_examples(corpus.experiment_files_voc, create_stft_output, batch_size=50))
    print(x['input'].shape)
    print(y['speaker_1'].shape)
