from keras.callbacks import TensorBoard, ModelCheckpoint, TerminateOnNaN
import numpy as np
import mir_eval

from utils import create_mixture, do_STFT_on_data, write_wav, istft
from utils import generate_stft_examples, create_stft_output
from utils import create_mixture_no_gt
from IntegratedGradients import integrated_gradients

import matplotlib.pyplot as plt


class Network(object):
    """Network class which is implemented by the different networks"""
    def __init__(self, network):
        super(Network, self).__init__()
        # np.random.seed(1120)
        self.network = network
        self.sequences = 1
        self.timesteps = 100
        self.features = 513
        self.win = 255
        self.overlap = 0.5
        self.batch_size = 855

        self.batch_hop = 500

        self.SDR = 0
        self.SIR = 0
        self.SAR = 0
        self.num_mics = 1
        self.mixtures = 100

        self.DROPOUT = 0.5
        self.RDROPOUT = 0.2
        self.units = [500, 300, 300, 500]

    def train(self, optim, corpus, loss='mse', BATCH_SIZE=20, EPOCHS=1001):
        train_gen = generate_stft_examples(corpus[0], create_stft_output)
        if len(corpus) > 1:
            valid_gen = generate_stft_examples(corpus[1], create_stft_output)
        # with tf.device('/cpu:0'):
        model = self.model()
#        model = multi_gpu_model(model,gpus=2)
        tb = TensorBoard()
        mc = ModelCheckpoint('checkpoints/' + self.network + '{epoch:02d}.hdf5', save_weights_only=True, period=50)
        callbacks_list = [tb, mc]
#        model.validation_data= None

        model.compile(loss=loss, optimizer=optim)
        model.save_weights(self.network)
        if len(corpus) > 1:
            model.fit_generator(train_gen,
                                steps_per_epoch=1083,
                                epochs=EPOCHS,
                                max_q_size=512,
                                callbacks=callbacks_list,
                                validation_data=valid_gen,
                                validation_steps=500,
                                shuffle=False)
        else:
            model.fit_generator(train_gen,
                                steps_per_epoch=1083,
                                epochs=EPOCHS,
                                max_q_size=512,
                                callbacks=callbacks_list,
                                shuffle=False)

        model.save_weights(self.network)
        print(model.to_json())
        model.save(self.network + '_mod')

    def test_network(self, optim, corpus, loss='mse', do_log=False,
                     do_mag=False, pobs=False, gt=True, files=1000):

        model = self.model()
        model.load_weights(self.network, by_name=True)

        # model.compile(loss=loss, optimizer=optim)
        # model.summary()
        for i in range(files):
            if gt:
                self.run_test(model, corpus, do_log=do_log, do_mag=do_mag,
                              pobs=pobs, index=i)
            else:
                self.run_test_no_gt(model, corpus, do_log=do_log,
                                    do_mag=do_mag,
                                    pobs=pobs, index=i)
        files = files * 2

        print ('!!!!!SDR = ' + str(self.SDR / files) + ', SIR = ' +
               str(self.SIR / files) + ', SAR = ' + str(self.SAR / files))
        print (str(self.SAR / files) + ', ' + str(self.SDR / files) + ', ' +
               str(self.SIR / files))

    def do_explain(self, optim, corpus, loss='mse', do_log=False,
                     do_mag=False, pobs=False, gt=True):
        model = self.model()
        model.load_weights(self.network, by_name=True)
        # model.compile(loss=loss, optimizer=optim)

        self.explain(model, corpus, do_log=do_log,
                do_mag=do_mag,
                pobs=pobs)

    def run_test(self, model, corpus, do_log=False, do_mag=False, pobs=False,
                 index=0, no_gt=False, debug=False):
        TIMESTEPS = 100

        mix, speaker_1_padded, speaker_2_padded = create_mixture(corpus)

        Mix_STFT = do_STFT_on_data(mix, do_log=do_log, do_mag=do_mag)
        # Mix_STFT, signal_1, signal_2 = do_STFT_on_data(mix, speaker_1_padded,
        #                                                speaker_2_padded,
        #                                                do_log=do_log,
        #                                                do_mag=do_mag)

        signal_1 = np.array([0, 0])
        signal_2 = np.array([0, 0])
        shift = TIMESTEPS // 2  # original
        # shift = 1

        i = 0
        while i + TIMESTEPS <= len(Mix_STFT):
            batch_x = Mix_STFT[i:i + TIMESTEPS].reshape((1, TIMESTEPS, -1))
            predicted_signal_1, predicted_signal_2 = model.predict([batch_x])

            if i + TIMESTEPS == len(Mix_STFT):
                predicted_signal_1 = predicted_signal_1[0, :]
                predicted_signal_2 = predicted_signal_2[0, :]

                predicted_signal_1 = predicted_signal_1.reshape(TIMESTEPS, -1)
                predicted_signal_2 = predicted_signal_2.reshape(TIMESTEPS, -1)

                signal_1 = np.concatenate((signal_1, predicted_signal_1))
                signal_2 = np.concatenate((signal_2, predicted_signal_2))
            else:
                predicted_signal_1 = predicted_signal_1[0, 0:(shift)]
                predicted_signal_2 = predicted_signal_2[0, 0:(shift)]

                predicted_signal_1 = predicted_signal_1.reshape(shift, -1)
                predicted_signal_2 = predicted_signal_2.reshape(shift, -1)
                if i == 0:
                    signal_1 = predicted_signal_1
                    signal_2 = predicted_signal_2
                else:
                    signal_1 = np.concatenate((signal_1, predicted_signal_1))
                    signal_2 = np.concatenate((signal_2, predicted_signal_2))

            i += shift

        signal_1 = istft(signal_1, windowsize=self.win, overlap=self.overlap)
        signal_2 = istft(signal_2, windowsize=self.win, overlap=self.overlap)

        lenght_x = len(signal_1)
        if lenght_x > len(signal_2):
            lenght_x = len(signal_2)

        speaker_2_padded = speaker_2_padded[:lenght_x]
        speaker_1_padded = speaker_1_padded[:lenght_x]
        if no_gt or debug:
            write_wav(mix, filename='audio/' + self.network + str(index) + '_mix.wav')
            write_wav(signal_1, filename='audio/' + self.network + str(index) + '_signal1.wav')
            write_wav(signal_2, filename='audio/' + self.network + str(index) + '_signal2.wav')
        if debug:
            write_wav(speaker_1_padded, filename='audio/' + self.network + str(index) + '_signal1_gt.wav')
            write_wav(speaker_2_padded, filename='audio/' + self.network + str(index) + '_signal2_gt.wav')
        if not no_gt:
            try:
                predicted = np.stack([speaker_1_padded, speaker_2_padded])
                original = np.stack([signal_1, signal_2])
                bss = mir_eval.separation.bss_eval_sources(original, predicted)
                SDR = bss[0]
                self.SDR = self.SDR + SDR[0] + SDR[1]
                SIR = bss[1]
                self.SIR = self.SIR + SIR[0] + SIR[1]
                SAR = bss[2]
                self.SAR = self.SAR + SAR[0] + SAR[1]
                # print '!!!!!SDR = {}, SIR = {}, SAR = {}, perm = {}'.format(*bss)
            except ValueError:
                print "outcome is 0 (zero)"


    def explain(self, model, corpus, do_log=False, do_mag=False, pobs=False,
                index=0):
        ig1 = integrated_gradients(model, outchannels=[0,1])

        TIMESTEPS = 100

        mix, speaker_1_padded, speaker_2_padded = create_mixture(corpus)

        Mix_STFT, ref_1, ref_2 = do_STFT_on_data(mix, speaker_1_padded,
                                                       speaker_2_padded,
                                                       do_log=do_log,
                                                       do_mag=do_mag)

        signal_1 = np.array([0, 0])
        signal_2 = np.array([0, 0])
        shift = TIMESTEPS // 2

        i = 0
        while i + TIMESTEPS <= len(Mix_STFT):
            batch_x = Mix_STFT[i:i + TIMESTEPS].reshape((TIMESTEPS, -1))
            ref_1x = ref_1[i:i + TIMESTEPS].reshape((TIMESTEPS, -1))
            ref_2x = ref_2[i:i + TIMESTEPS].reshape((TIMESTEPS, -1))
            predicted_signal_1 = ig1.explain([batch_x], reference=np.array(ref_1x))

            predicted_signal_2 = ig1.explain([batch_x], outc=1, reference=np.array(ref_2x))

            if i + TIMESTEPS == len(Mix_STFT):
                predicted_signal_1 = np.array(predicted_signal_1[0])
                predicted_signal_2 = np.array(predicted_signal_2[0])

                predicted_signal_1 = predicted_signal_1.reshape(TIMESTEPS, -1)
                predicted_signal_2 = predicted_signal_2.reshape(TIMESTEPS, -1)

                signal_1 = np.concatenate((signal_1, predicted_signal_1))
                signal_2 = np.concatenate((signal_2, predicted_signal_2))
            else:
                predicted_signal_1 = np.array(predicted_signal_1[0][0:(shift)])
                predicted_signal_2 = np.array(predicted_signal_2[0][0:(shift)])

                predicted_signal_1 = predicted_signal_1.reshape(shift, -1)
                predicted_signal_2 = predicted_signal_2.reshape(shift, -1)
                if i == 0:
                    signal_1 = predicted_signal_1
                    signal_2 = predicted_signal_2
                else:
                    signal_1 = np.concatenate((signal_1, predicted_signal_1))
                    signal_2 = np.concatenate((signal_2, predicted_signal_2))

            i += shift


        th = max(np.abs(np.min(signal_1)),np.abs(np.max(signal_1)))
        plt.subplots(figsize=(10, 6))
        plt.imshow(signal_1[:,:], cmap="seismic", vmin=-1*th, vmax=th)
        plt.savefig(self.network + '_signal_1.png')
        # plt.show()

        plt.subplots(figsize=(10, 6))
        plt.imshow(signal_2[:,:], cmap="seismic", vmin=-1*th, vmax=th)
        plt.savefig(self.network + '_signal_2.png')
        # plt.show()

        plt.subplots(figsize=(10, 6))
        plt.imshow(Mix_STFT[:,:], cmap="seismic", vmin=-1*th, vmax=th)
        plt.savefig(self.network + '_mix.png')
        # plt.show()
