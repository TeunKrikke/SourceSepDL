from CNN import CNN, LSTM, LSTM_backwards, DNN, RCNN

from keras.optimizers import SGD, Adam, Adamax, Nadam

from utils import read_wav, read_mp3


def main():
    top_level = '/home/teun/data/All_Recordings/'
    # folders = [join(top_level,f) for f in listdir(top_level) if isdir(join(top_level, f))]
    # train(folders)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#    optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#    optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
#    optimizer = SGD(lr=0.0005, momentum=0.9, nesterov=True)
#    gan = GAN()
#    gan.train(top_level, optimizer, optimizer, optimizer)
#   VOC corpus
    # network = LSTM('voc_LSTM_0')
    # network = LSTM('voc_LSTM_100')
    # network = LSTM('voc_LSTM_200')
    # network = LSTM('voc_LSTM_300')
    # network = LSTM('voc_LSTM_400')
    # network = LSTM('voc_LSTM_500')
    # network = LSTM_backwards('voc_LSTM_backwards_0')
    # network = LSTM_backwards('voc_LSTM_backwards_100')
    # network = LSTM_backwards('voc_LSTM_backwards_200')
    # network = LSTM_backwards('voc_LSTM_backwards_300')
    # network = LSTM_backwards('voc_LSTM_backwards_400')
    # network = LSTM_backwards('voc_LSTM_backwards_500')
    # network = DNN('voc_DNN_0')
    # network = DNN('voc_DNN_100')
    # network = DNN('voc_DNN_200')
    # network = CNN('voc_CNN_0')
    # network.test(top_level, optimizer, network.experiment_files_voc, noise_dim=(1, 2, 8))
    # network = CNN('voc_CNN_100')
    # network.test(top_level, optimizer, network.experiment_files_voc, noise_dim=(1, 2, 8))
    network = CNN('voc_CNN_200')
    #network = RCNN('voc_RCNN_0')
    #network.test(top_level, optimizer, network.experiment_files_voc, noise_dim=(1, 2, 8))
    #network = RCNN('voc_RCNN_100')
    # network.test(top_level, optimizer, network.experiment_files_voc, noise_dim=(1, 2, 8))
    # network = RCNN('voc_RCNN_200')
    
    network.test(top_level, optimizer, network.experiment_files_voc, noise_dim=(1, 2, 8))

# # #   MTA corpus

# #   MTE corpus

# #   MTS corpus

# #   bird corpus

if __name__ == "__main__":
    main()
