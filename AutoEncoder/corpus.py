import numpy as np

from utils import listdir_nohidden


def experiment_files_voc_train():
    """
        Gets the files from the vocalization corpus. Depending on where it
        is located hence why there is a base parameter.

        Returns two file paths that can be loaded.
    """
    #print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1"
    # base = '/Volumes/Promise_Pegasus/Teun_Corpora/vocalizationcorpus/data/'
    # base = '/home/teun/data_rack/vocalizationcorpus/data/'
    # base = '/home/visionlab/Teun/vocalizationcorpus/data/'
    # base = '/Volumes/Teun/vocalizationcorpus/data/'
    base = '/home/tinus/Workspace/corpus/data/'
    female = np.concatenate((np.arange(1,54), np.arange(72,202), np.arange(297,382),
                             np.arange(460,554), np.arange(583,615), np.arange(655,709),
                             np.arange(716,748), np.arange(816,900), np.arange(967,988),
                             np.arange(1027,1067), np.arange(1225,1274), np.arange(1316,1344),
                             np.arange(1368,1393), np.arange(1448,1466), np.arange(1472,1503),
                             np.arange(1517,1525), np.arange(1532,1613), np.arange(1639,1689)
                             ))
    male = np.concatenate((np.arange(54,72), np.arange(202,297), np.arange(382,460),
                           np.arange(554,583), np.arange(615,655), np.arange(709,716),
                           np.arange(748,816), np.arange(900,967), np.arange(988,1027),
                           np.arange(1067, 1225), np.arange(1274, 1316), np.arange(1344, 1368),
                           np.arange(1393, 1448), np.arange(1466, 1472), np.arange(1503, 1517),
                           np.arange(1525, 1532), np.arange(1613, 1639), np.arange(1689, 1712)
                           ))

    male_file = np.random.choice(male, 2)
    female_file = np.random.choice(female, 2)

    return base + create_filename(male_file[0]), base + create_filename(female_file[0])
    # return base + create_filename(male_file[0]), base + create_filename(1)
    # return base + create_filename(54), base + create_filename(1)

def experiment_files_voc_valid():
    """
        Gets the files from the vocalization corpus. Depending on where it
        is located hence why there is a base parameter.

        Returns two file paths that can be loaded.
    """
    #print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1"
    # base = '/Volumes/Promise_Pegasus/Teun_Corpora/vocalizationcorpus/data/'
    # base = '/home/teun/data_rack/vocalizationcorpus/data/'
    # base = '/home/visionlab/Teun/vocalizationcorpus/data/'
    # base = '/Volumes/Teun/vocalizationcorpus/data/'
    base = '/home/tinus/Workspace/corpus/data/'
    female = np.concatenate((np.arange(1712,1819), np.arange(2045,2060), np.arange(2095,2106),
                             np.arange(2185,2225), np.arange(2250,2260), np.arange(2329,2453)
                             ))
    male = np.concatenate((np.arange(1819, 2045), np.arange(2060, 2095), np.arange(2106, 2185),
                           np.arange(2225, 2250), np.arange(2260, 2329), np.arange(2453, 2545)
                           ))

    male_file = np.random.choice(male, 2)
    female_file = np.random.choice(female, 2)

    return base + create_filename(male_file[0]), base + create_filename(female_file[0])
    # return base + create_filename(male_file[0]), base + create_filename(1)
    # return base + create_filename(54), base + create_filename(1)

def experiment_files_voc_sample():
    """
        Gets the files from the vocalization corpus. Depending on where it
        is located hence why there is a base parameter.

        Returns two file paths that can be loaded.
    """
    #print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1"
    # base = '/Volumes/Promise_Pegasus/Teun_Corpora/vocalizationcorpus/data/'
    # base = '/home/teun/data_rack/vocalizationcorpus/data/'
    # base = '/home/visionlab/Teun/vocalizationcorpus/data/'
    base = '/Volumes/Teun/vocalizationcorpus/data/'

    return base + create_filename(54), base + create_filename(1)

def experiment_files_voc_test():
    """
        Gets the files from the vocalization corpus. Depending on where it
        is located hence why there is a base parameter.

        Returns two file paths that can be loaded.
    """
    base = '/Volumes/Teun/vocalizationcorpus/data/'
    female = np.concatenate((np.arange(2545,2574), np.arange(2666,2676), np.arange(2717,2763)))
    male = np.concatenate(( np.arange(2574, 2666), np.arange(2676, 2717)))

    male_file = np.random.choice(male, 2)
    female_file = np.random.choice(female, 2)

    return base + create_filename(male_file[0]), base + create_filename(female_file[0])

def experiment_files_voc_explain():
    """
        Gets the files from the vocalization corpus. Depending on where it
        is located hence why there is a base parameter.

        Returns two file paths that can be loaded.
    """
    #print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1"
    # base = '/Volumes/Promise_Pegasus/Teun_Corpora/vocalizationcorpus/data/'
    # base = '/home/teun/data_rack/vocalizationcorpus/data/'
    # base = '/home/visionlab/Teun/vocalizationcorpus/data/'
    base = '/Volumes/Teun/vocalizationcorpus/data/'
    female = np.concatenate((np.arange(1,54), np.arange(72,202), np.arange(297,382), np.arange(460,554), np.arange(583,615), np.arange(655,709), np.arange(716,748), np.arange(816,900), np.arange(967,988), np.arange(1027,1067), np.arange(1225,1274), np.arange(1316,1344), np.arange(1368,1393), np.arange(1448,1466), np.arange(1472,1503), np.arange(1517,1525), np.arange(1532,1613), np.arange(1639,1689), np.arange(1712,1819), np.arange(2045,2060), np.arange(2095,2106), np.arange(2185,2225), np.arange(2250,2260), np.arange(2329,2453), np.arange(2545,2574), np.arange(2666,2676), np.arange(2717,2763)))
    male = np.concatenate((np.arange(54,72), np.arange(202,297), np.arange(382,460), np.arange(554,583), np.arange(615,655), np.arange(709,716), np.arange(748,816), np.arange(900,967), np.arange(988,1027), np.arange(1067, 1225), np.arange(1274, 1316), np.arange(1344, 1368), np.arange(1393, 1448), np.arange(1466, 1472), np.arange(1503, 1517), np.arange(1525, 1532), np.arange(1613, 1639), np.arange(1689, 1712), np.arange(1819, 2045), np.arange(2060, 2095), np.arange(2106, 2185), np.arange(2225, 2250), np.arange(2260, 2329), np.arange(2453, 2545), np.arange(2574, 2666), np.arange(2676, 2717)))

    male_file = np.random.choice(male, 2)
    female_file = np.random.choice(female, 2)

    # return base + create_filename(male_file[0]), base + create_filename(female_file[0])
    # return base + create_filename(male_file[0]), base + create_filename(1)
    return base + create_filename(54), base + create_filename(1)

def experiment_files_stefan():
    """
        Gets the files from the Stefan corpus. Depending on where it
        is located hence why there is a base parameter.

        Returns two file paths that can be loaded.
    """
    base = '/Volumes/Teun/Stefan/'

    # rooms = listdir_nohidden(base)
    # room = np.random.choice(rooms, 1)
    # people = np.random.choice(listdir_nohidden(room[0]), 2)
    mixture = np.random.choice(listdir_nohidden(base), 1)
    # print(mixture[0])
    return mixture

def experiment_files_bird_set():
    """
        Gets the files from the bird corpus. Depending on where it
        is located hence why there is a base parameter.

        Returns two file paths that can be loaded.
    """
    # base = '/Volumes/Promise_Pegasus/Teun_Corpora/vocalizationcorpus/data/'
    base = '/home/teun/data_rack/bird_set/'
    # base = '/home/visionlab/Teun/bird_set/'
    folders = listdir_nohidden(base)

    c_folders = np.random.choice(folders, 2)

    file_1 = np.random.choice(listdir_nohidden(c_folders[0]), 1)
    file_2 = np.random.choice(listdir_nohidden(c_folders[1]), 1)

    return file_1[0], file_2[0]


def experiment_files_MTA(self):
    """
        Gets the files from the American MapTask corpus. Depending on where it
        is located hence why there is a base parameter.

        Returns two file paths that can be loaded.
    """
    #print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
    # base = '/Volumes/Promise_Pegasus/Teun_Corpora/vocalizationcorpus/data/'
    # base = '/home/visionlab/Teun/MapTask_American/'
    base = '/home/teun/data_rack/MapTask_American/'
    files = listdir_nohidden(base)

    c_files = np.random.choice(files, 2)

    #return c_files[0], c_files[1]
    return '/home/teun/data_rack/MapTask_American/AEMTlist_s8_ms.wav', '/home/teun/data_rack/MapTask_American/AEMTlist_s4_ms.wav'


def experiment_files_MTS():
    """
        Gets the files from the Scottish MapTask corpus. Depending on where it
        is located hence why there is a base parameter.

        Returns two file paths that can be loaded.
    """
    # base = '/Volumes/Promise_Pegasus/Teun_Corpora/vocalizationcorpus/data/'

    # base = '/home/visionlab/Teun/MapTask_Scots/'
    base = '/home/teun/data_rack/MapTask_Scots/'
    files = listdir_nohidden(base)

    c_files = np.random.choice(files, 2)

    return c_files[0], c_files[1]


def experiment_files_MTE():
    """
        Gets the files from the English MapTask corpus. Depending on where it
        is located hence why there is a base parameter.

        Returns two file paths that can be loaded.
    """
    # base = '/Volumes/Promise_Pegasus/Teun_Corpora/vocalizationcorpus/data/'
    # base = '/home/teun/data_rack/MapTask_English/'
    base = '/home/teun/data_rack/MapTask_English/'
    folders = listdir_nohidden(base)

    c_folders = np.random.choice(folders, 2)

    file_1 = np.random.choice(listdir_nohidden(c_folders[0]), 1)
    file_2 = np.random.choice(listdir_nohidden(c_folders[1]), 1)

    return file_1[0], file_2[0]


def create_filename(number):
    """
        Creates a filename for the vocalization corpus because this corpus
        has numbers as filename so we can randomly generate the file names.

        Keyword Argument:
        number -- the number which we translate to a file name.

        Returns a string that is the file name.
    """
    if number >= 1000:
        return 'S' + str(number) + '.wav'
    elif number >= 100:
        return 'S0' + str(number) + '.wav'
    elif number >= 10:
        return 'S00' + str(number) + '.wav'
    else:
        return 'S000' + str(number) + '.wav'
