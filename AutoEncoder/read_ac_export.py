import numpy as np
from utils import read_wav
'''
     reads a .dat file build a camera and gets the signal from the microphones
     Applies FD beamforming to the signal.
'''
__author__ = 'tfk'

class Mic(object):
    """docstring for Mic"""
    def __init__(self, mic_name):
        super(Mic, self).__init__()
        self.mic_name = mic_name
        self.samples = []

    def set_position(self, position):
        self.position = position

    def set_samples(self, samples):
        self.samples = samples

    def set_number_of_samples(self, samples_count):
        self.number_of_samples = samples_count

    def set_time_factors(self, time_factor):
        self.time_factor = time_factor

    def set_map_factors(self, map_factor):
        self.map_factor = map_factor

    def set_map_offset(self, map_offset):
        self.map_offset = map_offset

    def __getitem__(self, key):
        return self.samples[key]

class Camera(object):
    def __init__(self):
        super(Camera, self).__init__()

    def set_camera_positon(self, position):
        self.position = position

    def set_camera_view(self, view):
        self.view = view

    def set_up_vector(self, up_vector):
        self.up_vector = up_vector

    def set_opening_angle(self, angle):
        self.opening_angle = angle

    def set_focal_distance(self, distance):
        self.focal_distance = distance

class Beamforming(object): 
    def __init__(self, filename, height=120, width=120, start_x=0, start_y=0):
        super(Beamforming, self).__init__()
        self.calculate_distance_matrix(filename, height=height, width=width, start_x=start_x, start_y=start_y) 
       

    def read_file(self, filename, subsample=8, mic_nr=18):
        self.mics = []
        data_file = open(filename, 'rb')
        #number of channels
        nbr_channels = np.fromfile(data_file, np.int32, count=1)

        #number of channels plus pins
        nbr_channels_plus_pins = np.fromfile(data_file, np.int32,count=1)
        
        #number of samples per channel
        nbr_samples_channel = np.fromfile(data_file, np.int32,count=nbr_channels)
        #number of digital pins per channel
        nbr_digital_pins_channel =  np.fromfile(data_file, np.int32,count=nbr_channels)
        
        #channel quantization
        channel_quantization = np.fromfile(data_file, np.int32,count=nbr_channels)
        
        #time factors
        time_factors = np.fromfile(data_file, np.double,count=nbr_channels)
        #map factors
        map_factors = np.fromfile(data_file, np.double,count=nbr_channels)
        #channel name length

        #legend names
        channel_name_lengthes = np.fromfile(data_file, np.int32,count=nbr_channels_plus_pins)
        is_digital_channel_name = np.fromfile(data_file, np.int32,count=nbr_channels_plus_pins)
        #index of the channel name length
        channel_name_length_idx = 0
        
        #list of channels names
        # channel_names_lst = list()
        for i in range(0,nbr_channels):
            #digital
            if nbr_digital_pins_channel[i]>0:    
                channel_name = np.fromfile(data_file, np.int8,count=channel_name_lengthes[channel_name_length_idx])
                # channel_names_lst.append(channel_name.tostring())
                channel_name_length_idx = channel_name_length_idx + 1
            #analog        
            else:
                # channel_name = np.fromfile(data_file, np.int8,count=channel_name_lengthes[channel_name_length_idx])
                mic = Mic(np.fromfile(data_file, np.int8,count=channel_name_lengthes[channel_name_length_idx]))
                # channel_names_lst.append(channel_name.tostring())
                channel_name_length_idx = channel_name_length_idx + 1
                
                mic.set_number_of_samples(nbr_samples_channel[i])
                mic.set_time_factors(time_factors[i])
                mic.set_map_factors(map_factors[i])
                self.mics.append(mic)

        map_factors = None
        time_factors = None
        nbr_channels_plus_pins=None
        channel_name_lengthes = None
        is_digital_channel_name = None
        channel_name_length_idx = None

        camera = Camera()

        # #camera position vector
        # camera_position = np.fromfile(data_file, np.double,count=3)
        # #camera view vector
        # camera_view = np.fromfile(data_file, np.double,count=3)
        # #camera up vector
        # camera_up_vector = np.fromfile(data_file, np.double,count=3)
        # #camera opening angle
        # camera_opening_angle = np.fromfile(data_file, np.double,count=1)
        # #focal distance to object
        # focal_dst = np.fromfile(data_file, np.double,count=1)


        camera.set_camera_positon(np.fromfile(data_file, np.double,count=3))
        camera.set_camera_view(np.fromfile(data_file, np.double,count=3))
        camera.set_up_vector(np.fromfile(data_file, np.double,count=3))
        camera.set_opening_angle(np.fromfile(data_file, np.double,count=1))
        camera.set_focal_distance(np.fromfile(data_file, np.double,count=1))

        #microhpones positions
        for i in range(0,nbr_channels):
            # positions = np.fromfile(data_file, np.double, count=3)
            if i == 72:
                np.fromfile(data_file, np.double, count=3)
            else:
                self.mics[i].set_position(np.fromfile(data_file, np.double, count=3))

        for i in range(len(self.mics)):
            self.mics[i].set_position(self.mics[i].position - self.mics[mic_nr-1].position)

        #read the signal of each microphonexLengthPlane=dFocus*2; % size of projektion plane
        for i in range(0,nbr_channels):
            #digital
            if nbr_digital_pins_channel[i]>0:  
                #TODO: verify this
                signal = np.fromfile(data_file, np.int8,count=nbr_samples_channel[i])
                # signal_mic_lst.append(signal)
            #analog        
            else:
                if channel_quantization[i] == 16:
                    signal = np.fromfile(data_file, np.int16,count=nbr_samples_channel[i])
                elif channel_quantization[i] == 32:
                    signal = np.fromfile(data_file, np.int32,count=nbr_samples_channel[i])
                self.mics[i].set_samples(signal)
               
        channel_quantization = None
        nbr_samples_channel = None
        nbr_channels = None
        nbr_digital_pins_channel = None

        if not subsample == 1:
            for mic_nr in range(len(self.mics)):
                self.mics[mic_nr].set_samples(self.mics[mic_nr][::subsample])

#        print len(self.mics[0].samples)

#        print len(self.mics)
        # map_offset = np.zeros(nbr_channels)
        # dBinaryFileVersion = np.fromfile(data_file, np.double, count=1)
        # dScriptVersion = 4.600
        # if dScriptVersion < dBinaryFileVersion:
        #     'A T T E N T I O N: binary  file version is newer than script version !!!'
        # map_offset = np.fromfile(data_file, np.double, count=nbr_channels);
        # for i in range(0,nbr_channels):
        #     mics[i].set_map_offset(map_offset[i])

        self.read_wav_files(filename, 0, 1)

        return camera, self.mics

    def read_wav_files(self, filename, noise_removed, echo_removed):
        if noise_removed:
            directory = 'no_noise/'
        elif echo_removed:
            directory = 'nonoise_echo/'

        splitted = filename.split('/')
        location = filename[:-len(splitted[-1])]
        filename = splitted[-1]
        filename = filename[:-4]

        location = location + directory
        print location
        for i in range(len(self.mics)):
            #Ales_103Mic4
            file_nr = ''
            if i < 10:
                file_nr = str(0) + '' + str(i)
            else:
                file_nr = str(i)

            file_to_load = location + filename + file_nr + 'Mic' + str(i+1) + '.wav'
            self.mics[i].set_samples(read_wav(file_to_load))

    def calculate_distance_matrix(self, filename, height=120, width=120, start_x=0, start_y=0):
        camera, mics = self.read_file(filename)

        if start_x != 0 or start_y != 0:
            box_height = height
            box_width = width
            height = 776
            width = 578
            

        nr_pixels = width*height

        speed_sound = 343.42
        #Check this value
        delta_time = self.mics[0].time_factor

        # length of x and y planes
        x_length_plane = camera.focal_distance*2
        y_length_plane = 2*camera.focal_distance*(3/4)

        # values of the coordinate system, asumming that the center of the plane is
        # at the same position as the camera
        x = np.linspace(x_length_plane/2, -x_length_plane/2, num=width)
        y = np.linspace(-y_length_plane/2, y_length_plane/2, num=height)
        
        if start_x != 0 or start_y != 0:
            height = box_height
            width = box_width

        # compute the distance from each pixel to each microphone
        self.dst = np.zeros((len(self.mics),nr_pixels))

        for i in range(0,len(self.mics)):
            #microphone coordinates
            
            xm = self.mics[i].position[0]
            ym = self.mics[i].position[1]
            zm = self.mics[i].position[2]        
            for j in range(start_y,start_y+height):   
                
                #pixel y-coordinate
                yp = y[j]
                
                for k in range(start_x,start_x+width): 
                    
                    #pixel x-coordinate
                    xp = x[k] 
                        
                    #distance calculation
                    distance_mic_point = np.sqrt( pow((xm-xp),2) + pow((ym-yp),2) + pow((zm-camera.focal_distance),2) )           
                    
                    #TODO: check this formula
                    total_distance = distance_mic_point/speed_sound/delta_time   
                    
                    self.dst[i][k+j*width] = total_distance

        camera = None
        x = None
        y = None

        

    def do_td_beamforming(self, filename, width=120, height=120, start_frame=0, start_x=0, start_y=0, subsample=True):
        # dst, mics = calculate_distance_matrix(filename, height=height, width=width, start_x=start_x, start_y=start_y)
        if filename.endswith('Teun_1.dat'):
            framerate = 48000/30
            subsample = False
        else:
            framerate = 192000/30
        start= (start_frame * framerate)
        end = ((start_frame+1) * framerate)
        nr_pixels = width*height
        signal_size = framerate
        self.failed = False

        frame_x = 0
        frame_y = 0
        
#        print start
#        print end

        # array of total signals
        if subsample:
            total_signal_pixel = np.zeros((width, height, signal_size/4))
        else:
            total_signal_pixel = np.zeros((width, height, signal_size))
        
        width = width - 1

        total_signal_pixel = self.calculate_signal_per_pixel(total_signal_pixel, signal_size, start, end, start_x, start_y, width, height, frame_x, frame_y, subsample, filename)
        

#        if self.failed:
#            print "failed at"
#            print start_frame 
#            print " "
        return total_signal_pixel

    def calculate_signal_per_pixel(self, total_signal_pixel, signal_size, start, end, start_x, start_y, width, height, frame_x, frame_y, subsample, filename):
        # compute a signal for each pixel
        # for i in range(0,int(nr_pixels)):
        for y in range(start_y,start_y+height):
            frame_x = 0
            for x in range(start_x,start_x+width):
                if subsample:    
                    total_signal = np.zeros(signal_size/4)
                else:
                    total_signal = np.zeros(signal_size)
                
                for j in range(0,len(self.mics)):
                    
                    #shift calculated according to the distance from the pixel to the microphone        
                    shift = self.dst[j][x+y*width]
                    
                    # org_signal =  mics[j].samples[start:end]         
                    if subsample:
                        shifted_signal = self.mics[j].samples[int(start+shift):int(end+shift):4]
                    else:
                        shifted_signal = self.mics[j].samples[int(start+shift):int(end+shift)]
                    try:
                        total_signal = np.add(total_signal, shifted_signal)
                    except ValueError, e:
                        self.failed = True
                        return
                    

                total_signal_pixel[frame_x,frame_y] = total_signal
                # print x
                frame_x = frame_x + 1
            frame_y = frame_y + 1

        return total_signal_pixel

    def build_image(self, total_signal_pixel):
        color_map = np.zeros(total_signal_pixel.shape)

        for i in range(0,int(height)):
            for j in range(0,int(width)):
                signal = total_signal_pixel[i][j]
                value = signal.max()
                color_map[i][j] = value

        return color_map

    def determine_TDoA(self, max_theata=56,max_azimuth=74,radius=1):
        TDoA_per_mic = np.zeros((57,75,72))
        TDoA_per_2_mics = np.zeros((57,75,72))
        # W = zeros((57,75,72,size(X,2)));
        speed_of_sound = 344
        k_0 = np.asarray([0,0,radius])
        for theata in range(max_theata):
            for azimuth in range(max_azimuth):
                for mic in range(72):
                    k_1 = np.negative(k_0).T.dot(self.mics[mic].position)/speed_of_sound
                    # print k_1
                    next_mic = mic + 1
                    if mic == 71: 
                        next_mic = 0

                    k_2 = np.negative(k_0).T.dot(self.mics[next_mic].position)/speed_of_sound
                    TDoA_per_mic[theata, azimuth, mic] = k_1
                    
                    TDoA_per_2_mics[theata, azimuth, mic] = k_1 - k_2
                    
                    
                    # for i in range(size(X,2)):
                    #     f_i = (i - 1) * fs/size(X,2) ;
                    #     W[theata+1, azimuth+1, mic, i] = np.exp(1i*2*np.pi*f_i*k_1 - k_2);
                    # end
                    
                
                # plot3([0,cos(degtorad(theata))], [0,sin(degtorad(azimuth))], [0,radius]);
        return TDoA_per_mic, TDoA_per_2_mics

    def determine_W(self, I, O, fs, N, max_theata=56,max_azimuth=66,radius=1,):
        speed_of_sound = 344
        
        W = np.zeros((I, O, len(self.mics)))
        for theata in range(max_theata):
            for azimuth in range(max_azimuth):
                k_0 = np.asarray([np.cos(np.deg2rad(theata)),np.sin(np.deg2rad(azimuth)),radius])
                for mic in range(len(self.mics)):
                    k_1 = np.negative(k_0).T.dot(self.mics[mic].position)/speed_of_sound
                    # print k_1
                    next_mic = mic + 1
                    if mic == 71: 
                        next_mic = 0

                    k_2 = np.negative(k_0).T.dot(self.mics[next_mic].position)/speed_of_sound
                    # TDoA_per_mic[theata, azimuth, mic] = k_1
                    
                    # TDoA_per_2_mics[theata, azimuth, mic] = k_1 - k_2
                    
                    
                    for i in range(I):
                        f_i = (i - 1) * fs/I;
                        W[i, theata* azimuth, mic] = np.exp(1j*2*np.pi*f_i*k_1 - k_2);

        return W

