import os
import pymoduleconnector
import methods as m
import datetime
import numpy as np
from scipy import signal
from time import sleep
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class X4m200_reader:
    def __init__(self, device_name, FPS, iterations, pulses_per_step, dac_min, dac_max,
                    area_start, area_end, sample_time, path):
        self.device_name = device_name
        self.FPS = FPS
        self.iterations = iterations
        self.pulses_per_step = pulses_per_step
        self.dac_min = dac_min
        self.dac_max = dac_max
        self.area_start = area_start
        self.area_end = area_end
        self.sample_time = sample_time
        self.path = path
        self.bin_length = 8*1.5e8/23.328e9
        self.fast_sample_point = int((self.area_end - self.area_start)/self.bin_length + 2)
        self.bin_index = 0
        self.folder_name = ''
        self.values = []
        self.reset()
        self.mc = pymoduleconnector.ModuleConnector(self.device_name)
        self.xep = self.mc.get_xep()
        self.sys_init()
    
    def reset(self):
        mc = pymoduleconnector.ModuleConnector(self.device_name)
        xep = mc.get_xep()
        xep.module_reset()
        mc.close()
        sleep(3)

    def sys_init(self):
        app = self.mc.get_x4m200()
        try:
            app.set_sensor_mode(0x13, 0)  # Make sure no profile is running.
        except RuntimeError:
            # Profile not running, OK
            pass
        try:
            app.set_sensor_mode(0x12, 0)  # Manual mode.
        except RuntimeError:
            # Maybe running XEP firmware only?
            pass
        self.xep.x4driver_init()
        self.xep.x4driver_set_downconversion(1)

        self.xep.x4driver_set_iterations(self.iterations)
        self.xep.x4driver_set_pulses_per_step(self.pulses_per_step)
        self.xep.x4driver_set_dac_min(self.dac_min)
        self.xep.x4driver_set_dac_max(self.dac_max)
        # self.xep.x4driver_set_frame_area_offset(3)
        self.xep.x4driver_set_frame_area(self.area_start, self.area_end)
        self.xep.x4driver_set_fps(self.FPS)

    def read_apdata(self):
        #read a frame
        # data = xep.read_message_radar_baseband_float().get_I()
        data = self.xep.read_message_data_float().data
        data_length = len(data)

        # iq sample is complex, so the length of data is half of the length of data
        # iq sampling
        i_vec = np.array(data[:data_length//2])
        q_vec = np.array(data[data_length//2:])
        iq_vec = i_vec + 1j*q_vec

        ampli_data = abs(iq_vec)
        return ampli_data

    def get_data_matrix(self, bin):
        row = self.sample_time * self.FPS
        col = self.fast_sample_point
        amp_matrix = np.empty([row, col])

        old_time = datetime.datetime.now()
        n = 0
        seconds = 0
        new_time = datetime.datetime.now()
        while n < row:
            old_time = new_time
            new_time = datetime.datetime.now()
            interval = (new_time - old_time).microseconds
            if interval > 1/self.FPS*1000:
                ampli_data = self.read_apdata()
                amp_matrix[n] = ampli_data
                n += 1
                if n % self.FPS == 0:
                    seconds += 1
                    timer = '{:02d} second(s) passed!'.format(seconds)
                    print(timer, end="\r")
       
        folder_name = str(new_time.minute) + str(new_time.second) + 'time%ds' % self.sample_time
        isPath = os.path.exists(self.path)
        if not isPath:
            os.mkdir(self.path)
        path = self.path + folder_name
        folder = os.path.exists(path)
        if not folder:
            os.mkdir(path)
            amp_file = path + '\\amp_matrix.txt'
            np.savetxt(amp_file, amp_matrix)
            self.bin_index = bin
            JSON_data = {
            "device_name": self.device_name,
            "fps": self.FPS,
            "iterations": self.iterations,
            "pulses_per_step": self.pulses_per_step,
            "sample_time": self.sample_time,
            "bin_index": int(self.bin_index),
            "bin_length": self.bin_length,
            "area_start": self.area_start,
            "area_end": self.area_end,
            }
            m.write_json_data(JSON_data, path+"/param.json")
        else:
            print('error:the folder exists!!!')
        print(self.folder_name)
        print("data collection finished")
        return folder_name
        #TODO: make it like the other picker
    
    def plot_radar_raw_data_message(self):
        def read_frame():
            """Gets frame data from module"""
            d = self.xep.read_message_data_float()  # wait until get data
            frame = np.array(d.data)
            # Convert the resulting frame to a complex array if downconversion is enabled
            n = len(frame)
            # convert frame to complex
            frame = frame[:n // 2] + 1j * frame[n // 2:]
            frame = np.abs(frame)
            max_bin = np.argmax(frame) # find the index of the max value
            num_bin = max_bin % self.xep.x4driver_get_frame_bin_count() # do I really need this line?
            self.bin_index = num_bin
            bin_txt.set_text('Target bin number: {} from {} bins'.format(str(self.bin_index), str(self.xep.x4driver_get_frame_bin_count())))
            dist_arange = np.arange(self.area_start, self.area_end, self.bin_length)
            distance_txt.set_text('Target distance: {:.2f} cm'.format((dist_arange[self.bin_index]+self.bin_length)*100))
            return frame
        
        
        def animate(i):
            line.set_ydata(read_frame())  # update the data
            line2.set_data([ax_x[self.bin_index],ax_x[self.bin_index]], [0,0.02])

        def onpick(event):
            thisline = event.artist
            xdata = thisline.get_xdata()
            self.bin_index = np.where(ax_x == xdata[0])[0][0]
            plt.close()

        ax_x = np.arange((self.area_start-1e-5), (self.area_end-1e-5)+self.bin_length, self.bin_length)
        fig = plt.figure()
        plt.rcParams.update({'font.size': 15})
        bin_txt = fig.text(0.5, 0.82, 'Target bin number: ')
        distance_txt = fig.text(0.5, 0.85, 'Target distance: ')
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Raw Signal")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Amplitude")
        ax.set_ylim(0, 0.02)


        fig.canvas.mpl_connect('pick_event', onpick)

        frame = read_frame()
        line, = ax.plot(ax_x, frame)
        line2, = ax.plot(ax_x, frame, picker=True, pickradius=5)

        ani = FuncAnimation(fig, animate, interval=1)
        try:
            plt.show()
        except:
            print('Messages output finish!')
        return self.bin_index
    
    def plot_real_time(self):
        print(self.sample_time)
        dist_arange = np.arange(self.area_start, self.area_end, self.bin_length)
        
        def fft(data):
            def butter_bandpass(lowcut, highcut, fs, order=8):
                nyq = 0.5 * fs
                low = lowcut / nyq
                high = highcut / nyq
                sos = signal.butter(order, [low, high], analog=False, btype='band', output='sos')
                return sos
            
            def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
                sos = butter_bandpass(lowcut, highcut, fs, order=order)
                y = signal.sosfilt(sos, data)
                return y
            
            # Filter the data
            b, a = signal.butter(8, 0.08, 'lowpass')
            filteredData = signal.filtfilt(b, a, data)
            c, d = signal.butter(8, 0.01, 'highpass')
            filteredData = signal.filtfilt(c, d, data)
            # filteredData = butter_bandpass_filter(data, 0.2, 1.5, self.FPS) #second parameter is the lowcut frequency, third is the highcut frequency

            # Perform FFT on the filtered data
            fft_data = np.fft.rfft(filteredData)
            fft_data = np.abs(fft_data)
            RPM = np.where(fft_data == np.max(fft_data))[0][0]
            # Calculate breathing frequency
            breathing_frequency = RPM / self.sample_time

            frequency_axis = np.arange(0, (self.FPS / N) * ((N / 2) + 0.5), self.FPS / N) # +0.5 because of the arange function does not include the last number
            breathing_frequency = frequency_axis[RPM]

            hz_txt.set_text('Dominant frequency: {}'.format(round(breathing_frequency,2)))
            rpm_txt.set_text('RPM: {}'.format(str(RPM)))
            
            return fft_data
            
        #TODO: maybe refactor this to call it from the method.py
        def read_frame():
            """Gets frame data from module"""
            d = self.xep.read_message_data_float()
            frame = np.array(d.data)
            # Convert the resulting frame to a complex array if downconversion is enabled
            n = len(frame)
            # convert frame to complex
            frame = frame[:n // 2] + 1j * frame[n // 2:]
            frame = np.abs(frame)
            return frame

        def save(event):
            self.folder_name = str(datetime.datetime.now().minute) + str(datetime.datetime.now().second) + 'time%ds' % self.sample_time
            path = self.path + self.folder_name
            folder = os.path.exists(path)
            if not folder:
                os.mkdir(path)
                amp_file = path + '\\amp.txt'
                np.savetxt(amp_file, self.values[-N:])
                JSON_data = {
                "device_name": self.device_name,
                "fps": self.FPS,
                "iterations": self.iterations,
                "pulses_per_step": self.pulses_per_step,
                "sample_time": self.sample_time,
                "bin_index": int(self.bin_index),
                "distance(m)": round(dist_arange[self.bin_index]+self.bin_length, 2),
                "bin_length": self.bin_length,
                "area_start": self.area_start,
                "area_end": self.area_end,
                }
                m.write_json_data(JSON_data, path+"/param.json")
            else:
                print('error:the folder exists!!!')
            print("data collection finished")
       
        fig = plt.figure()
        raw_signal = fig.add_subplot(3, 1, 1) # 2 rows, 1 column, 1st plot
        fft_signal = fig.add_subplot(3, 1, 2)
        raw_dist = fig.add_subplot(3, 1, 3)

        fft_signal.set_title("FFT Signal")
        fft_signal.set_xlabel("Frequency (Hz)")
        fft_signal.set_ylabel("FFT Amplitude")

        raw_signal.set_title("Raw Signal")
        raw_signal.set_xlabel("Time (s)")
        raw_signal.set_ylabel("Amplitude")
        raw_signal.set_xlim(self.sample_time, 0)

        raw_dist.set_title("Distance")
        raw_dist.set_ylabel("Amplitude")
        raw_dist.set_xlabel("Distance (m)")

        bin_txt = fig.text(0.01, 0.97,'Target bin number: ', fontsize=12)
        dis_txt = fig.text(0.01, 0.94, 'Distance to target: ', fontsize=12)
        hz_txt = fig.text(0.01, 0.91, 'Dominant frequency: ', fontsize=12)
        rpm_txt = fig.text(0.01, 0.88, 'RPM: ', fontsize=12)
        plt.subplots_adjust(left=0.17, right= 0.97,hspace=0.5)
        raw_signal.set_ylim(0, 0.03)
        fft_signal.set_ylim(0, 0.7)
        # fft_signal.set_xlim(0, 1.7)
                
        save_button_place = fig.add_axes([0.01, 0.05, 0.1, 0.075])
        save_button = Button(save_button_place, 'Save')
        save_button.on_clicked(save)


        N = self.sample_time * self.FPS # Number of samples in the signal
        self.values = [0] * N
        time_axis = np.arange((self.FPS*self.sample_time)) / self.FPS
        time_axis = time_axis[::-1]
        print(time_axis)
        print(N)
        frequency_axis = np.arange(0, (self.FPS / N) * ((N / 2) + 0.5), self.FPS / N) # +0.5 because of the arange function does not include the last number
        raw_data, = raw_signal.plot(time_axis, self.values)
        fft_line, = fft_signal.plot(frequency_axis, fft(self.values))
        
        frame = read_frame()
        ax_x = np.arange((self.area_start-1e-5), (self.area_end-1e-5)+self.bin_length, self.bin_length)
        reflection_line, = raw_dist.plot(ax_x, frame)
        peaks, _ = find_peaks(frame, height=0)

        points, = raw_dist.plot(ax_x[peaks], frame[peaks], "o", picker=True, pickradius=5)

        def onpick(event):
            thisline = event.artist
            xdata = thisline.get_xdata()
            ind = event.ind
            self.bin_index = np.where(ax_x == xdata[ind])[0][0]
            self.values = [0] * N

        fig.canvas.mpl_connect('pick_event', onpick)


        def animate(i):
            frame = read_frame()
            peaks, _ = find_peaks(frame, height=0)
            # self.bin_index = np.argmax(frame)
            bin_txt.set_text('Target bin number: {}'.format(str(self.bin_index)))
            dis_txt.set_text('Distance to target: ~{} m'.format(round(dist_arange[self.bin_index]+self.bin_length,2)))

            self.values.append(frame[self.bin_index])
            self.values = self.values[-N:]
            raw_signal.set_ylim(np.min((self.values))/1.2, np.max((self.values))*1.2)
            fft_signal.set_ylim(0, np.max(fft(self.values))*1.2)
            raw_dist.set_ylim(0, np.max(frame)*1.2)
            # fft_signal.set_xlim(0, frequency_axis)
            raw_data.set_ydata(self.values)
            reflection_line.set_ydata(frame)
            points.set_data(ax_x[peaks], frame[peaks])
            fft_line.set_ydata(fft(self.values))




        ani = FuncAnimation(fig, animate, interval=1)
        try:
            plt.show()
        except:
            print('Messages output finish!')
        return self.folder_name
