import json
from optparse import Values
import time
import numpy as np
import datetime
import os
import pymoduleconnector
from scipy import signal
import methods as m
from time import sleep
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#TODO: show a plot while the data collection is running
class X4m200_reader:
    def __init__(self, device_name, FPS, iterations, pulses_per_step, dac_min, dac_max,
                    area_start, area_end, sample_time):
        self.device_name = device_name
        self.FPS = FPS
        self.iterations = iterations
        self.pulses_per_step = pulses_per_step
        self.dac_min = dac_min
        self.dac_max = dac_max
        self.area_start = area_start
        self.area_end = area_end
        self.sample_time = sample_time
        self.bin_length = 8*1.5e8/23.328e9
        self.fast_sample_point = int((self.area_end - self.area_start)/self.bin_length + 2)
        self.bin_index = 0
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
        self.xep.x4driver_set_frame_area_offset(0.18)
        self.xep.x4driver_set_frame_area(self.area_start, self.area_end)
        self.xep.x4driver_set_fps(self.FPS)

    def read_apdata(self):
        #read a frame
        # data = xep.read_message_radar_baseband_float().get_I()
        data = self.xep.read_message_data_float().data
        data_length = len(data)

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
                    print(f"{seconds} second(s) passed")
       
        folder_name = str(new_time.minute) + str(new_time.second) + 'time%ds' % self.sample_time
        path = 'C:\\Barna\\sze\\radar\\radar_x4m200\\meresek\\' + folder_name
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
            }
            m.write_json_data(JSON_data, path+"/param.json")
        else:
            print('error:the folder exists!!!')
        print("data collection finished")
        return folder_name
        
    def plot_radar_raw_data_message(self):
        def read_frame():
            """Gets frame data from module"""
            d = self.xep.read_message_data_float()  # wait until get data
            frame = np.array(d.data)
            # print('frame length:' + str(len(frame)))
            # Convert the resulting frame to a complex array if downconversion is enabled
            n = len(frame)
            # convert frame to complex
            frame = frame[:n // 2] + 1j * frame[n // 2:]
            frame = np.abs(frame)
            max_bin = np.argmax(frame) # find the index of the max value
            num_bin = max_bin % self.xep.x4driver_get_frame_bin_count() # do I really need this line?
            self.bin_index = num_bin
            bin_txt.set_text('Target bin number: {} from {} bins'.format(str(self.bin_index), str(self.xep.x4driver_get_frame_bin_count())))
            #TODO: calculate the distance based on the bin number -> devide the frame area according to the number of bins
            range_resolution = 3e8 / (2 * 143e6) # how to get range resolution?
            distance = num_bin * 6.95 # not accurate, just for demo
            distance_txt.set_text('Target distance: {:.2f} cm'.format(distance))
            return frame

        def animate(i):
            line.set_ydata(read_frame())  # update the data
            line2.set_data([self.bin_index,self.bin_index], [0,0.02])


        fig = plt.figure()
        fig.suptitle("Radar Raw Data")
        # encrease the size of the text on the graph
        plt.rcParams.update({'font.size': 15})
        bin_txt = fig.text(0.5, 0.9, 'Target bin number: ')
        distance_txt = fig.text(0.5, 0.85, 'Target distance: ')
        ax = fig.add_subplot(1, 1, 1)
        def onpick(event):
            thisline = event.artist
            xdata = thisline.get_xdata()
            self.bin_index = xdata[0]
            plt.close()

        fig.canvas.mpl_connect('pick_event', onpick)

        frame = read_frame()   
        ax.set_ylim(0, 0.02)
        plt.xticks(range(0, len(frame), 5))
        
        line, = ax.plot(frame)
        line2, = ax.plot(frame, picker=True, pickradius=5)

        ani = FuncAnimation(fig, animate, interval=1)
        try:
            plt.show()
        except:
            print('Messages output finish!')
        return self.bin_index

    def plot_real_time(self):  
        #TODO: show the dominant frequency on the plot
        #TODO: show the raw data on the plot and pick the target bin
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
            # filteredData = butter_bandpass_filter(data, 0.05, 1.5, self.FPS) #second parameter is the lowcut frequency, third is the highcut frequency

            # Perform FFT on the filtered data
            fft_data = np.fft.rfft(filteredData)
            fft_data = np.abs(fft_data)
            fd_data2 = abs(fft_data)*2/N
            RPM = np.where(fd_data2 == np.max(fd_data2))[0][0]
            # Calculate breathing frequency
            breathing_frequency = RPM / self.sample_time
            hz_txt.set_text('Dominant frequency: {}'.format(str(breathing_frequency)))
            rpm_txt.set_text('RPM: {}'.format(str(RPM)))
            
            return fft_data
            
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

        fig = plt.figure()
        raw_signal = fig.add_subplot(2, 1, 1) # 2 rows, 1 column, 1st plot
        fft_signal = fig.add_subplot(2, 1, 2)

        fft_signal.set_title("FFT Signal")
        fft_signal.set_xlabel("Frequency (Hz)")
        fft_signal.set_ylabel("FFT Amplitude")

        bin_txt = fig.text(0.5, 0.9, 'Target bin number: ')
        hz_txt = fig.text(0.5, 0.85, 'Dominant frequency: ')
        rpm_txt = fig.text(0.5, 0.8, 'RPM: ')

        raw_signal.set_ylim(0, 0.03)
        fft_signal.set_ylim(0, 0.7)
        # fft_signal.set_xlim(0, 1.7)
        
        N = self.sample_time * self.FPS # Number of samples in the signal
        values = [0] * N
        #TODO: flip the time axis without flipping the data
        time_axis = np.arange((self.FPS*self.sample_time)) / self.FPS
        # time_axis = np.flip(time_axis)
        frequency_axis = np.arange(0, (self.FPS / N) * ((N / 2) + 1), self.FPS / N)
        line, = raw_signal.plot(time_axis, values)
        line2, = fft_signal.plot(frequency_axis, fft(values))


        def animate(i, values):
            frame = read_frame()
            bin_index = np.argmax(frame)
            bin_txt.set_text('Target bin number: {}'.format(str(bin_index)))
            values.append(frame[bin_index])
            values = values[-N:]
            raw_signal.set_ylim(np.min((values))/1.2, np.max((values))*1.2)
            fft_signal.set_ylim(0, np.max(fft(values))*1.2)
            # fft_signal.set_xlim(0, frequency_axis)
            line.set_ydata(values)  # update the data
            line2.set_ydata(fft(values))
            return line,


        ani = FuncAnimation(fig, animate, fargs=(values,), interval=1)
        try:
            plt.show()
        except:
            print('Messages output finish!')
