import numpy as np
import datetime
import os
import pymoduleconnector
import methods as m
from time import sleep
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#TODO: countdown timer for data collection
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
        self.fast_sample_point = int(
            (self.area_end - self.area_start)/self.bin_length + 2)
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
        phase_data = np.arctan2(q_vec, i_vec)
        return ampli_data, phase_data

    def get_data_matrix(self):
        row = self.sample_time * self.FPS
        col = self.fast_sample_point
        amp_matrix = np.empty([row, col])
        pha_matrix = np.empty([row, col])

        old_time = datetime.datetime.now()
        print(old_time)
        n = 0
        while n < row:
            new_time = datetime.datetime.now()
            interval = (new_time - old_time).microseconds
            if interval > 1/self.FPS*1000:
                old_time = new_time
                ampli_data, phase_data = self.read_apdata()
                amp_matrix[n] = ampli_data
                pha_matrix[n] = phase_data
                n += 1

       
        folder_name = str(new_time.minute) + str(new_time.second) + 'time%ds' % self.sample_time
        path = 'C:\Barna\sze/radar/radar_x4m200\meresek/' + folder_name
        folder = os.path.exists(path)
        if not folder:
            os.mkdir(path)
            filename1 = path + '/amp_matrix.txt'
            filename2 = path + '/pha_matrix.txt'
            np.savetxt(filename1, amp_matrix)
            np.savetxt(filename2, pha_matrix)
        else:
            print('error:the folder exists!!!')
        print("data collection finished")
        return folder_name
        
    
    def plot_frame(self, amp_matrix, pha_matrix, sample_time):

        ax_x = np.arange((self.area_start-1e-5), (self.area_end-1e-5)+self.bin_length, self.bin_length)

        fig = plt.figure()
        amp_fig = fig.add_subplot(2,1,1)
        pha_fig = fig.add_subplot(2,1,2)
        amp_fig.set_ylim(0, 0.015)

        amp_fig.set_title("Amplitude")
        pha_fig.set_title("Phase") 
        line1, = amp_fig.plot(ax_x, amp_matrix[0])
        line2, = pha_fig.plot(ax_x, pha_matrix[0])


        def animate(i):
            fig.suptitle("frame count:%d" % i)
            amplitude = amp_matrix[i]
            phase = pha_matrix[i]
            line1.set_ydata(amplitude)
            line2.set_ydata(phase)
            return line1,line2,

        ani = FuncAnimation(fig, animate, frames = sample_time*self.FPS, interval=1/self.FPS*1000)

        plt.show()
    

