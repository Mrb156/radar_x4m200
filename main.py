import plot_data
from read_radar_data import X4m200_reader
import methods as m
from pymoduleconnector.extras.auto import auto
from time import sleep
#TODO: realtime baseband data plot for distance measurement

D = 0.95 # duty cycle
iterations = 300 # recommended to use 64
dac_min = 949
dac_max = 1100
bin_length = 8*1.5e8/23.328e9 # first value is the speed of light, second is the X4 sampling rate
area_start = 0.4 # start of the sensing area in meters
area_end = 5 # end of the sensing area in meters
prf = 15.1875 # pulse repetition frequency (Mhz)
# FPS = m.calc_max_fps(prf, D, dac_min, dac_max, iterations, pulses_per_step) # calculate the FPS based on the parameters
FPS = 17 # calculate the FPS based on the parameters
pulses_per_step = m.calc_pulses_per_step(prf, D, dac_min, dac_max, iterations, FPS) # adjust this value to get the desired FPS
print(f"Pulses per step value: {pulses_per_step}")


sample_time = 60 # in seconds

fast_sample_point = int((area_end - area_start)/bin_length + 2) # number of sample points in the fast time domain


# device_name = auto()[0]

m.countdown()

# reader = X4m200_reader(device_name, FPS, iterations, pulses_per_step, dac_min, dac_max, area_start, area_end, sample_time)
# amp_matrix_path = reader.get_data_matrix()
# print(amp_matrix_path)
# reader.plot_frame(a, b, sample_time)

# folder = amp_matrix_path
folder = "133time60s"
plot_data.perform_fft_analysis(folder, FPS)