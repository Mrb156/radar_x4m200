import plot_data
from read_radar_data import X4m200_reader
import methods as m
from pymoduleconnector.extras.auto import auto
from time import sleep

D = 0.95 # duty cycle
iterations = 16 # recommended to use 64
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


sample_time = 120 # in seconds

fast_sample_point = int((area_end - area_start)/bin_length + 2) # number of sample points in the fast time domain

def run_radar():
    # Run the radar and save the data to a file

    device_name = auto()[0]


    reader = X4m200_reader(device_name, FPS, iterations, pulses_per_step, dac_min, dac_max, area_start, area_end, sample_time)
    bin_index = 32
    # bin_index = reader.plot_radar_raw_data_message()
    print(bin_index)
    #FIXME: because of the countdown the real timer jumps with 3 seconds
    m.countdown(3)
    
    amp_matrix_path = reader.get_data_matrix(bin_index)


    folder = amp_matrix_path
    return folder

def run_realtime():
    device_name = auto()[0]
    reader = X4m200_reader(device_name, FPS, iterations, pulses_per_step, dac_min, dac_max, area_start, area_end, sample_time)
    reader.plot_real_time()

# run_realtime()
folder = "3152time120s"
folder = run_radar()
plot_data.perform_fft_analysis(folder)

