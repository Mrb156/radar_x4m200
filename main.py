import plot_data
from read_radar_data import X4m200_reader
import methods as m
from pymoduleconnector.extras.auto import auto
from argparse import ArgumentParser

#TODO: store noise map in a file and use it to remove noise from the data
#TODO: run the different functionalities from the command line
#TODO: add an init file for the parameters, where the user can set the parameters
#TODO: setup file for the project
#TODO: rename the line variable names to be more specific

D = 0.95 # duty cycle
iterations = 16 # recommended to use 64
dac_min = 949
dac_max = 1100
bin_length = 8*1.5e8/23.328e9 # first value is the speed of light, second is the X4 sampling rate
area_start = 0.4 # start of the sensing area in meters
area_end = 3 # end of the sensing area in meters
prf = 15.1875 # pulse repetition frequency (Mhz)
# FPS = m.calc_max_fps(prf, D, dac_min, dac_max, iterations, pulses_per_step) # calculate the FPS based on the parameters
FPS = 17 # frames per second
pulses_per_step = m.calc_pulses_per_step(prf, D, dac_min, dac_max, iterations, FPS) # calculate the pulses per step based on the parameters
sample_time = 60 # in seconds

# fast_sample_point = int((area_end - area_start)/bin_length + 2) # number of sample points in the fast time domain

def run_measure():
    # Run the radar and save the data to a file
    device_name = auto()[0]

    reader = X4m200_reader(device_name, FPS, iterations, pulses_per_step, dac_min, dac_max, area_start, area_end, sample_time)
    bin_index = reader.plot_radar_raw_data_message()
    
    amp_matrix_path = reader.get_data_matrix(bin_index)

    folder = amp_matrix_path
    return folder

def run_realtime():
    device_name = auto()[0]
    reader = X4m200_reader(device_name, FPS, iterations, pulses_per_step, dac_min, dac_max, area_start, area_end, sample_time)
    reader.plot_real_time()

run_realtime()
# # folder = "758time18000s"
# folder = run_radar()
# plot_data.perform_fft_analysis(folder)

def main():
    parser = ArgumentParser()
    parser.add_argument("--method", type=str, required=True, help="Method to run")
    parser.add_argument("--folder", type=str, help="Folder for perform_fft_analysis method")
    parser.add_argument("--area_end", type=int, default=area_end, help="Area end")
    parser.add_argument("--sample_time", type=int, default=sample_time, help="Sample time")
    args = parser.parse_args()

    if args.method == "run_realtime":
        run_realtime()
    elif args.method == "run_measure":
        folder = run_measure()
        plot_data.perform_fft_analysis(folder)
    elif args.method == "perform_fft_analysis":
        if args.folder is None:
            print("Please provide a folder for the perform_fft_analysis method")
        else:
            plot_data.perform_fft_analysis(args.folder)
    else:
        print(f"Unknown method: {args.method}")

if __name__ == "__main__":
    main()