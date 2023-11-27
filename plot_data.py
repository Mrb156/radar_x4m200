import dis
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import ifft, signal, fftpack
import methods as m

# TODO: if the sample time is more than 60 seconds, than have the option to analize every 60 seconds
# and plot the results in a subplot
# and compare the results


def perform_fft_analysis(folder, path):
    """
    Perform FFT analysis on radar data stored in a specified folder.

    Parameters:
    - folder (str): The folder path containing radar data.

    Returns:
    - None
    """
    matrix = False
    data_folder_path = path + str(folder) + "/amp.txt"
    if not os.path.isfile(data_folder_path):
        data_folder_path = path + str(folder) + "/amp_matrix.txt"
        matrix = True

    fps = int(m.read_json_data(folder, path)["fps"])
    sample_time = int(m.read_json_data(folder, path)["sample_time"])  # Duration of data collection in seconds
    sample_minutes = sample_time / 60  # Duration of data collection in minutes
    data = np.loadtxt(data_folder_path)  # Load data matrix from file
    bin_index = int(m.read_json_data(folder, path)["bin_index"])
    bin_length = float(m.read_json_data(folder, path)["bin_length"])
    area_start = float(m.read_json_data(folder, path)["area_start"])
    area_end = float(m.read_json_data(folder, path)["area_end"])
    dist_arange = np.arange(area_start, area_end, bin_length)

    try:
        distance = m.read_json_data(folder, path)["distance(m)"]
    except KeyError:
        distance = (dist_arange[bin_index]+bin_length)
    ###########################################################
    # if sample_time and we want to slice the data matrix
    # time_slice_start = 2600
    # time_slice_end = 2960
    # sample_time = time_slice_end - time_slice_start
    # sample_minutes = sample_time / 60
    # data = data[time_slice_start*fps:time_slice_end*fps, :]
    ###########################################################
    if matrix:
        data = data[:,bin_index]  # Extract bin data from the data matrix
    # it is the bin_indexth column of the data matrix
    # print("data shape: ", data.shape)
    N = sample_time * fps # Number of samples in the signal 
    # print("N: ", N)
    # Define bandpass filter
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
    # filteredData = butter_bandpass_filter(data, 1, 1.5, fps) #second parameter is the lowcut frequency, third is the highcut frequency

    # Perform FFT on the filtered data
    fft_data = np.fft.rfft(filteredData)
    max_power = np.max(np.abs(fft_data))
    indices = abs(fft_data) >= max_power
    filtered_fft_data = fft_data * indices
    ifft_data = np.fft.ifft(filtered_fft_data)

    # Calculate RPM and peak location
    # fd_data2 = abs(fft_data)*2/N
    # RPM = np.where(fd_data2 == np.max(fd_data2))[0][0]
    max_amplitude_freq_index = np.where(np.abs(filtered_fft_data) == np.max(np.abs(filtered_fft_data)))[0][0]
    RPM = round(max_amplitude_freq_index / sample_minutes, 1)

    # Calculate breathing frequency
    breathing_frequency = round(RPM / sample_minutes, 2)

    # Create subplots for the signals
    fig = plt.figure()
    raw_signal = fig.add_subplot(3, 1, 1)
    fft_signal = fig.add_subplot(3, 1, 2)
    ifft_signal = fig.add_subplot(3, 1, 3)

    # Define x-axis for the signals
    time_axis = np.arange((fps*sample_time)) / fps
    # time_axis = np.arange((fps*sample_time) * sample_minutes) / fps
    frequency_axis = np.arange(0, (fps / N) * ((N / 2) + 0.5), fps / N)
    ifft_time_axis = np.arange((fps*30) * sample_minutes + 1) / 8.5

    # Set titles and labels for the subplots
    raw_signal.set_title("Original Signal")
    raw_signal.set_xlabel("Time (s)\nBin Index: {}\nDistance to person: {} m".format(bin_index, round(distance,2)))
    raw_signal.set_ylabel("Amplitude")

    fft_signal.set_title("FFT Signal")
    fft_signal.set_xlabel("Frequency (Hz)\nBreathing Frequency: ~{}Hz\nRPM: {}".format(breathing_frequency, RPM))
    fft_signal.set_ylabel("FFT Amplitude")

    ifft_signal.set_title("Reconstructed Component(s)")
    ifft_signal.set_xlabel("Time (s)")
    ifft_signal.set_xlim(0, 60)
    ifft_signal.set_ylabel("Amplitude")

    # Plot the signals
    line, = raw_signal.plot(time_axis, data)
    line2, = fft_signal.plot(frequency_axis, abs(fft_data))
    line3, = ifft_signal.plot(ifft_time_axis, ifft_data)

    # Display the plots
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.97, top=0.9, hspace=1)
    plt.show()

# Example usage:
# perform_fft_analysis("your_folder_name_here")
