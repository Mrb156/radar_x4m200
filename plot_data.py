import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal, fftpack

# TODO: Add comprehensive documentation and comments throughout the code

def slice_datamatrix(amp_datamatrix, start_time, end_time, fps):
    """
    Slice a data matrix based on specified start and end times and frames per second (FPS).

    Parameters:
    - amp_datamatrix (numpy.ndarray): The input data matrix.
    - start_time (int): Start time in seconds.
    - end_time (int): End time in seconds.
    - fps (int): Frames per second.

    Returns:
    - numpy.ndarray: The sliced data matrix.
    """
    return amp_datamatrix[start_time * fps:end_time * fps, :]

def findTime(folderName):
    """
    Extract the time duration from a folder name.

    Parameters:
    - folderName (str): The folder name containing time information.

    Returns:
    - int: The extracted time duration in seconds.
    """
    start = folderName.find("e")
    end = folderName.find("s")
    time = int(folderName[start + 1:end])
    print(time)
    return time

def perform_fft_analysis(folder, fps):
    """
    Perform FFT analysis on radar data stored in a specified folder.

    Parameters:
    - folder (str): The folder path containing radar data.

    Returns:
    - None
    """
    data_folder_path = r"C:\Barna\sze\radar\radar_x4m200/meresek/" + str(folder) + "/amp_matrix.txt"
    fps = fps
    sample_time = findTime(folder)  # Duration of data collection in seconds
    sample_minutes = sample_time / 60  # Duration of data collection in minutes
    data_matrix = np.loadtxt(data_folder_path)  # Load data matrix from file
    ###########################################################
    # do I really need this line?
    # data_matrix[:, :4] = 0  # Zero out first 4 columns of the data
    ###########################################################
    range_bin_data = data_matrix[:, 32]  # Extract range bin data from the data matrix
    # it is the 24th column of the data matrix
    # the code should determine the distance between the measured person and the radar and than
    # get the proper column from the datamatrix (help: radar user guide from application notes)

    N = sample_time * fps # Number of samples in the signal 

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
    filteredData = signal.filtfilt(b, a, range_bin_data)
    c, d = signal.butter(8, 0.01, 'highpass') 
    filteredData = signal.filtfilt(c, d, range_bin_data)
    # filteredData = butter_bandpass_filter(range_bin_data, 1, 1.5, fps) #second parameter is the lowcut frequency, third is the highcut frequency

    # Perform FFT on the filtered data
    fft_data = np.fft.rfft(filteredData)
    max_power = np.max(np.abs(fft_data))
    indices = abs(fft_data) >= max_power
    filtered_fft_data = fft_data * indices
    ifft_data = np.fft.ifft(filtered_fft_data)

    # Calculate RPM and peak location
    fd_data2 = abs(fft_data)*2/N
    RPM = np.where(fd_data2 == np.max(fd_data2))[0][0]
    # max_amplitude_freq_index = np.where(np.abs(filtered_fft_data) == np.max(np.abs(filtered_fft_data)))[0][0]
    # RPM = max_amplitude_freq_index / sample_minutes

    # Calculate breathing frequency
    breathing_frequency = RPM / sample_time

    # Create subplots for the signals
    fig = plt.figure()
    raw_signal = fig.add_subplot(3, 1, 1)
    fft_signal = fig.add_subplot(3, 1, 2)
    ifft_signal = fig.add_subplot(3, 1, 3)

    # Define x-axis for the signals
    time_axis = np.arange((fps*sample_time) * sample_minutes) / fps
    frequency_axis = np.arange(0, (fps / N) * ((N / 2) + 1), fps / N)
    ifft_time_axis = np.arange((fps*30) * sample_minutes + 1) / 8.5

    # Set titles and labels for the subplots
    raw_signal.set_title("Original Signal")
    raw_signal.set_xlabel("Time (s)")
    raw_signal.set_ylabel("Amplitude")

    fft_signal.set_title("FFT Signal")
    fft_signal.set_xlabel("Frequency (Hz)\nBreathing Frequency: ~{}Hz\nRPM: {}".format(breathing_frequency, RPM))
    fft_signal.set_ylabel("FFT Amplitude")

    ifft_signal.set_title("Reconstructed Component(s)")
    ifft_signal.set_xlabel("Time (s)")
    ifft_signal.set_ylabel("Amplitude")

    # Plot the signals
    line, = raw_signal.plot(time_axis, range_bin_data)
    line2, = fft_signal.plot(frequency_axis, abs(fft_data))
    line3, = ifft_signal.plot(ifft_time_axis, ifft_data)

    # Display the plots
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.97, top=0.9, hspace=1)
    plt.show()

# Example usage:
# perform_fft_analysis("your_folder_name_here")
