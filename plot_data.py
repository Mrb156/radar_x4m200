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
    return int(folderName[start + 1:end])

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
    slice_start_time = 0
    slice_end_time = sample_time
    data_matrix = np.loadtxt(data_folder_path)
    data_matrix = slice_datamatrix(data_matrix, slice_start_time, slice_end_time, fps)
    data_matrix[:, :4] = 0  # Zero out first 4 columns of the data

    sample_time = sample_time - slice_start_time

    range_bin_data = data_matrix[:, 32]
    breath_rate = []

    N = sample_time * fps

    # Define Butterworth bandpass filter functions
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        sos = signal.butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        filtered_data = signal.sosfilt(sos, data)
        return filtered_data

    # Apply low-pass and high-pass filters to the range bin data
    lowpass_filtered_data = butter_bandpass_filter(range_bin_data, 0.08, 0.34, fps, order=8)
    highpass_filtered_data = butter_bandpass_filter(lowpass_filtered_data, 0.01, 0.34, fps, order=8)

    # Calculate breath rate from the filtered data
    for i in range(1):
        data_segment = highpass_filtered_data[i * fps:(i + 1) * fps]
        doublediff = np.diff(np.sign(np.diff(data_segment)))
        peak_locations = np.where(doublediff == -2)[0] + 1
        breath_rate.append(peak_locations.shape[0] * 2)
    

    ###############################
    # Calculate breath rate from the filtered data
    for i in range(int(sample_time / 30)):
        data_segment = highpass_filtered_data[i * 30 * fps:(i + 1) * 30 * fps]
        doublediff = np.diff(np.sign(np.diff(data_segment)))
        peak_locations = np.where(doublediff == -2)[0] + 1
        breath_rate.append(peak_locations.shape[0] * 2)

    # Calculate the average breathing rate over the entire sample
    avg_breath_rate = np.mean(breath_rate)
    # Calculate breathing frequency
    breathing_frequency = avg_breath_rate / sample_time
    ###############################
    print("breathing fr: ", breathing_frequency)
    print(avg_breath_rate)
    # Perform FFT on the filtered data
    fft_data = np.fft.rfft(highpass_filtered_data)
    max_amplitude = np.max(np.abs(fft_data))
    indices = abs(fft_data) >= max_amplitude
    filtered_fft_data = fft_data * indices
    ifft_data = np.fft.ifft(filtered_fft_data)

    # Calculate RPM and peak location
    max_amplitude_freq_index = np.where(np.abs(filtered_fft_data) == np.max(np.abs(filtered_fft_data)))[0][0]
    RPM = max_amplitude_freq_index / sample_minutes

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
