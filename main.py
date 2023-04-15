import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from time import sleep
import datetime
import os

import pymoduleconnector
from pymoduleconnector import DataType

from scipy import signal, fftpack


device_name = 'COM7'

FPS = 17
iterations = 16
pulses_per_step = 300
dac_min = 949
dac_max = 1100
area_start = 0.4
area_end = 5

sample_time = 60
slice_time_start = 0
slice_time_end = sample_time
bin_length = 8*1.5e8/23.328e9

fast_sample_point = int(
    (area_end - area_start)/bin_length + 2)


mc = pymoduleconnector.ModuleConnector(device_name)
app = mc.get_x4m200()
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

xep = mc.get_xep()
# # Set DAC range
xep.x4driver_set_dac_min(900)
xep.x4driver_set_dac_max(1150)

# Set integration
xep.x4driver_set_iterations(16)
xep.x4driver_set_pulses_per_step(26)

# xep.x4driver_set_downconversion(int(baseband))
# Start streaming of data
xep.x4driver_set_fps(10)

source = r"C:\Users\Morvai Barna\OneDrive - Széchenyi István Egyetem\TDK\!xethru\ujlog\1917_sampletime60s"
amp_matrix = source+"/amp_matrix.txt"
pha_matrix = source+"/pha_matrix.txt"
obj_amp_datamatrix = np.loadtxt(amp_matrix)
obj_pha_datamatrix = np.loadtxt(pha_matrix)

pha = np.array(obj_pha_datamatrix)
amp = np.array(obj_amp_datamatrix)


def slice_datamatrix(amp_datamatrix, start, end, FPS):
    return amp_datamatrix[start*FPS:end*FPS, :]


obj_amp_datamatrix = slice_datamatrix(
    obj_amp_datamatrix, slice_time_start, slice_time_end, FPS)

obj_amp_datamatrix[:, :4] = 0
sample_time = sample_time - slice_time_start


def reset(device_name):
    mc = pymoduleconnector.ModuleConnector(device_name)
    xep = mc.get_xep()
    xep.module_reset()
    mc.close()
    sleep(3)
    print("reseted")


def clear_buffer(mc):
    """Clears the frame buffer"""
    xep = mc.get_xep()
    while xep.peek_message_data_float():
        xep.read_message_data_float()
        print("clearing buffer")
    print("buffer cleared")


def read_apdata():
    # read a frame
    # data = xep.read_message_radar_baseband_float().get_I()
    data = xep.read_message_data_float().data
    #data = np.array(d.data)
    data_length = len(data)

    i_vec = np.array(data[:data_length//2])
    q_vec = np.array(data[data_length//2:])
    iq_vec = i_vec + 1j*q_vec
    # print(iq_vec)
    ampli_data = abs(iq_vec)
    phase_data = np.arctan2(q_vec, i_vec)
    return ampli_data, phase_data

def get_data_matrix(sample_time, save=False):
    row = sample_time * FPS
    col = fast_sample_point
    amp_matrix = np.empty([row, col])
    pha_matrix = np.empty([row, col])

    old_time = datetime.datetime.now()
    print(old_time)
    n = 0
    new_time = old_time
    while n < row:
        new_time = datetime.datetime.now()
        interval = (new_time - old_time).microseconds
        if interval > 1/17*1000:
            old_time = new_time
            ampli_data, phase_data = read_apdata()
            amp_matrix[n] = ampli_data
            pha_matrix[n] = phase_data
            n += 1

    if save:
        path = r'C:\Users\Morvai Barna\OneDrive - Széchenyi István Egyetem\TDK\!xethru\ujlog' + str(new_time.minute) + \
            str(new_time.second) + '_sampletime%ds' % sample_time
        folder = os.path.exists(path)
        if not folder:
            os.mkdir(path)
            filename1 = path + '/amp_matrix.txt'
            filename2 = path + '/pha_matrix.txt'
            np.savetxt(filename1, amp_matrix)
            np.savetxt(filename2, pha_matrix)
            print("done")
        else:
            print('error:the folder exists!!!')

    return amp_matrix, pha_matrix


def plot_frame(amp_matrix, pha_matrix, sample_time):
    # amp = np.array(amp_matrix)
    # pha = np.array(pha_matrix)


    ax_x = np.arange((area_start-1e-5),
                     (area_end-1e-5)+bin_length, bin_length)

    fig = plt.figure()
    amp_fig = fig.add_subplot(2, 1, 1)
    pha_fig = fig.add_subplot(2, 1, 2)
    amp_fig.set_ylim(0, 0.015)

    amp_fig.set_title("Amplitude")
    pha_fig.set_title("Phase")
    line1, = amp_fig.plot(ax_x, amp[0])
    line2, = pha_fig.plot(ax_x, pha[0])

    def animate(i):
        fig.suptitle("frame count:%d" % i)
        amplitude = amp_matrix[i]
        phase = pha_matrix[i]
        line1.set_ydata(amplitude)
        line2.set_ydata(phase)
        return line1, line2,

    ani = FuncAnimation(fig, animate, frames=sample_time *
                        FPS, interval=1/FPS*1000)

    plt.show()


def lowpass_filter(amp_datamatrix, bin_num, FPS, sampletime):
    rangebin_data = amp_datamatrix[:, bin_num]

    b, a = signal.butter(8, 0.08, 'lowpass')  # 配置滤波器 8 表示滤波器的阶数,a=2*Wc/Ws
    filtedData = signal.filtfilt(b, a, rangebin_data)  # data为要过滤的信号
    c, d = signal.butter(8, 0.01, 'highpass')
    filtedData = signal.filtfilt(c, d, filtedData)

    fd_data = np.fft.rfft(filtedData)

    fig = plt.figure()
    raw_signal = fig.add_subplot(3, 1, 1)
    bandpass_signal = fig.add_subplot(3, 1, 2)
    fft_signal = fig.add_subplot(3, 1, 3)

    fs = FPS
    N = fs*sampletime
    ax_x = np.arange(0, 60, 1/17)
    ax_x2 = np.arange(0, (fs/N)*((N/2)+1)*60, fs/N*60)

    raw_signal.set_title("Raw signal")
    raw_signal.set_xlabel("Time/s")
    raw_signal.set_ylabel("Amplitude")

    bandpass_signal.set_title("BPF signal")
    bandpass_signal.set_xlabel("Time/s")
    bandpass_signal.set_ylabel("Amplitude")

    fft_signal.set_title("FFT signal")
    fft_signal.set_xlabel("BPM")
    fft_signal.set_ylabel("Amplitude")
    fft_signal.set_xlim(0, 50)

    line1, = raw_signal.plot(ax_x, rangebin_data)
    line2, = bandpass_signal.plot(ax_x, filtedData)
    line3, = fft_signal.plot(ax_x2, abs(fd_data)*2/N)

    fd_data = abs(fd_data)*2/N
    print(np.where(fd_data == np.max(fd_data))[0][0])

    plt.show()
    #fig.savefig('test.png', dpi=600)


def FFT_fasttime(amp_datamatrix, bin_num, FPS, sampletime):
    win_length = 30
    rangebin_data = amp_datamatrix[:, 32]

    b, a = signal.butter(8, 0.08, 'lowpass')
    filtedData = signal.filtfilt(b, a, rangebin_data)
    c, d = signal.butter(8, 0.01, 'highpass')
    filtedData = signal.filtfilt(c, d, rangebin_data)

    fs = FPS
    N = sampletime*FPS
    sig_win = np.multiply(
        rangebin_data[170:win_length*FPS+170], np.hamming(win_length*FPS))
    #sig_win = rangebin_data[170:]
    fd_data = np.fft.rfft(filtedData)
    # print(abs(fd_data))

    PSD = fd_data * np.conj(fd_data) / N
    #PSD = abs(fd_data) ** 2
    freq = (1/(fs*N)) * np.arange(N)

    #freq = fftpack.fftfreq(rangebin_data.size, d=17/70)

    L = np.arange(1, np.floor(N/2), dtype='int')
    # indices = abs(fd_data) < 0.02
    # indices2 = indices > 0.015

    #PSD = PSD > 1
    #fd_data = fd_data > 0
    #indices = indices < 4
    # print(indices)

    fhat = fd_data  # * indices2
    # print(fhat)
    # cut_f_signal[(freq < 0.5)] = 0
    # print(fhat)
    fid = np.fft.ifft(fhat)

    fig = plt.figure()
    fft_signal = fig.add_subplot(2, 1, 1)
    ifft_signal = fig.add_subplot(2, 1, 2)

    ax_x2 = np.arange(0, (fs/N)*((N/2)+1), fs/N)

    fft_signal.set_title("FFT signal")
    fft_signal.set_xlabel("Freq/Hz")
    fft_signal.set_ylabel("Amplitude")

    line3, = fft_signal.plot(ax_x2, abs(fd_data))
    # fft_signal.set_xlim(1, 100)
    line, = ifft_signal.plot(fid)
    #line2, = ifft_signal.plot(fid.imag)
    plt.show()


def find_peakline(amp_datamatrix):
    """
    功能：跟踪目标范围段雷达的峰值，绘制图像
    """
    peakline_matrix = np.zeros(1020)
    peakline_matrix[0] = np.max(amp_datamatrix[0])
    index = np.where(amp_datamatrix[0] == np.max(amp_datamatrix[0]))[0][0]

    for i in range(1, 1020):
        #print(index)
        slice_matrix = amp_datamatrix[i][index-1:index+2]
        peakline_matrix[i] = np.max(slice_matrix)
        index = np.where(amp_datamatrix[i] == np.max(slice_matrix))[0][0]
    
    print(peakline_matrix[0])

    b, a = signal.butter(8, 0.08, 'lowpass')  # 配置滤波器 8 表示滤波器的阶数,a=2*Wc/Ws
    c, d = signal.butter(8, 0.015, 'highpass')
    filtedData = signal.filtfilt(b, a, peakline_matrix)  # data为要过滤的信号
    peakline_matrix = signal.filtfilt(c, d, filtedData)

    fig = plt.figure()
    raw_signal = fig.add_subplot(1, 1, 1)
    ax_x = np.arange(0, 60, 1/17)
    line1, = raw_signal.plot(ax_x, peakline_matrix)
    print(index)

    plt.show()


def Fft(amp_datamatrix):
    rangebin_data = amp_datamatrix[:, 32]
    breathrate = []

    N = sample_time*17

    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = signal.butter(order, [low, high],
                            analog=False, btype='band', output='sos')
        return sos

    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.sosfilt(sos, data)
        return y

    # b, a = signal.butter(10, 0.08, 'lowpass')
    # filtedData = signal.filtfilt(b, a, rangebin_data)
    c, d = signal.butter(3, 0.01, 'highpass')
    filtedData = signal.filtfilt(c, d, rangebin_data)
    filtedData = butter_bandpass_filter(rangebin_data, 1, 1.5 , FPS)

    for i in range(30):
        filtered_matrix = filtedData[i*17:(i+30)*17]
        doublediff = np.diff(np.sign(np.diff(filtered_matrix)))
        peak_locations = np.where(doublediff == -2)[0] + 1
        breathrate.append(peak_locations.shape[0]*2)
    # print(len(breathrate))
    #plt.semilogx(w, 20 * np.log10(abs(h)))

    fd_data = np.fft.rfft(filtedData)


    maxpower = np.max(np.abs(fd_data))
    indices = abs(fd_data) >= maxpower

    fhat = fd_data * indices

    fid = np.fft.ifft(fhat)
    fig = plt.figure()

    fs = FPS
    N = fs*sample_time

    RPM = np.where(abs(fd_data) == maxpower)[0][0]

    breathingFreq = RPM * fs / N

    raw_signal = fig.add_subplot(3, 1, 1)
    fft_signal = fig.add_subplot(3, 1, 2)
    ifft_signal = fig.add_subplot(3, 1, 3)
    # BPM_signal = fig.add_subplot(4, 1, 4)

    ax_x = np.arange(1020)/17
    ax_x2 = np.arange(0, (17/N)*((N/2)+1), 17/N)
    ax_x3 = np.arange(511)/8.5
    # ax_x4 = np.arange(0, (fs/N)*((N/2)+1)*60, fs/N*60)

    raw_signal.set_title("Eredeti jel")
    raw_signal.set_xlabel("Idő (s)")
    raw_signal.set_ylabel("Amplitudó")

    fft_signal.set_title("FFT jel")
    fft_signal.set_xlabel("Freq(Hz)\nLégzési frekvencia: ~{}Hz\nRPM: {}".format(breathingFreq, RPM))
    fft_signal.set_ylabel("FFT amplitudó")
    # fft_signal.set_xlim(1.1, 1.3)

    ifft_signal.set_title("Visszaállított komponens(ek)")
    ifft_signal.set_xlabel("Idő (s)")
    ifft_signal.set_ylabel("Amplitudó")

    

    # BPM_signal.set_title("RPM signal")
    # BPM_signal.set_xlabel("RPM: {}".format(RPM))
    # BPM_signal.set_ylabel("Magasság")
    # BPM_signal.set_xlim(0, 50)

    line, = raw_signal.plot(ax_x, rangebin_data)
    line2, = fft_signal.plot(ax_x2, abs(fd_data))
    line3, = ifft_signal.plot(ax_x3, fid)
    # line4, = BPM_signal.plot(ax_x4, abs(fd_data)*2/N)

    # ifft_signal.set_xlim(0, 60)
    # line2, = ifft_signal.plot(fid.imag)
    plt.subplots_adjust(left= 0.1, bottom=0.1, right=0.97, top=0.9, hspace=1)
    plt.show()


def main():
    # clear_buffer(mc)
    # reset(device_name)
    # get_data_matrix(sample_time, save=True)

    #FFT_fasttime(obj_amp_datamatrix, 32, FPS, sample_time)
    #find_peakline(obj_amp_datamatrix)
    Fft(obj_amp_datamatrix)
    # lowpass_filter(obj_amp_datamatrix, 32, FPS, sample_time)

    #plot_frame(obj_amp_datamatrix, obj_pha_datamatrix, sample_time)


if __name__ == "__main__":
    main()
