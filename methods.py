from time import sleep

def calc_max_fps(prf, D, dac_min, dac_max, iterations, pulses_per_step):
    # Calculate the maximum frames per second (FPS) of a radar system based on the given parameters

    # Convert the pulse repetition frequency (PRF) to Hz
    prf_hz = prf * 1e6

    # Calculate the range of the digital-to-analog converter (DAC)
    dac_range = dac_max - (dac_min + 1)

    # Calculate the FPS of the radar system
    fps = prf_hz / (iterations * pulses_per_step * dac_range) * D

    return int(fps)

def calc_pulses_per_step(prf, D, dac_min, dac_max, iterations, FPS):
    prf_hz = prf * 1e6 # convert prf to Hz
    # Calculate the number of pulses per step
    pulses_per_step = (prf_hz * D) / ((dac_max - (dac_min + 1)) * iterations * FPS)

    return int(pulses_per_step)

def countdown():
    # countdown from 3 seconds
    for i in range(3, 0, -1):
        print(i)
        sleep(1)
    
    print("Measurement starting after the 2nd beep")