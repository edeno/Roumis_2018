from os.path import abspath, dirname, join, pardir

from loren_frank_data_processing import Animal

# LFP sampling frequency
SAMPLING_FREQUENCY = 1500

# Data directories and definitions
ROOT_DIR = join(abspath(dirname(__file__)), pardir)
RAW_DATA_DIR = join(ROOT_DIR, 'Raw-Data')
PROCESSED_DATA_DIR = join(ROOT_DIR, 'Processed-Data')

ANIMALS = {
    'JZ1': Animal(directory=join(RAW_DATA_DIR, 'JZ1'), short_name='JZ1'),
}

_12Hz_Res = dict(
    sampling_frequency=SAMPLING_FREQUENCY,
    time_window_duration=0.250,
    time_window_step=0.250,
    time_halfbandwidth_product=3,
)

MULTITAPER_PARAMETERS = {
    '12Hz_Resolution': _12Hz_Res,
}

REPLAY_COVARIATES = ['session_time', 'replay_task',
                     'replay_order', 'replay_motion']

FREQUENCY_BANDS = {
    'theta': (4, 12),
    'beta': (12, 30),
    'slow_gamma': (30, 60),
    'mid_gamma': (60, 100),
    'fast_gamma': (100, 125),
    'ripple': (150, 250)
}
