from collections import namedtuple
from os.path import join, abspath, dirname, pardir

# LFP sampling frequency
SAMPLING_FREQUENCY = 1500

# Data directories and definitions
ROOT_DIR = join(abspath(dirname(__file__)), pardir)
RAW_DATA_DIR = join(ROOT_DIR, 'Raw-Data')
PROCESSED_DATA_DIR = join(ROOT_DIR, 'Processed-Data')

Animal = namedtuple('Animal', {'directory', 'short_name'})
