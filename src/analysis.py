from logging import getLogger

import pandas as pd

from loren_frank_data_processing import (get_interpolated_position_dataframe,
                                         get_LFP_dataframe,
                                         make_tetrode_dataframe)
from ripple_detection import Kay_ripple_detector

logger = getLogger(__name__)

_MARKS = ['channel_1_max', 'channel_2_max', 'channel_3_max',
          'channel_4_max']
_BRAIN_AREAS = 'ca1'


def detect_epoch_ripples(epoch_key, animals, brain_areas=_BRAIN_AREAS,
                         zscore_threshold=3):
    '''
    '''
    logger.info('Detecting ripples')

    SAMPLING_FREQUENCY = 1000

    tetrode_info = make_tetrode_dataframe(animals).xs(
        epoch_key, drop_level=False)
    # Get cell-layer CA1, iCA1 LFPs

    brain_areas = [brain_areas] if isinstance(
        brain_areas, str) else brain_areas
    is_brain_areas = tetrode_info.area.isin(brain_areas)
    logger.debug(tetrode_info[is_brain_areas].loc[:, ['area']])
    tetrode_keys = tetrode_info[is_brain_areas].index.tolist()
    hippocampus_lfps = pd.concat(
        [get_LFP_dataframe(tetrode_key, animals)
         for tetrode_key in tetrode_keys], axis=1
    ).astype(float).resample('1ms').mean().dropna()
    time = hippocampus_lfps.index

    def _time_function(epoch_key, animals):
        return time

    speed = get_interpolated_position_dataframe(
        epoch_key, animals, _time_function).speed

    return Kay_ripple_detector(
        time, hippocampus_lfps.values, speed.values, SAMPLING_FREQUENCY,
        minimum_duration=pd.Timedelta(milliseconds=15),
        zscore_threshold=zscore_threshold)
