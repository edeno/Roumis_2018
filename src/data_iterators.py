import numpy as np
import pandas as pd


def chunk_mark_timeseries(timeseries_df, ntrodes, segments_times, position,
                          resample_time='1ms'):
    '''row per segment

    Parameters
    ----------
    timeseries_df : list of pandas.DataFrame's (one per source, i.e. ntrode; each df requires common time index, NaN empty time rows)
    ntrodes : list of ntrodes to include
    segments_times : pandas.DataFrame
    position : pandas.DataFrame
    resample_time : str

    Returns
    -------
    segments_marks : pandas.DataFrame
    COLUMN_NAMES = ['segment_ID', 'segment_type', 'area', 'start_time',
                    'end_time', 'spikes', 'ntrodes', 'linear_position', 'times']
    '''
    COLUMN_NAMES = ['segment_ID', 'segment_type', 'area', 'start_time',
                    'end_time', 'spikes', 'ntrodes', 'linear_position', 'times']
    segments_marks = pd.DataFrame(columns=COLUMN_NAMES)

    segments_marks['spikes'] = [
        pd.concat([
            (timeseries_df[introde][seg['start_time']:seg['end_time']]
            .resample(resample_time).mean()).dropna()
            for introde, ntrode in enumerate(ntrodes)], keys=ntrodes)
        for segind, seg in segments_times.iterrows()
    ]

    segments_marks['linear_position'] = chunk_position_timeseries(
        position, resample_time, segments_times)

    segments_marks['segment_ID'] = segments_times.index
    segments_marks['start_time'] = segments_times['start_time']
    segments_marks['end_time'] = segments_times['end_time']
    return segments_marks

def chunk_position_timeseries(position, resample_time, segments_times):
    position_chunks = [
        position[seg['start_time']:seg['end_time']]
        .resample(resample_time).interpolate(method='linear')
        for segind, seg in segments_times.iterrows()
        ]
    return position_chunks

def minute_linspaced_epoch_times(animals, epoch_key):
    from loren_frank_data_processing import get_trial_time
    '''linspaced times to chunk timeseries

    Parameters
    ----------
    num_of_chunks : int (the number of chunks)
    animals = dict
    epoch_key : tuple e.g. (str(animal), int(day), int(epoch))

    Returns
    -------
    segment_times : pandas.DataFrame (start_time, end_time columns)

    '''

    epoch_time = get_trial_time(epoch_key, animals)
    epoch_minutes = pd.DataFrame(index=epoch_time).resample('1T').min()
    segments_times = pd.DataFrame(columns=['start_time', 'end_time'])
    segments_times['start_time'] = epoch_minutes.iloc[:-1].index
    segments_times['end_time'] = epoch_minutes.iloc[1:].index

    return segments_times
