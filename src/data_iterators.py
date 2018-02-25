import numpy as np
import pandas as pd


def create_segments_marks_df(marks_df, ntrodes, segments_times, position,
                             resample_time='1ms'):
    '''With row per segment

    Parameters
    ----------
    marks_df : pandas.DataFrame
    ntrodes : int??
    segment_times : pandas.DataFrame
    position : pandas.DataFrame
    resample_time : str

    Returns
    -------
    segments_marks : pandas.DataFrame

    '''
    COLUMN_NAMES = ['segment_ID', 'segment_type', 'area', 'start_time',
                    'end_time', 'marks', 'ntrodes', 'linear_position']
    segments_marks = pd.DataFrame(columns=COLUMN_NAMES)

    segments_marks['marks'] = [np.stack(
        [(marks_df[ntrode - 1][seg['start_time']:seg['end_time']]
          .resample(resample_time).mean().index)
         for ntrode in ntrodes]
        for segind, seg in segments_times.iterrows(), axis=0)]

    segments_marks['time'] = [
        marks_df[0][seg['start_time']:seg['end_time']].resample(
            resample_time).mean().index
        for segind, seg in segments_times.iterrows()]
    # bin times by period resample_time in ms "500ms" (via resample)

    segments_marks['segment_ID'] = segments_marks.index
    segments_marks['start_time'] = segments_times['start_time']
    segments_marks['end_time'] = segments_times['end_time']
    segments_marks['linear_position'] = [
        position[seg['start_time']:seg['end_time']
                 ].resample(resample_time).mean().index
        for segind, seg in segments_times.iterrows()]

    return segments_marks


def chunk_timeseries(num_of_chunks, timeseries_df):
    '''linspaced chunks of timeseries

    Parameters
    ----------
    num_of_chunks : int (the number of chunks)
    timeseries_df : pandas.DataFrame. (requires time index)

    Returns
    -------
    segment_times : pandas.DataFrame (start_time, end_time columns)

    '''
    startstamp = timeseries_df[0].index[0].value  # get epoch start and end
    endstamp = timeseries_df[0].index[-1].value
    segments_times = pd.DataFrame(columns=['start_time', 'end_time'])
    segtimes = np.linspace(startstamp, endstamp,
                           num_of_chunks + 2, endpoint=True)
    segments_times['start_time'] = pd.TimedeltaIndex(segtimes[0:-2])
    segments_times['end_time'] = pd.TimedeltaIndex(segtimes[1:-1])
    return segments_times


def create_segments_times_fromEvents(event_times):
    '''Create times df from events'''
    event_segments_times = event_times
    return event_segments_times
