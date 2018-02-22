
import numpy as np
import pandas as pd




def create_segments_marks_df(marks_dataF, ntrodes, segments_times, position, resample_time='1ms'): # with row per segment
    segments_marks = pd.DataFrame(columns=['segment_ID', 'segment_type', 'area', 'start_time', 'end_time',
    	'marks', 'ntrodes', 'linear_position'])
    # segments_marks['marks'] = [np.stack([marks_dataF[tetind].loc[segment_start:segment_end].resample(resample_time).mean().values
    #                                      for tetind, tet in enumerate(ntrodes)], axis=0)
    #                            for segment_start, segment_end  in zip(segments_times['start_time'], segments_times['end_time'])]
    segments_marks['marks'] = [np.stack([
    	marks_dataF[ntrode-1][seg['start_time']:seg['end_time']].resample(resample_time).mean().index
    	for ntrode in ntrodes]
    	for segind, seg in segments_times.iterrows(), axis=0)]


    segments_marks['time'] = [marks_dataF[0][seg['start_time']:seg['end_time']].resample(resample_time).mean().index for segind, seg in segments_times.iterrows()]
    #bin times by period resample_time in ms "500ms" (via resample) 
    



    # segments_marks['time'] = [marks_dataF[0][segment_start:segment_end].resample(resample_time).mean().index 
    #                           for segment_start, segment_end  in zip(segments_times['start_time'], segments_times['end_time'])]

    segments_marks['segment_ID'] = segments_marks.index
    segments_marks['start_time'] = segments_times['start_time'] 
    segments_marks['end_time'] = segments_times['end_time']
    segments_marks['linear_position'] = [position[seg['start_time']:seg['end_time']].resample(resample_time).mean().index 
    for segind, seg in segments_times.iterrows()]
    return(segments_marks)


def create_segments_times_fromChunks(numberSegments, marks_df):# create times df with start and end for each segment when 
    startstamp = marks_dataF[0].index[0].value # get epoch start and end
    endstamp = marks_dataF[0].index[-1].value
    segments_times = pd.DataFrame(columns=['start_time', 'end_time'])
    segtimes = np.linspace(startstamp, endstamp, numberSegments+2, endpoint=True)
    segments_times['start_time'] = pd.TimedeltaIndex(segtimes[0:-2])
    segments_times['end_time'] = pd.TimedeltaIndex(segtimes[1:-1])
    return(segments_times)


def create_segments_times_fromEvents(event_times):# create times df from events
    event_segments_times = event_times
    return(event_segments_times)