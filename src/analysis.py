from logging import getLogger

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import linregress

from loren_frank_data_processing import (get_interpolated_position_dataframe,
                                         get_LFP_dataframe,
                                         get_multiunit_indicator_dataframe,
                                         make_tetrode_dataframe,
                                         reshape_to_segments)
from replay_classification import ClusterlessDecoder
from ripple_detection import Kay_ripple_detector

logger = getLogger(__name__)

_MARKS = ['channel_1_max', 'channel_2_max', 'channel_3_max',
          'channel_4_max']
_BRAIN_AREAS = 'ca1'


def detect_epoch_ripples(epoch_key, animals, brain_areas=_BRAIN_AREAS,
                         zscore_threshold=3, sampling_frequency=1500):
    '''
    '''
    logger.info('Detecting ripples')

    tetrode_info = make_tetrode_dataframe(animals).xs(
        epoch_key, drop_level=False)

    brain_areas = [brain_areas] if isinstance(
        brain_areas, str) else brain_areas
    is_brain_areas = tetrode_info.area.isin(brain_areas)
    logger.debug(tetrode_info[is_brain_areas].loc[:, ['area']])
    tetrode_keys = tetrode_info[is_brain_areas].index.tolist()
    hippocampus_lfps = pd.concat(
        [get_LFP_dataframe(tetrode_key, animals)
         for tetrode_key in tetrode_keys], axis=1
    )
    time = hippocampus_lfps.index

    def _time_function(epoch_key, animals):
        return time

    speed = get_interpolated_position_dataframe(
        epoch_key, animals, _time_function).speed

    return Kay_ripple_detector(
        time, hippocampus_lfps.values, speed.values, sampling_frequency,
        minimum_duration=pd.Timedelta(milliseconds=15),
        zscore_threshold=zscore_threshold)


def decode_ripple_clusterless(epoch_key, animals, ripple_times,
                              sampling_frequency=1500,
                              n_place_bins=61,
                              place_std_deviation=None,
                              mark_std_deviation=20,
                              mark_names=_MARKS,
                              brain_areas=_BRAIN_AREAS):
    logger.info('Decoding ripples')
    tetrode_info = make_tetrode_dataframe(animals).xs(
        epoch_key, drop_level=False)
    brain_areas = [brain_areas] if isinstance(
        brain_areas, str) else brain_areas
    is_brain_areas = tetrode_info.area.isin(brain_areas)
    brain_areas_tetrodes = tetrode_info[is_brain_areas]
    logger.debug(brain_areas_tetrodes.loc[:, ['area', 'depth', 'descrip']])

    position_info = get_interpolated_position_dataframe(
        epoch_key, animals, max_distance_from_well=5)

    if mark_names is None:
        # Use all available mark dimensions
        mark_names = get_multiunit_indicator_dataframe(
            brain_areas_tetrodes.index[0], animals).columns.tolist()
        mark_names = [mark_name for mark_name in mark_names
                      if mark_name not in ['x_position', 'y_position']]

    is_training = (position_info.speed > 4) & position_info.is_correct
    marks = [(get_multiunit_indicator_dataframe(tetrode_key, animals)
              .loc[:, mark_names])
             for tetrode_key in brain_areas_tetrodes.index]
    marks = [tetrode_marks for tetrode_marks in marks
             if (tetrode_marks.loc[is_training].dropna()
                 .shape[0]) != 0]

    train_position_info = position_info.loc[is_training]

    training_marks = np.stack([
        tetrode_marks.loc[train_position_info.index, mark_names]
        for tetrode_marks in marks], axis=0)

    decoder = ClusterlessDecoder(
        train_position_info.linear_distance.values,
        train_position_info.task.values,
        training_marks,
        replay_speedup_factor=16,
    ).fit()

    test_marks = _get_ripple_marks(marks, ripple_times, sampling_frequency)
    logger.info('Predicting replay types')
    results = [decoder.predict(ripple_marks, time.total_seconds())
               for ripple_marks, time in test_marks]

    return summarize_replay_results(
        results, ripple_times, position_info, epoch_key)


def summarize_replay_results(results, ripple_times, position_info,
                             epoch_key):
    '''Summary statistics for decoded replays.

    Parameters
    ----------
    posterior_density : list of arrays
    test_spikes : array_like
    ripple_times : list of tuples
    state_names : list of str
    position_info : pandas DataFrame

    Returns
    -------
    replay_info : pandas dataframe
    decision_state_probability : array_like
    posterior_density : xarray DataArray

    '''
    replay_info = ripple_times.copy()

    # Includes information about the animal, day, epoch in index
    (replay_info['animal'], replay_info['day'],
     replay_info['epoch']) = epoch_key
    replay_info = replay_info.reset_index()

    replay_info['ripple_duration'] = ((
        replay_info['end_time'] - replay_info['start_time']) /
        np.timedelta64(1, 's'))

    # Add decoded states and probability of state
    replay_info['predicted_state'] = [
        result.predicted_state() for result in results]
    replay_info['predicted_state_probability'] = [
        result.predicted_state_probability() for result in results]

    replay_info = pd.concat(
        (replay_info,
         replay_info.predicted_state.str.split('-', expand=True)
         .rename(columns={0: 'replay_task',
                          1: 'replay_order'})
         ), axis=1)

    # When in the session does the ripple occur (early, middle, late)
    replay_info['session_time'] = pd.Categorical(
        _ripple_session_time(ripple_times, position_info.index), ordered=True,
        categories=['early', 'middle', 'late'])

    # Add stats about spikes
    replay_info['number_of_unique_spiking'] = [
        _num_unique_spiking(result.spikes) for result in results]
    replay_info['number_of_spikes'] = [_num_total_spikes(result.spikes)
                                       for result in results]

    # Include animal position information
    replay_info = pd.concat(
        [replay_info,
         position_info.loc[replay_info.start_time]
         .set_index(replay_info.index)
         ], axis=1)

    # Determine whether ripple is heading towards or away from animal's
    # position
    posterior_density = xr.concat(
        [result.results.posterior_density for result in results],
        dim=replay_info.index)
    posterior_density['time'] = posterior_density.time.to_index()
    replay_info['replay_motion'] = _get_replay_motion(
        replay_info, posterior_density)

    decision_state_probability = xr.concat(
        [result.state_probability().unstack().to_xarray().rename(
            'decision_state_probability')
         for result in results], dim=replay_info.index)

    return (replay_info, decision_state_probability,
            posterior_density)


def _num_unique_spiking(spikes):
    '''Number of units that spike per ripple
    '''
    if spikes.ndim > 2:
        return np.sum(~np.isnan(spikes), axis=(1, 2)).nonzero()[0].size
    else:
        return spikes.sum(axis=0).nonzero()[0].size


def _num_total_spikes(spikes):
    '''Total number of spikes per ripple
    '''
    if spikes.ndim > 2:
        return np.any(~np.isnan(spikes), axis=2).sum()
    else:
        return int(spikes.sum())


def _ripple_session_time(ripple_times, session_time):
    '''Categorize the ripples by the time in the session in which they
    occur.

    This function trichotimizes the session time into early session,
    middle session, and late session and classifies the ripple by the most
    prevelant category.
    '''
    session_time_categories = pd.Series(
        pd.cut(
            session_time, 3,
            labels=['early', 'middle', 'late'], precision=4),
        index=session_time)
    return pd.Series(
        [(session_time_categories.loc[ripple_start:ripple_end]
          .value_counts().argmax())
         for ripple_start, ripple_end
         in ripple_times.itertuples(index=False)],
        index=ripple_times.index, name='session_time',
        dtype=session_time_categories.dtype)


def _get_replay_motion_from_rows(ripple_times, posterior_density,
                                 distance_measure='linear_distance'):
    '''

    Parameters
    ----------
    ripple_info : pandas dataframe row
    posterior_density : array, shape (n_time, n_position_bins)
    state_names : list of str, shape (n_states,)
    place_bin_centers : array (n_position_bins)

    Returns
    -------
    is_away : array of str

    '''
    posterior_density = posterior_density.sum('state').dropna('time')
    replay_position = posterior_density.position.values[
        posterior_density.argmax('position').values]
    animal_position = ripple_times[distance_measure]
    replay_distance_from_animal_position = np.abs(
        replay_position - animal_position)
    is_away = linregress(
        posterior_density.get_index('time').values,
        replay_distance_from_animal_position).slope > 0
    return np.where(is_away, 'Away', 'Towards')


def _get_replay_motion(ripple_times, posterior_density,
                       distance_measure='linear_distance'):
    '''Motion of the replay relative to the current position of the animal.
    '''
    return np.array(
        [_get_replay_motion_from_rows(row, density, distance_measure)
         for (_, row), density
         in zip(ripple_times.iterrows(), posterior_density)]).squeeze()


def _get_ripple_marks(marks, ripple_times, sampling_frequency):
    mark_ripples = [reshape_to_segments(
        tetrode_marks, ripple_times, sampling_frequency=sampling_frequency,
        axis=0)
        for tetrode_marks in marks]

    return [(np.stack([df.loc[ripple_number, :].values
                       for df in mark_ripples], axis=0),
             mark_ripples[0].loc[ripple_number, :]
             .index.get_level_values('time'))
            for ripple_number in ripple_times.index]
