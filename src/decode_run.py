import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             mean_squared_error, median_absolute_error)
from sklearn.model_selection import LeaveOneGroupOut

from loren_frank_data_processing import (get_interpolated_position_dataframe,
                                         make_tetrode_dataframe)
from replay_classification import ClusterlessDecoder
from replay_classification.decoders import _DEFAULT_STATE_NAMES
from src.analysis import (_BRAIN_AREAS, _MARKS,
                          _get_multiunit_indicator_dataframe)


def load_data(epoch_key, animals, sampling_frequency,
              position_info=None, resample='1ms',
              mark_names=_MARKS, brain_areas=_BRAIN_AREAS, correct_only=True):
    if position_info is None:
        position_info = get_interpolated_position_dataframe(epoch_key, animals)
    else:
        position_info = position_info.copy()

    position_info = position_info.dropna().resample(resample).bfill()
    position_info['lagged_linear_distance'] = (
        position_info.linear_distance.shift(1))
    is_correct = (position_info.is_correct if correct_only
                  else np.ones_like(position_info.is_correct)).astype(bool)
    KEEP_COLUMNS = ['linear_distance', 'lagged_linear_distance', 'task',
                    'is_correct', 'turn', 'speed', 'head_direction',
                    'labeled_segments']

    position_info = position_info.loc[is_correct, KEEP_COLUMNS].dropna()

    brain_areas = [brain_areas] if isinstance(
        brain_areas, str) else brain_areas

    tetrode_info = make_tetrode_dataframe(animals).xs(
        epoch_key, drop_level=False)

    is_brain_areas = tetrode_info.area.isin(brain_areas)
    brain_areas_tetrodes = tetrode_info[is_brain_areas]

    def _time_func(*args, **kwargs):
        return position_info.index

    if mark_names is None:
        # Use all available mark dimensions
        mark_names = _get_multiunit_indicator_dataframe(
            brain_areas_tetrodes.nchans.argmax(), animals).columns.tolist()
        mark_names = [mark_name for mark_name in mark_names
                      if mark_name not in ['x_position', 'y_position']]

    marks = []
    for tetrode_key in brain_areas_tetrodes.index:
        multiunit_df = _get_multiunit_indicator_dataframe(
            tetrode_key, animals, _time_func)
        if multiunit_df is not None:
            marks.append(multiunit_df.loc[:, mark_names])

    marks = np.stack(marks, axis=0)

    return position_info, marks


def train_position_model(position_info, marks, n_position_bins=61,
                         place_std_deviation=None,
                         mark_std_deviation=20,
                         confidence_threshold=0.8):

    linear_distance = position_info.linear_distance.values
    lagged_linear_distance = position_info.lagged_linear_distance.values

    return ClusterlessDecoder(
        position=linear_distance,
        state_transition_state_order=[1],
        observation_state_order=[1],
        state_names=['replay_position'],
        initial_conditions='Uniform',
        trajectory_direction=np.ones_like(linear_distance),
        lagged_position=lagged_linear_distance,
        spike_marks=marks,
        n_position_bins=n_position_bins,
        place_std_deviation=place_std_deviation,
        mark_std_deviation=mark_std_deviation,
        replay_speedup_factor=1,
        confidence_threshold=confidence_threshold,
    ).fit()


def test_position_model(decoder, position_info, marks):
    results = decoder.predict(marks).results
    max_ind = results.posterior_density.argmax('position')
    predicted_position = (results.position[max_ind].squeeze())
    true_position = position_info.linear_distance.values
    return (median_absolute_error(true_position, predicted_position),
            np.sqrt(mean_squared_error(true_position, predicted_position))
            )


def train_classification_model(position_info, marks, n_position_bins=61,
                               place_std_deviation=None,
                               mark_std_deviation=20,
                               confidence_threshold=0.8):
    linear_distance = position_info.linear_distance.values
    lagged_linear_distance = position_info.lagged_linear_distance.values
    task = position_info.task.values

    return ClusterlessDecoder(
        position=linear_distance,
        trajectory_direction=task,
        lagged_position=lagged_linear_distance,
        spike_marks=marks,
        state_names=['Inbound', 'Outbound'],
        observation_state_order=['Inbound', 'Outbound'],
        state_transition_state_order=['Inbound', 'Outbound'],
        n_position_bins=n_position_bins,
        place_std_deviation=place_std_deviation,
        mark_std_deviation=mark_std_deviation,
        replay_speedup_factor=1,
        confidence_threshold=confidence_threshold,
    ).fit()


def test_classification_model(decoder, position_info, marks):
    predicted_state = decoder.predict(marks).predicted_state()
    true_state = position_info.task.unique().item()
    return true_state, predicted_state


def validate_position_decode(epoch_key, animals, sampling_frequency,
                             position_info=None,
                             resample='1ms',
                             mark_names=_MARKS,
                             brain_areas=_BRAIN_AREAS,
                             correct_only=True):
    position_info, marks = load_data(epoch_key, animals, sampling_frequency,
                                     position_info=position_info,
                                     resample=resample,
                                     mark_names=mark_names,
                                     brain_areas=brain_areas,
                                     correct_only=correct_only)
    position_error = []
    groups = position_info.labeled_segments

    for train, test in LeaveOneGroupOut().split(position_info, groups=groups):
        decoder = train_position_model(
            position_info.iloc[train], marks[:, train, :])
        position_error.append(
            test_position_model(decoder, position_info.iloc[test],
                                marks[:, test, :]))

    cv_std_error = (np.std(position_error, axis=0) /
                    np.sqrt(len(position_error)))
    cv_mean = np.mean(position_error, axis=0)
    return pd.DataFrame(np.stack((cv_mean, cv_std_error), axis=1),
                        columns=['cv_mean', 'cv_std_error'],
                        index=['median_absolute_error', 'mean_squared_error'])


def validate_classification_decode(epoch_key, animals, sampling_frequency,
                                   position_info=None, resample='1ms',
                                   mark_names=_MARKS, brain_areas=_BRAIN_AREAS,
                                   correct_only=True):
    position_info, marks = load_data(epoch_key, animals, sampling_frequency,
                                     position_info=position_info,
                                     resample=resample,
                                     mark_names=mark_names,
                                     brain_areas=brain_areas,
                                     correct_only=correct_only)
    decoded_labels = []
    groups = position_info.labeled_segments

    for train, test in LeaveOneGroupOut().split(position_info, groups=groups):
        decoder = train_classification_model(
            position_info.iloc[train], marks[:, train, :])
        decoded_labels.append(
            test_classification_model(decoder, position_info.iloc[test],
                                      marks[:, test, :]))

    decoded_labels = np.array(decoded_labels)
    return (
        accuracy_score(decoded_labels[:, 0], decoded_labels[:, 1]),
        confusion_matrix(decoded_labels[:, 0], decoded_labels[:, 1],
                         labels=['Inbound', 'Outbound']),
    )
