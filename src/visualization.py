import matplotlib.pyplot as plt
import numpy as np
import holoviews as hv
from scipy.signal import convolve, gaussian

from loren_frank_data_processing import (get_multiunit_indicator_dataframe,
                                         get_spike_indicator_dataframe,
                                         reshape_to_segments)
from datetime import timedelta


def plot_perievent_raster(neuron_or_tetrode_key, animals, events, tetrode_info,
                          window_offset=(-0.5, 0.5),
                          sampling_frequency=1500, ax=None,
                          **scatter_kwargs):
    '''Plot spike raster relative to an event.

    Parameters
    ----------
    neuron_or_tetrode_key : tuple
    animals : dict of namedtuples
    events : pandas DataFrame, shape (n_events, 2)
    tetrode_info : pandas DataFrame, shape (n_tetrodes, ...)
    window_offset : tuple, optional
    sampling_frequency : tuple, optional
    ax : matplotlib axes, optional
    scatter_kwargs : dict

    Returns
    -------
    ax : matplotlib axes

    '''
    if ax is None:
        ax = plt.gca()
    try:
        spikes = get_spike_indicator_dataframe(neuron_or_tetrode_key, animals)
    except ValueError:
        spikes = ((get_multiunit_indicator_dataframe(
            neuron_or_tetrode_key, animals) > 0).sum(axis=1) > 0) * 1.0
    event_locked_spikes = reshape_to_segments(
        spikes, events, window_offset=window_offset,
        sampling_frequency=sampling_frequency).unstack(level=0).fillna(0)
    time = event_locked_spikes.index.total_seconds()
    spike_index, event_number = np.nonzero(event_locked_spikes.values)

    ax.scatter(time[spike_index], event_number, **scatter_kwargs)
    ax.axvline(0.0, color='black')
    ax.set_title(
        tetrode_info.loc[neuron_or_tetrode_key[:4]].area.upper() + ' - ' +
        str(neuron_or_tetrode_key))
    ax.set_ylabel(events.index.name)
    ax.set_xlabel('time (seconds)')
    ax.set_ylim((0, events.index.max() + 1))
    ax.set_xlim(window_offset)

    ax2 = ax.twinx()
    kde = kernel_density_estimate(
        event_locked_spikes, sampling_frequency, sigma=0.025)
    m = ax2.plot(time, kde[:, 1], color='blue', alpha=0.8)
    ax2.fill_between(time, kde[:, 0], kde[:, 2],
                     color=m[0].get_color(), alpha=0.2)
    ax2.set_ylabel('Firing Rate (spikes / s)')

    return ax


def kernel_density_estimate(
        is_spike, sampling_frequency, sigma=0.025):
    '''The gaussian-smoothed kernel density estimate of firing rate over
    trials.

    Parameters
    ----------
    is_spike : ndarray, shape (n_time, n_trials)
    sampling_frequency : float
    bandwidth : float

    Returns
    -------
    firing_rate : ndarray, shape (n_time,)

    '''
    bandwidth = sigma * sampling_frequency
    n_window_samples = int(bandwidth * 8)
    kernel = gaussian(n_window_samples, bandwidth)[:, np.newaxis]
    density_estimate = convolve(
        is_spike, kernel, mode='same') / kernel.sum()
    n_events = density_estimate.shape[1]
    firing_rate = np.nanmean(density_estimate, axis=1,
                             keepdims=True) * sampling_frequency
    firing_rate_std = (np.nanstd(density_estimate, axis=1, keepdims=True) *
                       sampling_frequency / np.sqrt(n_events))
    ci = np.array([-1.96, 0, 1.96])
    return firing_rate + firing_rate_std * ci


def get_event_spikes_hmaps(event_list, ntrodes, event_times, spike_times, ntrode_df, window=.5):
    '''per event, event-triggered spike raster as Holoviews HoloMap

    Parameters
    ----------
    event_list : list of ints
    ntrodes : list of ints
    event_times : pd.DataFrame, columns=['start_time', 'end_time']
    spike_times : pd.DataFrame, index=timedelta labeled 'time'
    window : float

    Returns
    -------
    spikes_Spikes : Holoviews Spikes element

    '''
    spikes_Spikes = {}
    for ievent, event_number in enumerate(event_list):
        event_start_time = event_times.iloc[event_number].start_time
        event_end_time = event_times.iloc[event_number].end_time
        window_start_time = event_start_time - timedelta(seconds=window)
        window_end_time = event_start_time + timedelta(seconds=window)
        window_spikes = spike_times[ window_start_time: window_end_time ]
        window_spikes.reset_index( inplace=True )
        spikes_Spikes[ ievent ] = hv.NdOverlay({nti: hv.Spikes(window_spikes[ window_spikes.ntrode == nti ].time, kdims='time', group='multiunit',
                     label=ntrode_df.xs(epoch_index).area[nti])
      .opts(plot=dict(position=nti))
      for nti in ntrodes})
    return spikes_Spikes


def get_ntrode_spikes_dmap(event_times, spike_times, window=.4 ntrode=1):
    import pandas as pd
    '''per ntrode, event-triggered spike raster as Holoviews DynamicMap

    Parameters
    ----------
    ntrode_number : int, default 0 for initialization
    ntrodes : list of ints
    event_times : pd.DataFrame, columns=['start_time', 'end_time']
    spike_times : pd.DataFrame, index=timedelta labeled 'time'
    window : float

    Returns
    -------
    Spikes : Holoviews Spikes element

    '''
    events = np.arange(1,event_times.shape[0]+1)
    window_spikes = pd.DataFrame(pd.concat([(spike_times[ntrode].dropna()[(rv.start_time-timedelta(seconds=window)):(rv.start_time+timedelta(seconds=window))].reset_index()['time'] - rv.start_time).dt.total_seconds()
                 for irip, rv in event_times.iterrows()
                 ], keys=events, names=['event_number'])).reset_index()

    Spikes = {}
    for irip, ripvals in event_times.iterrows():
        Spikes[irip] = hv.Spikes(window_spikes[window_spikes.event_number == irip].time,
         kdims = 'time',  group = 'SWR-trig_multi-unit').opts(plot = dict(position = irip))

    Spikes_dmap = hv.NdOverlay(overlays=Spikes, kdims=['event_number']).opts(plot = dict(yticks = events))

    return Spikes_dmap

def get_event_spikes_dmap(ntrodes, epoch_index, event_times, spike_times, ntrode_df, window=.5, event_number=0):
    '''per event, event-triggered spike raster as Holoviews DynamicMap

    Parameters
    ----------
    event_number : int, default 0 for initialization
    ntrodes : list of ints
    event_times : pd.DataFrame, columns=['start_time', 'end_time']
    spike_times : pd.DataFrame, index=timedelta labeled 'time'
    window : float

    Returns
    -------
    Spikes : Holoviews Spikes element

    '''
    event_start_time = event_times.iloc[event_number].start_time
    event_end_time = event_times.iloc[event_number].end_time
    window_start_time = event_start_time - timedelta(seconds=window)
    window_end_time = event_start_time + timedelta(seconds=window)
    window_spikes = spike_times[window_start_time:window_end_time]
    window_spikes.index = window_spikes.index.total_seconds()
    window_spikes.reset_index(inplace=True)
    Spikes = {}
    for nti in ntrodes:
        Spikes[nti] = hv.Spikes(
        window_spikes[window_spikes.ntrode == nti].time,
         kdims = 'time',  group = 'multiunit', label = ntrode_df.xs(epoch_index).area[nti]).opts(plot = dict(position = nti))

    Spikes = hv.NdOverlay(overlays=Spikes, kdims=['event_number']).opts(plot = dict(yticks = ntrodes))

    return Spikes


def get_event_bounds_hmaps(event_list, ntrodes, event_times):
    '''per event, event duration as Holoviews Box Polygon

    Parameters
    ----------
    event_list : list of ints
    ntrodes : list of ints
    event_times : pd.DataFrame, columns=['start_time', 'end_time']

    Returns
    -------
    event_bounds : Holoviews Box Polygon element

    '''
    event_bounds = {}
    for ievent, event_list in enumerate(event_number):
        event_start_time = event_times.iloc[event_number].start_time
        event_end_time = event_times.iloc[event_number].end_time
        xbox_center = np.mean(
            [event_start_time.total_seconds(), event_end_time.total_seconds()])
        ybox_center = np.mean([min(ntrodes), max(ntrodes)])
        xbox_width = event_end_time.total_seconds() - event_start_time.total_seconds()
        ybox_width = max(ntrodes) - min(ntrodes)
        event_bounds[ievent] = hv.Polygons(
            [hv.Box(xbox_center, ybox_center, (xbox_width, ybox_width))])
    return event_bounds


def get_event_bounds_dmap(ntrodes, event_times, event_number=0):
    '''per event, event duration as Holoviews Box Polygon

    Parameters
    ----------
    event_number : int, default 0 for initialization
    ntrodes : list of ints
    event_times : pd.DataFrame, columns=['start_time', 'end_time']

    Returns
    -------
    event_bounds : Holoviews Box Polygon element

    '''
    event_start_time = event_times.iloc[event_number].start_time
    event_end_time = event_times.iloc[event_number].end_time
    xbox_center = np.mean(
        [event_start_time.total_seconds(), event_end_time.total_seconds()])
    ybox_center = np.mean([min(ntrodes), max(ntrodes)+1])
    xbox_width = event_end_time.total_seconds() - event_start_time.total_seconds()
    ybox_width = (max(ntrodes) - min(ntrodes))+1
    event_bounds = hv.Polygons(
        [hv.Box(xbox_center, ybox_center, (xbox_width, ybox_width))])
    return event_bounds
