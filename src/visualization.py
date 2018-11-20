from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LogNorm
from scipy.signal import convolve, gaussian

import holoviews as hv
from loren_frank_data_processing import (get_LFPs,
                                         get_multiunit_indicator_dataframe,
                                         get_spike_indicator_dataframe,
                                         reshape_to_segments)
from spectral_connectivity import Connectivity, Multitaper
from src.analysis import _center_time


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

    return ax, spikes, event_locked_spikes, kde


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
        window_spikes = spike_times[window_start_time: window_end_time]
        window_spikes.reset_index(inplace=True)
        spikes_Spikes[ievent] = hv.NdOverlay({nti: hv.Spikes(window_spikes[window_spikes.ntrode == nti].time, kdims='time', group='multiunit',
                                                             label=ntrode_df.xs(epoch_index).area[nti])
                                              .opts(plot=dict(position=nti))
                                              for nti in ntrodes})
    return spikes_Spikes


def get_ntrode_spikes_dmap(event_times, window_spikes, epoch_index, ntrode_df, window=.4, ntrode=1):
    '''per ntrode, event-triggered spike raster as Holoviews DynamicMap

    Parameters
    ----------
    ntrode_number : int, default 0 for initialization
    ntrodes : list of ints
    event_times : pd.DataFrame, columns=['start_time', 'end_time']
    windows_spikes : pd.DataFrame, index=timedelta labeled 'time'
    window : float

    Returns
    -------
    Spikes : Holoviews Spikes element

    '''
    # events = np.arange(1,event_times.shape[0]+1)
    # window_spikes = pd.DataFrame(pd.concat([(spike_times[ntrode].dropna()[(rv.start_time-timedelta(seconds=window)):(rv.start_time+timedelta(seconds=window))].reset_index()['time'] - rv.start_time).dt.total_seconds()
    #              for irip, rv in event_times.iterrows()
    #              ], keys=events, names=['event_number'])).reset_index()

    Spikes = {}
    for irip, ripvals in event_times.iterrows():
        Spikes[irip] = hv.Spikes(window_spikes[ntrode - 1][window_spikes[ntrode - 1].event_number == irip].time,
                                 kdims='time',  group='multiunit').opts(plot=dict(position=irip))

    Spikes_dmap = hv.NdOverlay(overlays=Spikes, kdims=['event_number'], group='{0} {1} {2} {3} {4}'
                               .format(epoch_index[0], epoch_index[1], epoch_index[2], ntrode,
                                       ntrode_df.xs(epoch_index).area[ntrode]))  # .opts(plot = dict(yticks = events))

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
            kdims='time',  group='multiunit', label=ntrode_df.xs(epoch_index).area[nti]).opts(plot=dict(position=nti))

    Spikes = hv.NdOverlay(overlays=Spikes, kdims=['event_number']).opts(
        plot=dict(yticks=ntrodes))

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
        xbox_width = (event_end_time.total_seconds() -
                      event_start_time.total_seconds())
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
    ybox_center = np.mean([min(ntrodes), max(ntrodes) + 1])
    xbox_width = event_end_time.total_seconds() - event_start_time.total_seconds()
    ybox_width = (max(ntrodes) - min(ntrodes)) + 1
    event_bounds = hv.Polygons(
        [hv.Box(xbox_center, ybox_center, (xbox_width, ybox_width))])
    return event_bounds


def _plot_distribution(
        ds, dims=None, quantiles=[0.025, 0.25, 0.5, 0.75, 0.975],
        **plot_kwargs):
    alphas = np.array(quantiles)
    alphas[alphas > 0.5] = 1 - alphas[alphas > 0.5]
    alphas = (alphas / 0.5)
    alphas[alphas < 0.2] = 0.2

    for q, alpha in zip(quantiles, alphas):
        ds.quantile(q, dims).plot.line(alpha=alpha, **plot_kwargs)


def plot_power(path, group, brain_area, frequency, figsize=(15, 10),
               vmin=0.5, vmax=2):

    def _preprocess(ds):
        return ds.sel(
            tetrode=ds.tetrode[ds.brain_area == brain_area],
            frequency=frequency
        )
    try:
        ds = xr.open_mfdataset(
            path, concat_dim='session', group=group,
            preprocess=_preprocess, autoclose=True).power.compute()
    except ValueError:
        return
    DIMS = ['session', 'tetrode']

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    _plot_distribution(
        ds.isel(time=0), dims=DIMS, ax=axes[0, 0], color='midnightblue')
    axes[0, 0].set_title('Baseline Power')

    _plot_distribution(
        ds.isel(time=0), dims=DIMS, ax=axes[1, 0], color='midnightblue')
    axes[1, 0].set_title('Baseline Power')

    ds.mean(DIMS).plot(x='time', y='frequency', ax=axes[0, 1])
    axes[0, 1].set_title('Raw power')

    _plot_distribution(
        ds.sel(time=0.0, method='backfill'), dims=DIMS,
        ax=axes[1, 1], color='midnightblue')
    axes[1, 1].set_title('Raw power after ripple')

    (ds / ds.isel(time=0)).mean(DIMS).plot(
        x='time', y='frequency', ax=axes[0, 2],
        norm=LogNorm(vmin=vmin, vmax=vmax), cmap='RdBu_r',
        vmin=vmin, vmax=vmax, center=0)
    axes[0, 2].set_title('Change from baseline power')

    _plot_distribution(
        (ds / ds.isel(time=0)).sel(time=0.0, method='backfill'), dims=DIMS,
        ax=axes[1, 2], color='midnightblue')
    axes[1, 2].set_title('Change after ripple')
    axes[1, 2].axhline(1, color='black', linestyle='--')
    axes[1, 2].set_ylim((vmin, vmax))

    for ax in axes[0, 1:3]:
        ax.axvline(0, color='black', linestyle='--')

    plt.tight_layout()
    plt.suptitle(brain_area, fontsize=18, fontweight='bold')
    plt.subplots_adjust(top=0.90)


def plot_position_data(position_info, highlight, highlight_label):
    fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

    axes[0].plot(position_info.index.total_seconds(),
                 position_info.linear_distance)

    for label, df in position_info.groupby('task'):
        axes[0].scatter(df.index.total_seconds(),
                        df.linear_distance, label=label)

    axes[0].fill_between(
        position_info.index.total_seconds(),
        highlight * position_info.linear_distance.max(),
        alpha=0.3, color='orange', label=highlight_label)
    axes[0].set_ylabel('Distance from center well (cm)', fontsize=18)
    axes[0].legend()

    axes[1].fill_between(
        position_info.index.total_seconds(),
        highlight * position_info.speed.max(),
        alpha=0.3, color='orange', label=highlight_label)
    axes[1].plot(position_info.index.total_seconds(), position_info.speed)
    axes[1].set_ylabel('Speed', fontsize=18)
    axes[1].axhline(4, color='red', linestyle='--')
    axes[1].set_xlabel('Time (seconds)', fontsize=18)
    plt.tight_layout()


def plot_all_perievent_raster(tetrode_info, events, animals, col_wrap=5,
                              window_offset=(-0.5, 0.5)):
    tetrode_keys = tetrode_info.index
    n_tetrodes = len(tetrode_keys)

    n_rows = np.ceil(n_tetrodes / col_wrap).astype(int)

    fig, axes = plt.subplots(n_rows, col_wrap,
                             figsize=(col_wrap * 3, n_rows * 3),
                             sharex=True, sharey=True)
    for ax, tetrode_key in zip(axes.ravel(), tetrode_keys):
        try:
            plot_perievent_raster(tetrode_key, animals, events,
                                  tetrode_info, window_offset=window_offset,
                                  ax=ax, s=1)
        except IndexError:
            brain_area = tetrode_info.loc[tetrode_key[:4]].area.upper()
            tetrode_id = tetrode_info.loc[tetrode_key[:4]].tetrode_id
            ax.set_title(f'{brain_area} - {tetrode_id}')

    for ax in axes.ravel()[n_tetrodes:]:
        ax.axis('off')
    plt.tight_layout()


def plot_power_change(tetrode_info, events, animals, sampling_frequency,
                      multitaper_params=None, window_offset=(-0.5, 0.5),
                      vmin=0.5, vmax=2.0, col_wrap=5):
    lfps = get_LFPs(tetrode_info.index, animals)
    ripple_locked_lfps = reshape_to_segments(
        lfps, events, window_offset=window_offset,
        sampling_frequency=sampling_frequency)
    ripple_locked_lfps = (ripple_locked_lfps.to_xarray().to_array()
                          .rename({'variable': 'tetrode'})
                          .transpose('time', 'ripple_number', 'tetrode')
                          .dropna('ripple_number'))
    ripple_locked_lfps = (ripple_locked_lfps
                          - ripple_locked_lfps.mean(['ripple_number']))
    start_time = ripple_locked_lfps.time.min().values / np.timedelta64(1, 's')

    m = Multitaper(ripple_locked_lfps.values, **multitaper_params,
                   start_time=start_time)
    c = Connectivity.from_multitaper(m)
    dimension_names = ['time', 'frequency', 'tetrode']
    data_vars = {
        'power': (dimension_names, c.power())}
    coordinates = {
        'time': _center_time(c.time),
        'frequency': c.frequencies + np.diff(c.frequencies)[0] / 2,
        'tetrode': tetrode_info.tetrode_id.values,
        'brain_area': ('tetrode', tetrode_info.area.tolist()),
    }
    power = xr.Dataset(data_vars, coords=coordinates).power
    g = (power / power.isel(time=0)).sel(frequency=slice(0, 300)).plot(
        x='time', y='frequency', col='tetrode', col_wrap=col_wrap,
        norm=LogNorm(vmin=vmin, vmax=vmax), cmap='RdBu_r',
        vmin=vmin, vmax=vmax, center=0)

    for ax, (_, t) in zip(g.axes.flat, tetrode_info.iterrows()):
        ax.axvline(0, linestyle='--', color='black')
        ax.set_title(f'{t.area.upper()} - {t.tetrode_id}')


def plot_blah():
    position_info = get_interpolated_position_dataframe(epoch_key, ANIMALS)
    ripple_times_CA1 = detect_epoch_ripples(
    epoch_key, ANIMALS, SAMPLING_FREQUENCY,
    position_info=position_info, brain_areas='ca1')


    time = position_info.index
    times = [hse_times_CA1, ripple_times_CA1, hse_times_MEC, ripple_times_MEC]
    times_names = ['hse_times_CA1', 'ripple_times_CA1', 'hse_times_MEC', 'ripple_times_MEC']
    data = [MUA_CA1, ripple_power_change_CA1, MUA_MEC, ripple_power_change_MEC]
    data_names = ['MUA_CA1', 'ripple_power_change_CA1', 'MUA_MEC', 'ripple_power_change_MEC']

    fig, axes = plt.subplots(4, 4, figsize=(12, 12), sharex=True)

    for ax, (datum, times), (datum_name, times_name) in zip(axes.flat, product(data, times), product(data_names, times_names)):
        if isinstance(datum, np.ndarray):
            datum = pd.DataFrame(datum, index=position_info.index.rename('time'))
        elif isinstance(datum, xr.DataArray):
            datum = datum.to_dataframe()
            datum = datum.set_index(pd.TimedeltaIndex(datum.index, unit='s', name='time'))
            time_index = np.digitize(datum.index.total_seconds(),
                                     time.total_seconds())
            time_index[time_index >= len(time)] = len(time) - 1
            datum = (datum.groupby(time[time_index]).mean()
                     .reindex(index=time).interpolate())

        time_locked_datum = reshape_to_segments(
            datum, times, window_offset=(-0.5, 0.5), sampling_frequency=SAMPLING_FREQUENCY)
        t = time_locked_datum.unstack(level=0).index.total_seconds()
        average = np.squeeze(time_locked_datum.mean(level='time').values)
        n_events = time_locked_datum.unstack().shape[0]
        sem = np.squeeze(time_locked_datum.std(level='time').values / np.sqrt(n_events))
        ci_lo, ci_hi = average - sem, average + sem
        ax.plot(t, average, linewidth=2)
        ax.fill_between(t, ci_lo, ci_hi, alpha=0.3)
        ax.axvline(0, color='black', linestyle='--', zorder=-1)
        ax.axhline(np.nanmean(datum), color='black', linestyle='--', zorder=-1)
        ax.set_ylabel(datum_name)
        ax.set_title(times_name)
        ax.set_ylim(np.nanpercentile(datum, [1, 99]))

    plt.tight_layout()
