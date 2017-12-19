# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: droumis, edeno
"""
import loren_frank_data_processing as lfdp
# import ripple_detection as ripdetect
import replay_classification as replay
from replay_classification.simulate import get_trajectory_direction
# import pandas as pd
from collections import namedtuple

Animal = namedtuple('Animal', {'short_name', 'directory', 'preprocessing_directory'})
animals = {
    'JZ1': Animal(short_name='JZ1',
                  directory='../Raw-Data/JZ1',
                  preprocessing_directory='../Raw-Data/JZ1')}
date = 20161114
epoch_index = ('JZ1', 1, 2)
tets = [21]
areas = ['ca1']

full_tetrode_info = lfdp.make_tetrode_dataframe(animals)

multiunit_data = [lfdp.get_multiunit_indicator_dataframe(tetindex, animals).values
                  for tetindex in full_tetrode_info.xs(epoch_index, drop_level=False).
                      query('area.isin(@areas) & tetrode_number.isin(@tets)').index]

position_variables = ['linear_distance', 'trajectory_direction',
                      'speed']

position_info = lfdp.get_interpolated_position_dataframe(epoch_index, animals)

train_position_info = position_info.query('speed > 4')

# marks = train_position_info.join(multiunit_data[0])


# get_trajectory_direction()
trajectory_direction, is_inbound = get_trajectory_direction(position_info.linear_distance)
# b = [0,trajectory_direction]

decoder = replay.ClusterlessDecoder(position=position_info.linear_distance,
                              trajectory_direction=trajectory_direction,
                              spike_marks=multiunit_data,
                              replay_speedup_factor=1)
decoder.fit()
decoder.plot_initial_conditions();
decoder.plot_state_transition_model();

decoder.plot_observation_model();

outbound_time = time < 0.5
outbound_results = decoder.predict(spike_marks[:, outbound_time, :], time=time[outbound_time])


g = outbound_results.plot_posterior_density()

for ax in g.axes.ravel().tolist():
    ax.plot(time[outbound_time], linear_distance[outbound_time], color='white', linestyle='--', linewidth=3, alpha=0.8)