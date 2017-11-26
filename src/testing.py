# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: droumis, edeno
"""
from loren_frank_data_processing import make_tetrode_dataframe
from collections import namedtuple
# from tqdm import tqdm_notebook as tqdm

Animal = namedtuple('Animal', {'short_name', 'directory', 'preprocessing_directory'})
animals = {
    'JZ1': Animal(short_name='JZ1',
                  directory='../Raw-Data/JZ1',
                  preprocessing_directory='../Raw-Data/JZ1')}
date = 20161114
epoch_index = ('JZ1', 1, 2)

tetrode_info = make_tetrode_dataframe(animals)



print('done')