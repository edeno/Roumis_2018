import logging
from argparse import ArgumentParser
from signal import SIGUSR1, SIGUSR2, signal
from subprocess import PIPE, run
from sys import exit

from loren_frank_data_processing import (get_interpolated_position_dataframe,
                                         save_xarray)
from src.analysis import (decode_ripple_clusterless, detect_epoch_ripples,
                          estimate_lfp_ripple_connectivity)
from src.parameters import ANIMALS, PROCESSED_DATA_DIR, SAMPLING_FREQUENCY


def get_command_line_arguments():
    parser = ArgumentParser()
    parser.add_argument('Animal', type=str, help='Short name of animal')
    parser.add_argument('Day', type=int, help='Day of recording session')
    parser.add_argument('Epoch', type=int,
                        help='Epoch number of recording session')
    parser.add_argument(
        '-d', '--debug',
        help='More verbose output for debugging',
        action='store_const',
        dest='log_level',
        const=logging.DEBUG,
        default=logging.INFO,
    )
    return parser.parse_args()


def main():
    args = get_command_line_arguments()
    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=FORMAT, level=args.log_level)

    def _signal_handler(signal_code, frame):
        logging.error('***Process killed with signal {signal}***'.format(
            signal=signal_code))
        exit()

    for code in [SIGUSR1, SIGUSR2]:
        signal(code, _signal_handler)

    epoch_key = (args.Animal, args.Day, args.Epoch)
    logging.info(
        'Processing epoch: Animal {0}, Day {1}, Epoch #{2}...'.format(
            *epoch_key))
    git_hash = run(['git', 'rev-parse', 'HEAD'],
                   stdout=PIPE, universal_newlines=True).stdout
    logging.info('Git Hash: {git_hash}'.format(git_hash=git_hash.rstrip()))

    position_info = get_interpolated_position_dataframe(epoch_key, ANIMALS)
    ripple_times = detect_epoch_ripples(
        epoch_key, ANIMALS, SAMPLING_FREQUENCY, position_info)
    (replay_info_ca1, decision_state_probability_ca1,
     posterior_density_ca1) = decode_ripple_clusterless(
        epoch_key, ANIMALS, ripple_times, position_info=position_info,
        brain_areas='ca1')
    save_xarray(PROCESSED_DATA_DIR, epoch_key, replay_info_ca1.to_xarray(),
                '/replay_info_ca1')
    save_xarray(PROCESSED_DATA_DIR, epoch_key, posterior_density_ca1,
                '/posterior_density_ca1')

    (replay_info_mec, decision_state_probability_mec,
     posterior_density_mec) = decode_ripple_clusterless(
        epoch_key, ANIMALS, ripple_times, position_info=position_info,
        brain_areas='mec')
    save_xarray(PROCESSED_DATA_DIR, epoch_key, replay_info_mec.to_xarray(),
                '/replay_info_mec')
    save_xarray(PROCESSED_DATA_DIR, epoch_key, posterior_density_mec,
                '/posterior_density_mec')

    (replay_info_ca1_mec, decision_state_probability_ca1_mec,
     posterior_density_ca1_mec) = decode_ripple_clusterless(
        epoch_key, ANIMALS, ripple_times, position_info=position_info,
        brain_areas=['ca1', 'mec'])
    save_xarray(PROCESSED_DATA_DIR, epoch_key, replay_info_ca1_mec.to_xarray(),
                '/replay_info_ca1_mec')
    save_xarray(PROCESSED_DATA_DIR, epoch_key, posterior_density_ca1_mec,
                '/posterior_density_ca1_mec')

    logging.info('Estimating ripple-locked LFP connectivity...')
    estimate_lfp_ripple_connectivity(epoch_key, ripple_times, replay_info_ca1)

    logging.info('Finished Processing')


if __name__ == '__main__':
    exit(main())
