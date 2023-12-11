#© 2023 Nokia
#Licensed under the Creative Commons Attribution Non Commercial 4.0 International license
#SPDX-License-Identifier: CC-BY-NC-4.0
#

from utils import preprocessing, utils
from args import hdfs_config, thunderbird_config, bgl_config
import os
import sys
from model import initGPT, logGPT
import warnings
warnings.filterwarnings("ignore")


def main():
    if len(sys.argv) < 2:
        parser = hdfs_config.get_args()
    else:
        if sys.argv[1] == 'Thunderbird':
            parser = thunderbird_config.get_args()
        elif sys.argv[1] == 'BGL':
            parser = bgl_config.get_args()
        elif sys.argv[1] == 'HDFS':
            parser = hdfs_config.get_args()
        else:
            print("Only support HDFS, Thunderbird and BGL dataset!")
            return 1
    args, unknown = parser.parse_known_args()
    options = vars(args)
    print(options)
    utils.set_seed(options['seed'])
    if options['download_datasets']:
        print('Downloading datasets...')
        preprocessing.parsing(options['dataset_name'], options['output_dir'])
    else:
        print('Loading datasets...')

    path = "datasets"
    if len(os.listdir(path)) == 0:
        print("Please download the dataset first!")
        return 1

    utils.preprocessing(options['preprocessing'], options['dataset_name'], options)
    train_df, test_df = utils.train_test_split(options['dataset_name'], options['train_samples'], options['seed'], options=options)
    initGPT_model = initGPT.InitGPT(options, train_df, test_df)
    print('Initializing LogGPT...')
    logGPT_model = logGPT.LogGPT(options, train_df, test_df, initGPT_model)
    print('Done')


if __name__ == '__main__':
    main()
