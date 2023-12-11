#© 2023 Nokia
#Licensed under the Creative Commons Attribution Non Commercial 4.0 International license
#SPDX-License-Identifier: CC-BY-NC-4.0
#
from argparse import ArgumentParser

def get_args():
    '''
    :param args:
    --------------------------------
    dataset args:
    dataset_name: str = 'HDFS'
    output_dir: str = '/datasets/'
    seed: int = 7
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    download_datasets: bool = True
    preprocessing: bool = True
    max_lens: int = 256
    sliding_window: bool = False

    --------------------------------
    GPT2 args:
    train_samples: int = 5000
    building_vocab: bool = True
    init_lr: float = 1e-4
    init_num_epochs: int = 100
    init_batch_size: int = 16
    init_logGPT: bool = True
    num_return_sequences: int = 20
    top_k: int = 7
    tqdm: bool = False
    n_layers: int = 6
    n_heads: int = 6
    n_embd: int = 60

    --------------------------------
    LogGPT args:
    logGPT_episode: int = 20
    logGPT_lr: float = 1e-6
    save_memory: bool = False
    logGPT_training: bool = True
    :return: parser
    '''

    parser = ArgumentParser()
    parser.add_argument('--dataset_name', default='HDFS', type=str,
                        help='The name of the dataset to be parsed (default: HDFS)')
    parser.add_argument('--output_dir', default='./datasets/', type=str,
                        help='The output directory of parsing results')
    parser.add_argument('--seed', default=7, type=int, help='random seed (default: 7)')
    parser.add_argument('--device', default='cuda:0', type=str, help='device (default: cuda:0)')
    parser.add_argument('--download_datasets', default=True, type=bool, help='Download datasets (default: True)')
    parser.add_argument('--preprocessing', default=True, type=bool, help='Preprocessing datasets (default: True)')
    parser.add_argument('--max_lens', default=512, type=int, help='Max length of sequence (default: 512)')
    parser.add_argument('--sliding_window', default=False, type=bool, help='Sliding window (default: False)')

    #GPT2
    parser.add_argument('--train_samples', default=5000, type=int, help='Train samples (default: 5000)')
    parser.add_argument('--building_vocab', default=False, type=bool, help='Building vocab (default: True)')
    parser.add_argument('--init_lr', default=1e-4, type=float, help='Initial learning rate (default: 1e-4)')
    parser.add_argument('--init_num_epochs', default=100, type=int, help='Initial number of epochs (default: 100)')
    parser.add_argument('--init_batch_size', default=16, type=int, help='Initial batch size (default: 16)')
    parser.add_argument('--init_logGPT', default=True, type=bool, help='Initial logGPT (default: True)')
    parser.add_argument('--num_return_sequences', default=20, type=int, help='Number of generated sequences (default: 20)')
    parser.add_argument('--top_k', default=7, type=int, help='Top k (default: 7)')
    parser.add_argument('--tqdm', default=False, type=bool, help='Tqdm (default: False)')
    parser.add_argument('--n_layers', default=6, type=int, help='Number of layers (default: 6)')
    parser.add_argument('--n_heads', default=6, type=int, help='Number of heads (default: 6)')
    parser.add_argument('--n_embd', default=60, type=int, help='Number of embeddings (default: 60)')

    #LogGPT
    parser.add_argument('--logGPT_episode', default=20, type=int, help='LogGPT episode (default: 20)')
    parser.add_argument('--logGPT_lr', default=1e-6, type=float, help='LogGPT learning rate (default: 1e-6)')
    parser.add_argument('--save_memory', default=False, type=bool, help='Save memory (default: False)')
    parser.add_argument('--logGPT_training', default=True, type=bool, help='LogGPT training (default: True)')
    return parser