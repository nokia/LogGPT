#© 2023 Nokia
#Licensed under the Creative Commons Attribution Non Commercial 4.0 International license
#SPDX-License-Identifier: CC-BY-NC-4.0
#

import random
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
import regex as re
from tqdm import tqdm
from ast import literal_eval



def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def hdfs_blk_process(df, blk_label_dict):
    data_dict = defaultdict(list)
    for idx, row in tqdm(df.iterrows()):
        blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
        blkId_set = set(blkId_list)
        for blk_Id in blkId_set:
            if blk_Id not in data_dict:
                data_dict[blk_Id] = [row['EventId']]
            else:
                data_dict[blk_Id].append(row["EventId"])

    data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])

    data_df["Label"] = data_df["BlockId"].apply(
        lambda x: blk_label_dict.get(x))  # add label to the sequence of each blockid

    return data_df

def sliding_window(df, options):
    if options['dataset_name'] == 'BGL':
        df['datatime'] = pd.to_datetime(df['Time'], format='%Y-%m-%d-%H.%M.%S.%f')
    if options['dataset_name'] == 'Thunderbird':
        df['datatime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y-%m-%d %H:%M:%S')
    if options['dataset_name'] == 'OpenStack':
        df['datatime'] = pd.to_datetime(df['Time'] + ' ' + df['Pid'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        df['datatime'] = df['datatime'].fillna(method='ffill')
    df['timestamp'] = df['datatime'].values.astype(np.int64) // 10 ** 9
    df = df.sort_values('timestamp')

    df.set_index('timestamp', drop=False, inplace=True)
    start_time = df.timestamp.min()
    end_time = df.timestamp.max()

    new_data = []
    while start_time < end_time:
        df_window = df.loc[start_time:start_time+options["window_size"]]
        if len(df_window) > 1:  # Only consider windows with more than one value
            if len(df_window) > options['max_lens']:
                start_time_inner = df_window.timestamp.min()
                end_time_inner = df_window.timestamp.max()
                while (end_time_inner - start_time_inner) > options['max_lens']:
                    df_window_inner = df_window.loc[start_time_inner:start_time_inner+options['max_lens']]
                    new_data.append([
                        df_window_inner['Label'].values.tolist(),
                        df_window_inner['Label'].max(),
                        df_window_inner['EventId'].values.tolist()
                    ])
                    start_time_inner += options['max_lens'] // 2
            else:
                new_data.append([
                    df_window['Label'].values.tolist(),
                    df_window['Label'].max(),
                    df_window['EventId'].values.tolist()
                ])
        start_time += options['step_size']

    print('there are %d instances (sliding windows) in this dataset\n' % len(new_data))
    return pd.DataFrame(new_data, columns=['Label_org', 'Label', 'EventSequence'])



def preprocessing(preprocessing=True, dataset_name='HDFS', options=None):
    if preprocessing:
        if dataset_name == 'HDFS':
            print("Preprocessing HDFS dataset")
            df = pd.read_csv('./datasets/HDFS.log_structured.csv', engine='c', na_filter=False, memory_map=True)
            blk_df = pd.read_csv('./datasets/anomaly_label.csv', engine='c', na_filter=False, memory_map=True)
            blk_label_dict = {}
            for _, row in tqdm(blk_df.iterrows()):
                blk_label_dict[row['BlockId']] = 1 if row['Label'] == 'Anomaly' else 0

            hdfs_df = hdfs_blk_process(df, blk_label_dict)
            hdfs_df.to_csv('./datasets/HDFS.BLK.csv')
            del df
            del blk_label_dict
            del blk_df
        elif dataset_name == 'BGL':
            print("Preprocessing BGL dataset")
            df = pd.read_csv('./datasets/BGL.log_structured.csv', engine='c', na_filter=False, memory_map=True)
            print('There are %d instances in this dataset\n' % len(df))
            df['Label'] = df['Label'].ne('-').astype(int)
            new_df = sliding_window(df, options)
            new_df.to_csv('./datasets/BGL.W{}.S{}.csv'.format(options['window_size'],
                                                              options['step_size']))
            del new_df

        elif dataset_name == 'Thunderbird':
            print("Preprocessing Thunderbird dataset")
            df = pd.read_csv('./datasets/Thunderbird.log_structured.csv', engine='c', na_filter=False, memory_map=True)
            df['Label'] = df['Label'].ne('-').astype(int)
            print('There are %d instances in this dataset\n' % len(df))
            new_df = sliding_window(df, options)
            new_df.to_csv('./datasets/Thunderbird.W{}.S{}.csv'.format(options['window_size'],
                                                                      options['step_size']))
            del new_df

        elif dataset_name == 'OpenStack':
            print("Preprocessing OpenStack dataset")
            df = pd.read_csv('./datasets/OpenStack.log_structured.csv', engine='c', na_filter=False, memory_map=True)
            with open('./datasets/OpenStack_anomaly_labels.txt', 'r') as f:
                abnormal_label = f.readlines()
            lst_abnormal_label = []
            for i in abnormal_label[2:]:
                lst_abnormal_label.append(i.strip())
            df['Label'] = 0
            for i in range(len(lst_abnormal_label)):
                for j in range(len(df)):
                    if lst_abnormal_label[i] in df['Content'][j]:
                        df['Label'][j] = 1
            print(df['Label'].value_counts())


            print('There are %d instances in this dataset\n' % len(df))
            new_df = sliding_window(df, options)
            new_df.to_csv('./datasets/OpenStack.W{}.S{}.csv'.format(options['window_size'],
                                                                      options['step_size']))
            del new_df

def train_test_split(dataset_name='HDFS', train_samples=5000, seed=42, options=None, dir='.'):
    if dataset_name == 'HDFS':
        hdfs_df = pd.read_csv(dir + '/datasets/HDFS.BLK.csv', index_col=0, dtype={'BlickId': str, 'Label':int})
        hdfs_df.EventSequence = hdfs_df.EventSequence.apply(literal_eval)
        normal_df = hdfs_df[hdfs_df['Label'] == 0]
        normal_df = normal_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        anomaly_df = hdfs_df[hdfs_df['Label'] == 1]
        train_df = normal_df[:train_samples]
        test_df = normal_df[train_samples:].append(anomaly_df)
        train_df.to_csv(dir + '/datasets/HDFS.BLK.train.csv')
        test_df.to_csv(dir + '/datasets/HDFS.BLK.test.csv')
        print(f'datasets contains: {len(hdfs_df)} blocks, {len(normal_df)} normal blocks, '
              f'{len(anomaly_df)} anomaly blocks')
        print(f'Trianing dataset contains: {len(train_df)} blocks')
        print(f'Testing dataset contains: {len(test_df)} blocks, '
              f'{len(test_df.loc[test_df["Label"] == 0])} normal blocks ,{len(anomaly_df)} anomaly blocks')
        return train_df, test_df
    elif dataset_name == 'BGL':
        df = pd.read_csv(dir + '/datasets/BGL.W{}.S{}.csv'.format(options['window_size'], options['step_size']), index_col=0, dtype={'Label': int})
        df.EventSequence = df.EventSequence.apply(literal_eval)
        normal_df = df[df['Label'] == 0]
        normal_df = normal_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        anomaly_df = df[df['Label'] == 1]
        train_df = normal_df[:train_samples]
        test_df = normal_df[train_samples:].append(anomaly_df)
        train_df.to_csv(dir + '/datasets/BGL.W{}.S{}.train.csv'.format(options['window_size'], options['step_size']))
        test_df.to_csv(dir + '/datasets/BGL.W{}.S{}.test.csv'.format(options['window_size'], options['step_size']))
        print(f'datasets contains: {len(df)} windows, {len(normal_df)} normal windows, '
                f'{len(anomaly_df)} anomaly windows')
        print(f'Trianing dataset contains: {len(train_df)} windows')
        print(f'Testing dataset contains: {len(test_df)} windows, '
                f'{len(test_df.loc[test_df["Label"] == 0])} normal windows ,{len(anomaly_df)} anomaly windows')
        return train_df, test_df
    elif dataset_name == 'Thunderbird':
        df = pd.read_csv(dir + '/datasets/Thunderbird.W{}.S{}.csv'.format(options['window_size'], options['step_size']), index_col=0, dtype={'Label': int})
        df.EventSequence = df.EventSequence.apply(literal_eval)
        normal_df = df[df['Label'] == 0]
        normal_df = normal_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        anomaly_df = df[df['Label'] == 1]
        train_df = normal_df[:train_samples]
        test_df = normal_df[train_samples:].append(anomaly_df)
        train_df.to_csv(dir + '/datasets/Thunderbird.W{}.S{}.train.csv'.format(options['window_size'], options['step_size']))
        test_df.to_csv(dir + '/datasets/Thunderbird.W{}.S{}.test.csv'.format(options['window_size'], options['step_size']))
        print(f'datasets contains: {len(df)} windows, {len(normal_df)} normal windows, '
                f'{len(anomaly_df)} anomaly windows')
        print(f'Trianing dataset contains: {len(train_df)} windows')
        print(f'Testing dataset contains: {len(test_df)} windows, '
                f'{len(test_df.loc[test_df["Label"] == 0])} normal windows ,{len(anomaly_df)} anomaly windows')
        return train_df, test_df
    elif dataset_name == 'OpenStack':
        df = pd.read_csv(dir + '/datasets/OpenStack.W{}.S{}.csv'.format(options['window_size'], options['step_size']), index_col=0, dtype={'Label': int})
        df.EventSequence = df.EventSequence.apply(literal_eval)
        normal_df = df[df['Label'] == 0]
        normal_df = normal_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        anomaly_df = df[df['Label'] == 1]
        train_df = normal_df[:train_samples]
        test_df = normal_df[train_samples:].append(anomaly_df)
        train_df.to_csv(dir + '/datasets/OpenStack.W{}.S{}.train.csv'.format(options['window_size'], options['step_size']))
        test_df.to_csv(dir + '/datasets/OpenStack.W{}.S{}.test.csv'.format(options['window_size'], options['step_size']))
        print(f'datasets contains: {len(df)} windows, {len(normal_df)} normal windows, '
                f'{len(anomaly_df)} anomaly windows')
        print(f'Trianing dataset contains: {len(train_df)} windows')
        print(f'Testing dataset contains: {len(test_df)} windows, '
                f'{len(test_df.loc[test_df["Label"] == 0])} normal windows ,{len(anomaly_df)} anomaly windows')
        return train_df, test_df




def get_training_dictionary(df):
    '''Get training dictionary

    Arg:
        df: dataframe of preprocessed sliding windows

    Return:
        dictionary of training datasets
    '''
    dic = {}
    count = 0
    for i in range(len(df)):
        lst = list(df['EventSequence'].iloc[i])
        for j in lst:
            if j in dic:
                pass
            else:
                dic[j] = str(count)
                count += 1
    return dic
