import os
import tarfile
import urllib.request
import shutil
from . import drain, drainTB


# Drain: https://github.com/logpai/logparser

def parsing(dataset_name, output_dir='./datasets/'):
    """Download and parsing dataset

    Args:
        dataset_name: name of the log dataset
        output_dir: directory name for datasets storage

    Returns:
        Structured log datasets in Pandas Dataframe after adopt Drain
    """
    path = os.getcwd()
    directory = path + output_dir[1:]
    if not os.path.exists(directory):
        print(f'Making directory for dataset storage {directory}')
        os.makedirs(directory)

    if dataset_name == 'BGL':
        url = 'https://zenodo.org/record/3227177/files/BGL.tar.gz?download=1'
        downloaded_filename = 'BGL.tar.gz'
        print(downloaded_filename)
        urllib.request.urlretrieve(url, downloaded_filename)
        tar = tarfile.open(downloaded_filename, "r|gz")
        tar.extractall()
        tar.close()
        try:
            os.remove(downloaded_filename)
        except OSError:
            pass

        input_dir = ''  # The input directory of log file
        log_file = 'BGL.log'  # The input log file name
        log_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>'  # BGL log format
        # Regular expression list for optional preprocessing (default: [])
        regex = [
            r'core\.\d+',
            r'blk_(|-)[0-9]+',  # block id
            r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',  # IP
            r'([0-9a-f]+[:][0-9a-f]+)',
            r'fpr[0-9]+[=]0x[0-9a-f]+ [0-9a-f]+ [0-9a-f]+ [0-9a-f]+',
            r'r[0-9]+[=]0x[0-9a-f]+',
            r'[l|c|xe|ct]r=0x[0-9a-f]+',
            r'0x[0-9a-f]+',
            r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
        ]
        st = 0.5  # Similarity threshold
        depth = 4  # Depth of all leaf nodes

        parser = drain.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex)
        parser.parse(log_file)

        try:
            os.remove(log_file)
        except OSError:
            pass

    elif dataset_name == 'Thunderbird':
        url = 'https://zenodo.org/record/3227177/files/Thunderbird.tar.gz?download=1'
        downloaded_filename = 'Thunderbird.tar.gz'
        urllib.request.urlretrieve(url, downloaded_filename)
        tar = tarfile.open(downloaded_filename, "r|gz")
        tar.extractall()
        tar.close()
        try:
            os.remove(downloaded_filename)
        except OSError:
            pass

        input_dir = ''  # The input directory of log file
        log_file = 'Thunderbird.log'  # The input log file name
        log_format = '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>'
        # Regular expression list for optional preprocessing (default: [])
        regex = [r'(\d+\.){3}\d+',
                 r'[a-d]n[0-9]+',
                 r'\<[0-9a-f]{16}\>\{.+\}']
        st = 0.3  # Similarity threshold
        depth = 2  # Depth of all leaf nodes

        parser = drainTB.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st,
                                            rex=regex)
        parser.parse(log_file)

        try:
            os.remove(log_file)
        except OSError:
            pass

    elif dataset_name == 'HDFS':
        url = 'https://zenodo.org/record/3227177/files/HDFS_1.tar.gz?download=1'
        downloaded_filename = 'HDFS_1.tar.gz'
        urllib.request.urlretrieve(url, downloaded_filename)
        tar = tarfile.open(downloaded_filename, "r|gz")
        tar.extractall()
        tar.close()
        try:
            os.remove(downloaded_filename)
        except OSError:
            pass

        input_dir = ''  # The input directory of log file
        log_file = 'HDFS.log'  # The input log file name
        log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
        # Regular expression list for optional preprocessing (default: [])
        regex = [
            r'blk_(|-)[0-9]+',  # block id
            r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',  # IP
            r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
        ]
        st = 0.5  # Similarity threshold
        depth = 4  # Depth of all leaf nodes

        parser = drain.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex)
        parser.parse(log_file)

        try:
            shutil.move('anomaly_label.csv', 'datasets/anomaly_label.csv')
            os.remove(log_file)
        except OSError:
            pass

    elif dataset_name == 'OpenStack':
        url = 'https://zenodo.org/record/3227177/files/OpenStack.tar.gz?download=1'
        downloaded_filename = 'OpenStack.tar.gz'
        urllib.request.urlretrieve(url, downloaded_filename)
        tar = tarfile.open(downloaded_filename, "r|gz")
        tar.extractall('OpenStack')
        tar.close()
        try:
            os.remove(downloaded_filename)
        except OSError:
            pass

        count = 0
        with open('OpenStack.log', mode='w') as outfile:
            for i in os.listdir('./OpenStack'):
                if i != 'abnormal_labels.txt':
                    with open(f'./OpenStack/{i}', mode='r') as infile:
                        for line in infile:
                            outfile.write(f'{i} {line}')
                            count += 1
        input_dir = ''  # The input directory of log file
        log_file = 'OpenStack.log'  # The input log file name
        log_format = '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>'  # OpenStack log format
        # Regular expression list for optional preprocessing (default: [])
        regex = [r'((\d+\.){3}\d+,?)+', r'/.+?\s', r'\d+']
        st = 0.5  # Similarity threshold
        depth = 2  # Depth of all leaf nodes

        parser = drain.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex)
        parser.parse(log_file)

        try:
            shutil.move('./OpenStack/anomaly_labels.txt', './datasets/OpenStack_anomaly_labels.txt')
            os.remove(log_file)
            shutil.rmtree('./OpenStack')
        except OSError:
            pass

    else:
        raise ValueError('Invalid dataset name')