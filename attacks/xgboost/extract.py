import random
import numpy as np
import logging
import glob
import os
import sys
import subprocess
import argparse
from os import makedirs
import constants as ct
import multiprocessing as mp


logger = logging.getLogger('xgboost')

def config_logger(args):
    # Set file
    log_file = sys.stdout
    if args.log != 'stdout':
        log_file = open(args.log, 'w')
    ch = logging.StreamHandler(log_file)

    # Set logging format
    ch.setFormatter(logging.Formatter(ct.LOG_FORMAT))
    logger.addHandler(ch)

    # Set level format
    logger.setLevel(logging.INFO)




def extract(features, x, times, directions):   # x 为从列表的第50到列表的倒数第50之间    x对应一个值调用一次extract函数
    st = x - 50       # x前移50
    gaps = [] 
    for j in range(0, 99):
        gaps.append(times[st + j + 1] - times[st + j])   # gaps 列表存储了相邻时间戳之间的差值
    features.append(float(np.mean(gaps)))
    features.append(float(np.std(gaps)))
    features.append(gaps[48])
    features.append(gaps[49])
    features.append(gaps[50])
    features.append(gaps[51])
    features.append(gaps[52])
    features.append(max(gaps))

    nextinc = 10                                 #通过循环查找一个特定的索引（在当前位置 x 之后的一定范围内）来获取下一个特定条件（directions[x+j] < 0）的位置，其中 j 在范围（1 到 20）内。找到这个位置后，计算这两个位置之间的时间差并将其附加到 features 列表中
    for j in range(1, 20):
        if (directions[x+j] < 0):
            nextinc = j
            break
    features.append(times[x+nextinc] - times[x])
    features.append(times[x] - times[0])

    #packet rate should be lowest around a gap
    for j in range(1, 10):         #计算 x 周围的一些时间戳之间的时间差并将其附加到 features 列表中。在循环中，以 j 为步长，获取一些时间戳并计算它们与当前位置 x 之间的时间差
        features.append(times[x+j*2] - times[x-j*2])

    #number of outgoing packets should be highest around a gap

    count = 0            #对于当前位置 x 周围的时间戳，统计方向为正（大于零）的数量，并将其附加到 features 列表中。这里分别使用不同范围内的时间戳来计算出站数据包的数量。
    for j in range(0, 11):
        if (directions[x+j] > 0):
            count += 1
    features.append(count)
    count = 0
    for j in range(0, 6):
        if (directions[x+j] > 0):
            count += 1
    features.append(count)
    count = 0
    for j in range(0, 11):
        if (directions[x-j] > 0):
            count += 1
    features.append(count)
    count = 0
    for j in range(0, 6):
        if (directions[x-j] > 0):
            count += 1
    features.append(count)

def readfile(infile, times, directions):
    f = open(infile, "r")
    l = f.readlines()
    f.close()

    for x in l:
        x = x.split("\t")
        times.append(float(x[0]))
        directions.append(int(x[1]))

def writefile(outfile, features):
    f = open(outfile, "w")
    for feat in features:
        f.write(str(feat) + "\n")
    f.close()

def get_truesplit(times, directions):
    #also does basic checking on whether or not the file is too bad to work
    if len(directions) < 3:
        logging.warning("This file's length is {}, abnormal".format(len(directions)))
        return None
    # for i in range(len(directions)-1):
    #     if abs(int(directions[i+1])) != abs(int(directions[i])):
    #         return i+1
    # return -1

    truesplits = []
    cnt = 2
    for i in range(1, len(directions)):
        direction = int(directions[i])
        if abs(direction) > 800:   #  当检测到 directions 中的值在绝对值上大于 800 时，会跳过这些值的检查
            continue 
        if  direction == cnt:     #当检测到 directions 中的值符合预期的递增顺序（例如，方向值从 1 开始递增），会将这些位置添加到 truesplits 列表中，并递增 cnt 值。
            truesplits.append(i)
            cnt += 1
        # if (abs(int(directions[i])) < 888 and (   abs(int(directions[i])) != abs(int(directions[i-1])))    ):
        #     truesplits.append(i)
        
    # print("truesplits", truesplits)
    # f1count = 0
    # f2count = 0
    # for i in range(0, len(directions)):
    #     if (abs(directions[i]) == 1):
    #         f1count += 1
    #     if (abs(directions[i]) == 2):
    #         f2count += 1

    # if (f1count < 50 or f2count < 50):
    #     truesplit = -1

    # if (truesplit < 50):
    #     truesplit = -1

    return truesplits

def parse_arguments():

    parser = argparse.ArgumentParser(description='XGboost split algorithm.')

    parser.add_argument('t',
                        metavar='<traces path>',
                        help='Path to the directory with the traffic traces to be simulated.')
    parser.add_argument('-mode',
                        metavar='<mode>',
                        default = 'train',
                        help='Extract features for training set or test set.')
    parser.add_argument('--log',
                        type=str,
                        dest="log",
                        metavar='<log path>',
                        default='stdout',
                        help='path to the log file. It will print to stdout by default.')


    # Parse arguments
    args = parser.parse_args()
    config_logger(args)
    return args
def parallel(flist,testfolder, n_jobs = 20): 
    pool = mp.Pool(n_jobs)
    pool.map(work, zip(flist,[testfolder]*len(flist)))


def work(file):   #此处的file为(flist, [testfolder]*len(flist))元组， 其中flist为符合搜索模式的文件名列表，testfolder为测试文件夹路径
    f = file[0]
    testfolder = file[1]
    fname = f.split('/')[-1].split(".")[0]    
    logger.debug('extract feature for file {}'.format(f))
    times = []
    directions = []
    badfile = 0
    try:
        readfile(f, times, directions)
    except:
        badfile = 1
    data_dict = {'feature':[],'location':[],'truesplits':[]}    
    truesplits = get_truesplit(times, directions)
    data_dict['truesplits'] = truesplits

    if (badfile == 0 and truesplits != None):
        for x in range(50, len(times) - 50):
            if directions[x] > 0:
                features = []
                extract(features, x, times, directions)
                data_dict['feature'].append(features)
                data_dict['location'].append(x)
    else:
        dic = {1:'bad', 0:'good'}
        logger.warning('File {} is {}. True split is {}.'.format(f,dic[badfile],truesplits))      

    instpath = os.path.join(testfolder, fname)         
    np.save(instpath, data_dict)



if __name__ == '__main__':
    args = parse_arguments()
    logger.debug("Arguments: %s" % (args))
    if args.mode == 'test':
        tmp = args.t.split('/')[-2]
        testfolder = os.path.join(ct.outputdir, tmp)
        if not os.path.exists(testfolder):
            makedirs(testfolder)
        fpath = os.path.join(args.t, '*.merge')  # fpath是args.t路径下以.merge结尾的文件的匹配模式
        flist = glob.glob(fpath)            # flist所有符合fpath模式的文件列表
        parallel(flist,testfolder)

        # for f in flist:
        #     fname = f.split('/')[-1].split(".")[0]
        #     logging.info('extract feature for file {}'.format(f))
        #     times = []
        #     directions = []
        #     badfile = 0
        #     try:
        #         readfile(f, times, directions)
        #     except:
        #         badfile = 1
        #     data_dict = {'feature':[],'location':[],'truesplits':[]}    
        #     truesplits = get_truesplit(times, directions)
        #     data_dict['truesplits'] = truesplits

        #     if (badfile == 0 and truesplits != None):
        #         for x in range(50, len(times) - 50):
        #             if directions[x] > 0:
        #                 features = []
        #                 extract(features, x, times, directions)
        #                 data_dict['feature'].append(features)
        #                 data_dict['location'].append(x)
        #     else:
        #         dic = {1:'bad', 0:'good'}
        #         logger.warning('File {} is {}. True split is {}.'.format(f,dic[badfile],truesplits))      

        #     instpath = os.path.join(testfolder, fname)         
        #     np.save(instpath, data_dict)



    elif args.mode == 'train':
        fpath = os.path.join(args.t, '*.merge')
        flist = glob.glob(fpath)
        flist.sort(key = lambda x: int(x.split('/')[-1].split('.merge')[0]))

        data_dict = {'feature':[],'label':[]}

        for infile in flist:
            logger.debug('Processing file {}'.format(infile))
            '''1000 controls how much training data are used'''
            badfile = 0
            times = []
            directions = []
            
            try:
                readfile(infile, times, directions)
                logging.info('Reading file {}'.format(infile))
            except:
                badfile = 1

            #if the file is really bad don't bother
            truesplits = get_truesplit(times, directions)
            logging.debug('True split is {}'.format(truesplits))

            if (badfile == 0 and truesplits != None):
                gb = [[], []]
                gb[0].extend(truesplits)


                baddies = []
                for i in range(50, len(directions)-50):
                    if (directions[i] > 0 and directions[i] < 999) and i not in truesplits:
                        baddies.append(i)

                if len(baddies) > 1:
                    bad = np.random.choice(baddies,min(len(truesplits),len(baddies)),False)
                    gb[1].extend(bad)

                np.random.shuffle(gb[0])
                np.random.shuffle(gb[1])

                for gbi in range(0, 2):
                    for i in range(0, len(gb[gbi])):
                        features = []                
                        x = gb[gbi][i]
                        extract(features, x, times, directions)
                        data_dict['feature'].append(features)
                        data_dict['label'].append(gbi)
        logger.debug("Training set shape:{}".format(np.array(data_dict['feature']).shape))
        outputdir = ct.outputdir+args.t.split('/')[-2]
        np.save(outputdir,data_dict)
        logger.debug("Training set is saved to {}".format(outputdir))
    else:
        logger.error('Wrong mode:{}'.format(args.mode))
