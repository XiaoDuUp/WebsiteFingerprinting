import sys
import subprocess
import numpy
import os
from loaders import *
import glob 
# import logging

# FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
# logging.basicConfig(format=FORMAT)
# logger = logging.getLogger('fextractor')

def extract(times, sizes, features):

    #Transmission size features
    features.append(len(sizes))               #流量包数    特征1

    count = 0
    for x in sizes:
        if x > 0:
            count += 1
    features.append(count)                #出流量包数 特征2
    features.append(len(times)-count)     #入流量包数 特征3

    features.append(times[-1] - times[0])    #一个文件内所有流量包传递时间 特征3

    #Unique packet lengths
##    for i in range(-1500, 1501):
##        if i in sizes:
##            features.append(1)
##        else:
##            features.append(0)

    #Transpositions (similar to good distance scheme)
    count = 0               # 4 根据出流量数据包的索引位置构建特征，但在达到500个特征之后停止，不够500则补X
    for i in range(0, len(sizes)):
        if sizes[i] > 0:
            count += 1
            features.append(i)
        if count == 500:
            break
    for i in range(count, 500):
        features.append("X")
        
    count = 0    # 5 此出数据包与上一个出数据包的距离作为特征，生成500个特征，不够500补X
    prevloc = 0
    for i in range(0, len(sizes)):
        if sizes[i] > 0:
            count += 1
            features.append(i - prevloc)
            prevloc = i
        if count == 500:
            break
    for i in range(count, 500):
        features.append("X")


    #Packet distributions (where are the outgoing packets concentrated)
    count = 0
    for i in range(0, min(len(sizes), 3000)):
        if i % 30 != 29:
            if sizes[i] > 0:
                count += 1
        else:
            features.append(count)         # 6 每30个数据流量包统计出流量的数，作为一个特征。共100个特征
            count = 0
    for i in range(len(sizes)/30, 100):
        features.append(0)

    #Bursts                     # 7 当有连续的入流量数据包时即为一次突发，统计距离上一次出现突发时，出流量数据包的数量
                                #压入bursts列表，将列表中的最大值，平均值，列表长度作为特征；若列表长为0，则将将X作为特征
    bursts = []
    curburst = 0
    consnegs = 0
    stopped = 0
    for x in sizes:
        if x < 0:
            consnegs += 1
            if (consnegs == 2):
                bursts.append(curburst)
                curburst = 0
                consnegs = 0
        if x > 0:
            consnegs = 0
            curburst += x
    if curburst > 0:
        bursts.append(curburst)
    if (len(bursts) > 0):
        features.append(max(bursts))
        features.append(numpy.mean(bursts))
        features.append(len(bursts))
    else:
        features.append("X")
        features.append("X")
        features.append("X")
##    print bursts
    counts = [0, 0, 0, 0, 0, 0]
    for x in bursts:
        if x > 2:
            counts[0] += 1
        if x > 5:
            counts[1] += 1
        if x > 10:
            counts[2] += 1
        if x > 15:
            counts[3] += 1
        if x > 20:
            counts[4] += 1
        if x > 50:
            counts[5] += 1
    features.append(counts[0])
    features.append(counts[1])
    features.append(counts[2])
    features.append(counts[3])
    features.append(counts[4])
    features.append(counts[5])    # 8 统计出现突发时，出流量的种类：出流量数大于2、大于5、大于10、15、20、50 情况的次数
    for i in range(0, 100):    # 9 将突发的前100个值作为特征添加到 features 列表中。如果突发不足100个，则将剩余的位置用 "X" 填充
        try:
            features.append(bursts[i])
        except:
            features.append("X")

    for i in range(0, 10):   # 10 将数据包大小列表 sizes 的前10个值加上1500作为特征添加到 features 列表中。如果 sizes 列表不足10个值，则将剩余的位置用 "X" 填充。
        try:
            features.append(sizes[i] + 1500)
        except:
            features.append("X")

    itimes = [0]*(len(sizes)-1)   # 11 计算了数据包时间间隔列表 itimes 中每个时间间隔的平均值和标准差作为特征
    for i in range(1, len(sizes)):
        itimes[i-1] = times[i] - times[i-1]
    if len(itimes) > 0:
        features.append(numpy.mean(itimes))
        features.append(numpy.std(itimes))
    else:
        features.append("X")
        features.append("X")

try:
    '''python fex...py option.txt tracepath'''
    optfname = sys.argv[1]
    d = load_options(optfname)
    data_name = sys.argv[2]
    # print data_name
except Exception,e:
    print sys.argv[0], str(e)
    sys.exit(0)

'''check whether feature already extracted'''
path = d['OUTPUT_LOC']+ data_name.split('/')[-2] + '/'
tmp = glob.glob(os.path.join(path,'*'))
tmp2 = glob.glob(os.path.join(path,'*/*'))
if len(tmp) > 1 or len(tmp2) > 1:
    '''sometimes in mac os there will be a .DS store file, so use >1'''
    print('Feature already extracted, skip.')
    exit(0)

flist = []
fold = data_name
# for s in range(0, d["CLOSED_SITENUM"]):
#     for i in range(0, d["CLOSED_INSTNUM"]):
#         flist.append(os.path.join(fold, str(s)+'-'+str(i)))
# for i in range(0, d["OPEN_INSTNUM"]):
#     flist.append(os.path.join(fold, str(i)))


#this flag is used for different format of input insts
#for training data: folder/inst1, 2 ...
#for testing data: folder/0/inst1,2... folder/1/inst1,2..
fpath = os.path.join(fold, '*/*')
flist = glob.glob(fpath)
flag = 'test'
if len(flist) == 0:
    fpath = os.path.join(fold,'*')
    flist = glob.glob(fpath)
    flag = 'train'
    if len(flist)  == 0:
        logger.error('Path {} is incorrect!'.format(fpath))
        flag = 'Error'


if(not os.path.isdir(d['OUTPUT_LOC'])):
    os.mkdir(d['OUTPUT_LOC'])

for fname in flist:
    # if "-0.cell" in fname:

    #load up times, sizes
    times = []
    sizes = []
    try:
        f = open(fname, "r")
    except:
        print(fname + ' does not exist!')
        continue

    print fname
    tname = fname.split('/')[-1] + ".cellkNN"
     
    for x in f:
        x = x.split("\t")
        times.append(float(x[0]))
        sizes.append(int(x[1]))
    f.close()

    #Extract features. All features are non-negative numbers or X. 
    features = []    # 12 对提取的特征进行字符串化处理，如果特征是 'X'，则替换为 -1
    try: 
        extract(times, sizes, features)
        writestr = ""
        for x in features:
            if x == 'X':
                x = -1
            writestr += (repr(x) + "\n")
        if flag == 'train':
            path = d['OUTPUT_LOC']+ data_name.split('/')[-2] + '/'
            if not os.path.exists(path):
                os.makedirs(path)
            fout = open(path+ tname, "w")
            fout.write(writestr[:-1])
            fout.close()
        elif flag == 'test':
            tmp = fname.split('/')
            folder = tmp[-3]
            subfolder = tmp[-2]
            inst = tmp[-1]
            path = os.path.join(d['OUTPUT_LOC'],folder,subfolder)
            if not os.path.exists(path):
                os.makedirs(path)
            fout = open(os.path.join(path,tname), "w")
            fout.write(writestr[:-1])
            fout.close()
        else:
            raise Exception("Invalid flag!")

    except Exception as e:
        print(e)
