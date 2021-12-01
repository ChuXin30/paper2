import sys
sys.path.append('../')


if __name__ == '__main__':
    count = int(sys.argv[1])
    output_dir = sys.argv[2]

    # count = 1000
    # output_dir='../logs/HDFS/HDFS_1k.log'

    print count
    print output_dir
    # f = open('../logs/HDFS/HDFS.log', mode='r')
    # f = open('../logs/HPC/HPC.log', mode='r')
    # f = open('../logs/Zookeeper/Zookeeper.log', mode='r')
    # f = open('../logs/Proxifier/Proxifier.log', mode='r')
    f = open('../logs/BGL/BGL.log', mode='r')

    f_w = open(output_dir , mode='w')

    j = 0
    for line in f.readlines():
        if j < count :
            f_w.write(line)
        else:
            break
        j = j +1
    f_w.close()
    f.close()