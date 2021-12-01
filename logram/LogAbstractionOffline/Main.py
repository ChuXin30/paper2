from DictionarySetUp import dictionaryBuilder
from MatchToken import tokenMatch
from evaluator import evaluate
from datetime import datetime

HDFS_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
Andriod_format = '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>' #Andriod log format
Spark_format = '<Date> <Time> <Level> <Component>: <Content>'#Spark log format
Zookeeper_format = '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>' #Zookeeper log format
Windows_format = '<Date> <Time>, <Level>                  <Component>    <Content>' #Windows log format
Thunderbird_format = '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>' #Thunderbird_format
Apache_format = '\[<Time>\] \[<Level>\] <Content>' #Apache format
BGL_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>' #BGL format
Hadoop_format = '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>' #Hadoop format
HPC_format = '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>' #HPC format
Linux_format = '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>' #Linux format
Mac_format = '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>' #Mac format
OpenSSH_format = '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>' #OpenSSH format
OpenStack_format = '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>' #OpenStack format
HealthApp_format = '<Time>\|<Component>\|<Pid>\|<Content>'
Proxifier_format = '\[<Time>\] <Program> - <Content>'

HDFS_Regex = [
        r'blk_(|-)[0-9]+' , # block id
        r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', # IP
        r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$', # Numbers
]
Hadoop_Regex = [r'(\d+\.){3}\d+']
Spark_Regex = [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+']
Zookeeper_Regex = [r'(/|)(\d+\.){3}\d+(:\d+)?']
BGL_Regex = [r'core\.\d+']
HPC_Regex = [r'=\d+']
Thunderbird_Regex = [r'(\d+\.){3}\d+']
Windows_Regex = [r'0x.*?\s']
Linux_Regex = [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}']
Andriod_Regex = [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b']
Apache_Regex = [r'(\d+\.){3}\d+']
OpenSSH_Regex = [r'(\d+\.){3}\d+', r'([\w-]+\.){2,}[\w-]+']
OpenStack_Regex = [r'((\d+\.){3}\d+,?)+', r'/.+?\s', r'\d+']
Mac_Regex = [r'([\w-]+\.){2,}[\w-]+']
HealthApp_Regex = []
Proxifier_Regex = [r'<\d+\ssec', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\d{2}:\d{2}(:\d{2})*', r'[KGTM]B']


start_time = datetime.now()
doubleDictionaryList, triDictionaryList, allTokenList  = dictionaryBuilder(Spark_format, '../TestLogs/Spark/Spark_1k_match.log', Spark_Regex)
print(len(doubleDictionaryList))
print(len(triDictionaryList))
tokenMatch(allTokenList,doubleDictionaryList,triDictionaryList,15 ,10,'Output/')
print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))
evaluate('../GroundTruth/Spark_1k_match.log_structured.csv', 'Output/event.csv')

start_time = datetime.now()
doubleDictionaryList, triDictionaryList, allTokenList  = dictionaryBuilder(Spark_format, '../TestLogs/Spark/Spark_10k_match.log', Spark_Regex)
print(len(doubleDictionaryList))
print(len(triDictionaryList))
tokenMatch(allTokenList,doubleDictionaryList,triDictionaryList,15 ,10,'Output/')
print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))
evaluate('../GroundTruth/Spark_10k_match.log_structured.csv', 'Output/event.csv')

start_time = datetime.now()
doubleDictionaryList, triDictionaryList, allTokenList  = dictionaryBuilder(Spark_format, '../TestLogs/Spark/Spark_100k_match.log', Spark_Regex)
print(len(doubleDictionaryList))
print(len(triDictionaryList))
tokenMatch(allTokenList,doubleDictionaryList,triDictionaryList,15 ,10,'Output/')
print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))
evaluate('../GroundTruth/Spark_100k_match.log_structured.csv', 'Output/event.csv')

start_time = datetime.now()
doubleDictionaryList, triDictionaryList, allTokenList  = dictionaryBuilder(Spark_format, '../TestLogs/Spark/Spark_1m_match.log', Spark_Regex)
print(len(doubleDictionaryList))
print(len(triDictionaryList))
tokenMatch(allTokenList,doubleDictionaryList,triDictionaryList,15 ,10,'Output/')
print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))
evaluate('../GroundTruth/Spark_1m_match.log_structured.csv', 'Output/event.csv')




start_time = datetime.now()
doubleDictionaryList, triDictionaryList, allTokenList  = dictionaryBuilder(BGL_format, '../TestLogs/BGL/BGL_1k.log', BGL_Regex)
print(len(doubleDictionaryList))
print(len(triDictionaryList))
tokenMatch(allTokenList,doubleDictionaryList,triDictionaryList,92 ,4,'Output/')
print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))
evaluate('../GroundTruth/BGL_1k.log_structured.csv', 'Output/event.csv')

start_time = datetime.now()
doubleDictionaryList, triDictionaryList, allTokenList  = dictionaryBuilder(BGL_format, '../TestLogs/BGL/BGL_10k.log', BGL_Regex)
print(len(doubleDictionaryList))
print(len(triDictionaryList))
tokenMatch(allTokenList,doubleDictionaryList,triDictionaryList,92 ,4,'Output/')
print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))
evaluate('../GroundTruth/BGL_10k.log_structured.csv', 'Output/event.csv')

start_time = datetime.now()
doubleDictionaryList, triDictionaryList, allTokenList  = dictionaryBuilder(BGL_format, '../TestLogs/BGL/BGL_100k.log', BGL_Regex)
print(len(doubleDictionaryList))
print(len(triDictionaryList))
tokenMatch(allTokenList,doubleDictionaryList,triDictionaryList,92 ,4,'Output/')
print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))
evaluate('../GroundTruth/BGL_100k.log_structured.csv', 'Output/event.csv')

start_time = datetime.now()
doubleDictionaryList, triDictionaryList, allTokenList  = dictionaryBuilder(BGL_format, '../TestLogs/BGL/BGL_1m.log', BGL_Regex)
print(len(doubleDictionaryList))
print(len(triDictionaryList))
tokenMatch(allTokenList,doubleDictionaryList,triDictionaryList,92 ,4,'Output/')
print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))
evaluate('../GroundTruth/BGL_1m.log_structured.csv', 'Output/event.csv')


start_time = datetime.now()
doubleDictionaryList, triDictionaryList, allTokenList  = dictionaryBuilder(HDFS_format, '../TestLogs/HDFS/HDFS_1k.log', HDFS_Regex)
print(len(doubleDictionaryList))
print(len(triDictionaryList))
tokenMatch(allTokenList,doubleDictionaryList,triDictionaryList,15,10,'Output/')
print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))
evaluate('../GroundTruth/HDFS_1k.log_structured.csv', 'Output/event.csv')

start_time = datetime.now()
doubleDictionaryList, triDictionaryList, allTokenList  = dictionaryBuilder(HDFS_format, '../TestLogs/HDFS/HDFS_10k.log', HDFS_Regex)
print(len(doubleDictionaryList))
print(len(triDictionaryList))
tokenMatch(allTokenList,doubleDictionaryList,triDictionaryList,15,10,'Output/')
print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))
evaluate('../GroundTruth/HDFS_10k.log_structured.csv', 'Output/event.csv')

start_time = datetime.now()
doubleDictionaryList, triDictionaryList, allTokenList  = dictionaryBuilder(HDFS_format, '../TestLogs/HDFS/HDFS_100k.log', HDFS_Regex)
print(len(doubleDictionaryList))
print(len(triDictionaryList))
tokenMatch(allTokenList,doubleDictionaryList,triDictionaryList,15,10,'Output/')
print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))
evaluate('../GroundTruth/HDFS_100k.log_structured.csv', 'Output/event.csv')

start_time = datetime.now()
doubleDictionaryList, triDictionaryList, allTokenList  = dictionaryBuilder(HDFS_format, '../TestLogs/HDFS/HDFS_1m.log', HDFS_Regex)
print(len(doubleDictionaryList))
print(len(triDictionaryList))
tokenMatch(allTokenList,doubleDictionaryList,triDictionaryList,15,10,'Output/')
print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))
evaluate('../GroundTruth/HDFS_1m.log_structured.csv', 'Output/event.csv')




# doubleDictionaryList, triDictionaryList, allTokenList  = dictionaryBuilder(BGL_format, '../TestLogs/BGL/BGL_1k.log', BGL_Regex)
# print(len(doubleDictionaryList))
# print(len(triDictionaryList))
# tokenMatch(allTokenList,doubleDictionaryList,triDictionaryList,92 ,4,'Output/')
# evaluate('../GroundTruth/BGL_1k.log_structured.csv', 'Output/event.csv')
# print()


# doubleDictionaryList, triDictionaryList, allTokenList  = dictionaryBuilder(BGL_format, '../TestLogs/BGL/BGL_10k.log', BGL_Regex)
# print(len(doubleDictionaryList))
# print(len(triDictionaryList))
# tokenMatch(allTokenList,doubleDictionaryList,triDictionaryList,92 ,4,'Output/')
# evaluate('../GroundTruth/BGL_10k.log_structured.csv', 'Output/event.csv')
# print()




# doubleDictionaryList, triDictionaryList, allTokenList  = dictionaryBuilder(BGL_format, '../TestLogs/BGL/BGL_400.log', BGL_Regex)
# print(len(doubleDictionaryList))
# print(len(triDictionaryList))
# tokenMatch(allTokenList,doubleDictionaryList,triDictionaryList,92 ,4,'Output/')
# evaluate('../GroundTruth/BGL_400.log_structured.csv', 'Output/event.csv')
# print()

# # doubleDictionaryList, triDictionaryList, allTokenList  = dictionaryBuilder(BGL_format, '../TestLogs/BGL/BGL_4k.log', BGL_Regex)
# # print(len(doubleDictionaryList))
# # print(len(triDictionaryList))
# # tokenMatch(allTokenList,doubleDictionaryList,triDictionaryList,92 ,4,'Output/')
# # evaluate('../GroundTruth/BGL_4k.log_structured.csv', 'Output/event.csv')
# # print()

# doubleDictionaryList, triDictionaryList, allTokenList  = dictionaryBuilder(BGL_format, '../TestLogs/BGL/BGL_40k.log', BGL_Regex)
# print(len(doubleDictionaryList))
# print(len(triDictionaryList))
# tokenMatch(allTokenList,doubleDictionaryList,triDictionaryList,92 ,4,'Output/')
# evaluate('../GroundTruth/BGL_40k.log_structured.csv', 'Output/event.csv')
# print()

# doubleDictionaryList, triDictionaryList, allTokenList  = dictionaryBuilder(BGL_format, '../TestLogs/BGL/BGL_400k.log', BGL_Regex)
# print(len(doubleDictionaryList))
# print(len(triDictionaryList))
# tokenMatch(allTokenList,doubleDictionaryList,triDictionaryList,92 ,4,'Output/')
# evaluate('../GroundTruth/BGL_400k.log_structured.csv', 'Output/event.csv')
# print()


#doubleDictionaryList, triDictionaryList, allTokenList = dictionaryBuilder(BGL_format, 'TestLogs/BGL_500M.log', BGL_Regex)





#Parameters
#HDFS: 15, 10