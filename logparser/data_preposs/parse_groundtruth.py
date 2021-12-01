# -*- coding: utf-8 -*
from collections import defaultdict
import pandas as pd
import re
import datetime
import  random
import csv
import  random
import hashlib



def generate_logformat_regex(logformat):
    """ Function to generate regular expression to split log messages
    """
    headers = []
    splitters = re.split(r'(<[^<>]+>)', logformat)
    regex = ''
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(' +', '\\\s+', splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip('<').strip('>')
            regex += '(?P<%s>.*?)' % header
            headers.append(header)
    regex = re.compile('^' + regex + '$')
    return headers, regex


def log_to_dataframe(log_file, regex, headers):
    """ Function to transform log file to dataframe 
    """
    log_messages = []
    linecount = 0
    with open(log_file, 'r') as fin:
        for line in fin.readlines():
            try:
                match = regex.search(line.strip())
                message = [match.group(header) for header in headers]
                log_messages.append(message)
                linecount += 1
            except Exception as e:
                pass
    logdf = pd.DataFrame(log_messages, columns=headers)
    logdf.insert(0, 'LineId', None)
    logdf['LineId'] = [i + 1 for i in range(linecount)]
    return logdf



def parse_hdfs(row_log , log_template , logformat , save_csv ):
    f_row_log = open(row_log , mode='r')
    headers, regex = generate_logformat_regex(logformat)
    df_log = log_to_dataframe(row_log , regex, headers)
    dict_logkey_parse={}  

    f_template = open(log_template,mode='r')
    for line in f_template:
        line = line.rstrip()
        obj = re.match(r'([0-9]+).(.*)',line)
        dict_logkey_parse[obj.group(1)] = obj.group(2)

    # log_template_csv = pd.read_csv(log_template)
    # log_template_csv = log_template_csv.values.tolist()
    # for line in log_template_csv:
    #     print(line)
    #     dict_logkey_parse[line[0]] = line[1][3:-3].replace( '<*>' , '(.*)' )


    print(dict_logkey_parse)
    count_match = 0

    df_events = []
    df_tmp = []
    LineId = 0

    list_eventid = []

    for line in f_row_log.readlines():
        eventid = ''
        LineId += 1
        if(LineId%1000 == 0):
            print("LineId={}".format(LineId))
        for var in dict_logkey_parse:
            obj = re.search(dict_logkey_parse[var], line)
            if obj != None:
                count_match += 1
                eventid = var
                break
        if(eventid == ''):
            print(line)
        if eventid != '':
            list_eventid.append( [eventid , dict_logkey_parse[var] ] )
            df_events.append([LineId, eventid])

    df_event = pd.DataFrame(df_events, columns=['LineId', 'EventId'])
    df_event.to_csv(save_csv , index=False, columns=['LineId', 'EventId'])
    print("count_match={}".format(count_match))

    # logdf = pd.DataFrame(list_eventid, columns=['EventId', 'EventTemplate'])
    # df_tmp = pd.DataFrame()
    # df_tmp['EventTemplate'] = logdf['EventTemplate'].unique()
    # df_tmp['EventId'] = df_tmp['EventTemplate'].map(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest()[0:8])
    # df_tmp.to_csv(save_tmp , index=False, columns=['EventId', 'EventTemplate'])

    # for idx, line in df_log.iterrows():
    #     # print(line)
    #     log_line = line['Content']        
    #     eventid = ''
    #     for var in dict_logkey_parse:
    #         obj = re.search( dict_logkey_parse[var], log_line)
    #         if obj != None:
    #             count_match += 1
    #             eventid = var
    #             break
    #     if(eventid == ''):
    #         print(log_line)
    

def parse_bgl(row_log , log_template   , save_csv ):
    f_row_log = open(row_log , mode='r')
    # headers, regex = generate_logformat_regex(logformat)
    # df_log = log_to_dataframe(row_log , regex, headers)
    dict_logkey_parse={}  

    # f_template = open(log_template,mode='r')
    # for line in f_template:
    #     line = line.rstrip()
    #     obj = re.match(r'([0-9]+).(.*)',line)
    #     dict_logkey_parse[obj.group(1)] = obj.group(2)

    log_template_csv = pd.read_csv(log_template)
    log_template_csv = log_template_csv.values.tolist()
    for line in log_template_csv:
        # print(line)
        dict_logkey_parse[line[0]] = line[1].replace('(' , '\(').replace(')','\)').replace( '<*>' , '(.*)' )


    # print(dict_logkey_parse)
    count_match = 0

    df_events = []
    LineId = 0


        # for var in dict_logkey_parse:
        #     obj = re.search(dict_logkey_parse[var], line)
        #     if obj != None:
        #         count_match += 1
        #         eventid = var
        #         break

    count_line = 0
    last_match = ""
    for line in f_row_log.readlines():
        line_tmp = line
        eventid = ''
        LineId += 1
        count_line += 1

        if(count_line%10000 == 0):
            print("count_line={}".format(count_line))

        if dict_logkey_parse.has_key(last_match):
            obj1 = re.search(dict_logkey_parse[last_match], line_tmp) 
            if obj1 != None:
                count_match += 1
                eventid = last_match
        if eventid == '':
            for var in dict_logkey_parse:
                obj = re.search(dict_logkey_parse[var], line_tmp)
                if obj != None:
                    # print(2)
                    count_match += 1
                    eventid = var
                    break

        if(eventid == ''):
            print(count_line)
            print(line)
        if eventid != '':
            df_events.append([LineId, eventid])
            last_match = eventid

    df_event = pd.DataFrame(df_events, columns=['LineId', 'EventId'])
    df_event.to_csv(save_csv , index=False, columns=['LineId', 'EventId'])
    print("count_match={}".format(count_match))


def parse_csv(row_log , log_template, logformat  , save_csv ):

    headers, regex = generate_logformat_regex(logformat)
    df_log = log_to_dataframe(row_log , regex, headers)

    dict_logkey_parse={}  

    log_template_csv = pd.read_csv(log_template)
    log_template_csv = log_template_csv.values.tolist()
    for line in log_template_csv:
        dict_logkey_parse[line[0]] = line[1].replace('|' , '\|').replace('[' , '\[').replace(']' , '\]').replace('(' , '\(').replace(')','\)').replace( '<*>' , '(.*)' )

    count_match = 0

    df_events = []
    LineId = 0
    count_line = 0
    last_match = ""


    dict_not_match = {}
    LineId_not_match = {}
    for idx, line in df_log.iterrows():
    # for line in f_row_log.readlines():
        line_tmp = line['Content']
        eventid = ''
        LineId += 1
        count_line += 1

        if(count_line%1000 == 0):
            print("count_line={}".format(count_line))

        if dict_logkey_parse.has_key(last_match):
            # print(dict_logkey_parse[last_match])
            obj1 = re.search(dict_logkey_parse[last_match], line_tmp) 
            if obj1 != None:
                count_match += 1
                eventid = last_match
        if eventid == '':
            for var in dict_logkey_parse:
                # print(dict_logkey_parse[var])
                obj = re.search(dict_logkey_parse[var], line_tmp)
                if obj != None:
                    count_match += 1
                    eventid = var
                    break

        if(eventid == ''):
            # print(count_line)
            # print(line['Content'])
            # df_events.append([LineId, '-1'])
            LineId_not_match[LineId] = 1

            if dict_not_match.has_key(line['Content']) :
                dict_not_match[line['Content'] ] = dict_not_match.get( line['Content'] ) + 1
            else:
                dict_not_match[line['Content'] ] = 1

        if eventid != '':
            df_events.append([LineId, eventid])
            last_match = eventid

    df_event = pd.DataFrame(df_events, columns=['LineId', 'EventId'])
    df_event.to_csv(save_csv , index=False, columns=['LineId', 'EventId'])
    print("count_line={} count_match={}".format(count_line , count_match))
    print(sorted(dict_not_match.items(), key = lambda kv:(kv[1], kv[0])))     

    # f_row_log_read = open(row_log  , mode='r')
    # f_row_log_write = open(row_log + '.match' , mode='w')
    # LineId = 0
    # count_reduce_log = 0
    # for line in f_row_log_read.readlines():
    #     LineId += 1
    #     if not LineId_not_match.has_key(LineId):
    #         f_row_log_write.write(line)
    #         count_reduce_log += 1
    # print("count_reduce_log={}".format(count_reduce_log))


def parse_match_file(row_log , log_template,  row_log_write ):
    
    f_row_log_read = open(row_log  , mode='r')
    f_row_log_write = open(row_log_write , mode='w')

    dict_logkey_parse={}  

    log_template_csv = pd.read_csv(log_template)
    log_template_csv = log_template_csv.values.tolist()
    for line in log_template_csv:
        dict_logkey_parse[line[0]] = line[1].replace('|' , '\|').replace('[' , '\[').replace(']' , '\]').replace('(' , '\(').replace(')','\)').replace( '<*>' , '(.*)' )

    count_match = 0
    LineId = 0
    count_line = 0
    last_match = ""
    count_reduce_log = 0
    dict_not_match = {}
    LineId_not_match = {}

    for  line in f_row_log_read.readlines():
        line_tmp = line
        eventid = ''
        LineId += 1
        count_line += 1
        if(count_line%1000 == 0):
            print("count_line={}".format(count_line))
        if dict_logkey_parse.has_key(last_match):
            obj1 = re.search(dict_logkey_parse[last_match], line_tmp) 
            if obj1 != None:
                count_match += 1
                eventid = last_match
        if eventid == '':
            for var in dict_logkey_parse:
                obj = re.search(dict_logkey_parse[var], line_tmp)
                if obj != None:
                    count_match += 1
                    eventid = var
                    break
        if(eventid == ''):
            LineId_not_match[LineId] = 1
            if dict_not_match.has_key(line_tmp) :
                dict_not_match[line_tmp ] = dict_not_match.get( line_tmp ) + 1
            else:
                dict_not_match[line_tmp ] = 1

        if eventid != '':
            f_row_log_write.write(line)
            last_match = eventid
            count_reduce_log += 1
    print("count_reduce_log={}".format(count_reduce_log))    
    print("count_line={} count_match={}".format(count_line , count_match))
    print(sorted(dict_not_match.items(), key = lambda kv:(kv[1], kv[0])))     


def parse_csv_hdfs(row_log , log_template, logformat  , save_csv ):
    headers, regex = generate_logformat_regex(logformat)
    df_log = log_to_dataframe(row_log , regex, headers)

    dict_logkey_parse={}  

    f_template = open(log_template,mode='r')
    for line in f_template:
        line = line.rstrip()
        obj = re.match(r'([0-9]+).(.*)',line)
        dict_logkey_parse[obj.group(1)] = obj.group(2)

    count_match = 0

    df_events = []
    LineId = 0
    count_line = 0
    last_match = ""


    dict_not_match = {}
    LineId_not_match = {}
    for idx, line in df_log.iterrows():
    # for line in f_row_log.readlines():
        line_tmp = line['Content']
        eventid = ''
        LineId += 1
        count_line += 1

        if(count_line%1000 == 0):
            print("count_line={}".format(count_line))

        if dict_logkey_parse.has_key(last_match):
            # print(dict_logkey_parse[last_match])
            obj1 = re.search(dict_logkey_parse[last_match], line_tmp) 
            if obj1 != None:
                count_match += 1
                eventid = last_match
        if eventid == '':
            for var in dict_logkey_parse:
                # print(dict_logkey_parse[var])
                obj = re.search(dict_logkey_parse[var], line_tmp)
                if obj != None:
                    count_match += 1
                    eventid = var
                    break

        if(eventid == ''):
            # print(count_line)
            print(line['Content'])
            # df_events.append([LineId, '-1'])
            LineId_not_match[LineId] = 1

            if dict_not_match.has_key(line['Content']) :
                dict_not_match[line['Content'] ] = dict_not_match.get( line['Content'] ) + 1
            else:
                dict_not_match[line['Content'] ] = 1

        if eventid != '':
            df_events.append([LineId, eventid])
            last_match = eventid

    df_event = pd.DataFrame(df_events, columns=['LineId', 'EventId'])
    df_event.to_csv(save_csv , index=False, columns=['LineId', 'EventId'])
    print("count_line={} count_match={}".format(count_line , count_match))
    print(sorted(dict_not_match.items(), key = lambda kv:(kv[1], kv[0])))     


if __name__ == "__main__":

    parse_csv_hdfs('../logs/HDFS/HDFS.log' ,   '../logs/HDFS/col_header.txt', '<Date> <Time> <Pid> <Level> <Component>: <Content>' , '../logs/HDFS/HDFS.log_structured.csv')


    # parse_match_file('../logs/OpenSSH/OpenSSH_600.log' ,   '../logs/OpenSSH/OpenSSH_2k.log_templates.csv' ,  '../logs/OpenSSH/OpenSSH_600_match.log' )
    # parse_match_file('../logs/OpenSSH/OpenSSH_6k.log' ,    '../logs/OpenSSH/OpenSSH_2k.log_templates.csv' ,  '../logs/OpenSSH/OpenSSH_6k_match.log' )
    # parse_match_file('../logs/OpenSSH/OpenSSH_60k.log' ,   '../logs/OpenSSH/OpenSSH_2k.log_templates.csv' ,  '../logs/OpenSSH/OpenSSH_60k_match.log' )
    # parse_match_file('../logs/OpenSSH/OpenSSH_600k.log' ,  '../logs/OpenSSH/OpenSSH_2k.log_templates.csv' ,  '../logs/OpenSSH/OpenSSH_600k_match.log' )

    # parse_csv('../logs/OpenSSH/OpenSSH_600_match.log' ,   '../logs/OpenSSH/OpenSSH_2k.log_templates.csv',  '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>' ,  '../logs/OpenSSH/OpenSSH_600_match.log_structured.csv' ,)
    # parse_csv('../logs/OpenSSH/OpenSSH_6k_match.log' ,   '../logs/OpenSSH/OpenSSH_2k.log_templates.csv',  '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>' ,  '../logs/OpenSSH/OpenSSH_6k_match.log_structured.csv' ,)
    # parse_csv('../logs/OpenSSH/OpenSSH_60k_match.log' ,   '../logs/OpenSSH/OpenSSH_2k.log_templates.csv',  '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>' ,  '../logs/OpenSSH/OpenSSH_60k_match.log_structured.csv' ,)
    # parse_csv('../logs/OpenSSH/OpenSSH_600k_match.log' ,   '../logs/OpenSSH/OpenSSH_2k.log_templates.csv',  '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>' ,  '../logs/OpenSSH/OpenSSH_600k_match.log_structured.csv' ,)


    # parse_match_file('../logs/Spark/Spark_100k.log' ,   '../logs/Spark/Spark.log_templates.csv' ,  '../logs/Spark/Spark_100k_match.log' )
    # parse_match_file('../logs/Spark/Spark_10k.log' ,   '../logs/Spark/Spark.log_templates.csv' ,  '../logs/Spark/Spark_10k_match.log' )
    # parse_match_file('../logs/Spark/Spark_1k.log' ,   '../logs/Spark/Spark.log_templates.csv' ,  '../logs/Spark/Spark_1k_match.log' )
    # parse_match_file('../logs/Spark/Spark_1m.log' ,   '../logs/Spark/Spark.log_templates.csv' ,  '../logs/Spark/Spark_1m_match.log' )

    # parse_csv('../logs/Spark/Spark_1k_match.log' ,   '../logs/Spark/Spark.log_templates.csv',  '<Date> <Time> <Level> <Component>: <Content>' ,  '../logs/Spark/Spark_1k_match.log_structured.csv' ,)
    # parse_csv('../logs/Spark/Spark_10k_match.log' ,   '../logs/Spark/Spark.log_templates.csv',  '<Date> <Time> <Level> <Component>: <Content>' ,  '../logs/Spark/Spark_10k_match.log_structured.csv' ,)
    # parse_csv('../logs/Spark/Spark_100k_match.log' ,   '../logs/Spark/Spark.log_templates.csv',  '<Date> <Time> <Level> <Component>: <Content>' ,  '../logs/Spark/Spark_100k_match.log_structured.csv' ,)
    # parse_csv('../logs/Spark/Spark_1m_match.log' ,   '../logs/Spark/Spark.log_templates.csv',  '<Date> <Time> <Level> <Component>: <Content>' ,  '../logs/Spark/Spark_1m_match.log_structured.csv' ,)

    # parse_csv('../logs/Spark/Spark_1m.log' ,   '../logs/Spark/Spark_2k.log_templates.csv',  '<Date> <Time> <Level> <Component>: <Content>' ,  '../logs/Spark/Spark_1m.log_structured.csv.no_use' ,)
    # parse_csv('../logs/Spark/Spark_100k.log' ,   '../logs/Spark/Spark_2k.log_templates.csv',  '<Date> <Time> <Level> <Component>: <Content>' ,  '../logs/Spark/Spark_100k.log_structured.csv.no_use' ,)
    # parse_csv('../logs/Spark/Spark_10k.log' ,   '../logs/Spark/Spark_2k.log_templates.csv',  '<Date> <Time> <Level> <Component>: <Content>' ,  '../logs/Spark/Spark_10k.log_structured.csv.no_use' ,)
    # parse_csv('../logs/Spark/Spark_1k.log' ,   '../logs/Spark/Spark_2k.log_templates.csv',  '<Date> <Time> <Level> <Component>: <Content>' ,  '../logs/Spark/Spark_1k.log_structured.csv.no_use' ,)

    # parse_csv('../logs/Spark/Spark_1m.log.match.match' ,   '../logs/Spark/Spark_2k.log_templates.csv',  '<Date> <Time> <Level> <Component>: <Content>' ,  '../logs/Spark/Spark_1m.log_structured.csv' ,)
    # parse_csv('../logs/Spark/Spark_100k.log.match' ,   '../logs/Spark/Spark_2k.log_templates.csv',  '<Date> <Time> <Level> <Component>: <Content>' ,  '../logs/Spark/Spark_100k.log_structured.csv' ,)
    # parse_csv('../logs/Spark/Spark_10k.log.match' ,   '../logs/Spark/Spark_2k.log_templates.csv',  '<Date> <Time> <Level> <Component>: <Content>' ,  '../logs/Spark/Spark_10k.log_structured.csv' ,)
    # parse_csv('../logs/Spark/Spark_1k.log.match' ,   '../logs/Spark/Spark_2k.log_templates.csv',  '<Date> <Time> <Level> <Component>: <Content>' ,  '../logs/Spark/Spark_1k.log_structured.csv' ,)


    # parse_csv('../logs/Spark/Spark_100.log' ,   '../logs/Spark/Spark.log_templates.csv',  '<Date> <Time> <Level> <Component>: <Content>' ,  '../logs/Spark/Spark_100.log_structured.csv' ,)
    # parse_csv('../logs/Spark/Spark_1k.log' ,   '../logs/Spark/Spark.log_templates.csv',  '<Date> <Time> <Level> <Component>: <Content>' ,  '../logs/Spark/Spark_1k.log_structured.csv' ,)
    # parse_csv('../logs/Spark/Spark_10k.log' ,   '../logs/Spark/Spark.log_templates.csv',  '<Date> <Time> <Level> <Component>: <Content>' ,  '../logs/Spark/Spark_10k.log_structured.csv' ,)
    # parse_csv('../logs/Spark/Spark_100k.log' ,   '../logs/Spark/Spark.log_templates.csv',  '<Date> <Time> <Level> <Component>: <Content>' ,  '../logs/Spark/Spark_100k.log_structured.csv' ,)
    # parse_csv('../logs/Spark/Spark_1m.log' ,   '../logs/Spark/Spark.log_templates.csv',  '<Date> <Time> <Level> <Component>: <Content>' ,  '../logs/Spark/Spark_1m.log_structured.csv' ,)


    # parse_csv('../logs/Andriod/Andriod.log' ,   '../logs/Andriod/Andriod.log_templates.csv',  '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>' ,  '../logs/Andriod/Andriod.log_structured.csv' ,)

    # parse_csv('../logs/OpenSSH/OpenSSH_600.log' ,   '../logs/OpenSSH/OpenSSH.log_templates.csv',  '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>' ,  '../logs/OpenSSH/OpenSSH_600.log_structured.csv' ,)
    # parse_csv('../logs/OpenSSH/OpenSSH_6k.log' ,   '../logs/OpenSSH/OpenSSH.log_templates.csv',  '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>' ,  '../logs/OpenSSH/OpenSSH_6k.log_structured.csv' ,)
    # parse_csv('../logs/OpenSSH/OpenSSH_60k.log' ,   '../logs/OpenSSH/OpenSSH.log_templates.csv',  '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>' ,  '../logs/OpenSSH/OpenSSH_60k.log_structured.csv' ,)
    # parse_csv('../logs/OpenSSH/OpenSSH_600k.log' ,   '../logs/OpenSSH/OpenSSH.log_templates.csv',  '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>' ,  '../logs/OpenSSH/OpenSSH_600k.log_structured.csv' ,)



    # parse_csv('../logs/BGL/BGL_100.log' ,   '../logs/BGL/BGL_templates.csv', '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>' , '../logs/BGL/BGL_100.log_structured.csv')
    # parse_csv('../logs/BGL/BGL_1k.log' ,   '../logs/BGL/BGL_templates.csv', '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>' , '../logs/BGL/BGL_1k.log_structured.csv')
    # parse_csv('../logs/BGL/BGL_10k.log' ,   '../logs/BGL/BGL_templates.csv','<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>' , '../logs/BGL/BGL_10k.log_structured.csv')
    # parse_csv('../logs/BGL/BGL_100k.log' ,   '../logs/BGL/BGL_templates.csv','<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>' , '../logs/BGL/BGL_100k.log_structured.csv')

    # parse_csv('../logs/BGL/BGL_400.log' ,   '../logs/BGL/BGL_templates.csv', '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>' ,'../logs/BGL/BGL_400.log_structured.csv')
    # parse_csv('../logs/BGL/BGL_4k.log' ,   '../logs/BGL/BGL_templates.csv', '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>' ,'../logs/BGL/BGL_4k.log_structured.csv')
    # parse_csv('../logs/BGL/BGL_40k.log' ,   '../logs/BGL/BGL_templates.csv', '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>' ,'../logs/BGL/BGL_40k.log_structured.csv')
    # parse_csv('../logs/BGL/BGL_400k.log' ,   '../logs/BGL/BGL_templates.csv', '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>' ,'../logs/BGL/BGL_400k.log_structured.csv')

    # parse_csv('../logs/BGL/BGL_1m.log' ,   '../logs/BGL/BGL_templates.csv', '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>'  , '../logs/BGL/BGL_1m.log_structured.csv')


    # parse_hdfs('../logs/HDFS/HDFS_1k.log' ,   '../logs/HDFS/col_header.txt', '<Date> <Time> <Pid> <Level> <Component>: <Content>' , '../logs/HDFS/HDFS_1k.log_structured.csv' , '../logs/HDFS/HDFS_1k.log_templates.csv' )
    # parse_hdfs('../logs/HDFS/HDFS_10k.log' ,   '../logs/HDFS/col_header.txt', '<Date> <Time> <Pid> <Level> <Component>: <Content>' , '../logs/HDFS/HDFS_10k.log_structured.csv')
    # parse_hdfs('../logs/HDFS/HDFS_2k.log' ,   '../logs/HDFS/HDFS_templates.csv', '<Date> <Time> <Pid> <Level> <Component>: <Content>' , '../logs/HDFS/HDFS_10k.log_structured.csv')
    # parse_hdfs('../logs/HDFS/HDFS_100k.log' ,   '../logs/HDFS/col_header.txt', '<Date> <Time> <Pid> <Level> <Component>: <Content>' , '../logs/HDFS/HDFS_100k.log_structured.csv')
    # parse_hdfs('../logs/HDFS/HDFS_1m.log' ,   '../logs/HDFS/col_header.txt', '<Date> <Time> <Pid> <Level> <Component>: <Content>' , '../logs/HDFS/HDFS_1m.log_structured.csv')
    # parse_hdfs('../logs/HDFS/HDFS_10m.log' ,   '../logs/HDFS/col_header.txt', '<Date> <Time> <Pid> <Level> <Component>: <Content>' , '../logs/HDFS/HDFS_10m.log_structured.csv')
