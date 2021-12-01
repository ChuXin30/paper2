# coding=utf-8
"""
Description : This file implements the Drain algorithm for log parsing
Author      : LogPAI team
License     : MIT
"""

import re
import os
import numpy as np
import pandas as pd
import hashlib
from datetime import datetime
import sys
import pickle
from tqdm import tqdm
import wordninja
# reload(sys)
# sys.setdefaultencoding('utf-8')
import string

def hasDigit(inputString):
    return any(char.isdigit() for char in inputString)


def dword_tep(template):
    dictionary = None
    with open("EngCorpus.pkl", 'rb') as f:
        dictionary = pickle.load(f)
    
    wordset = list()
    for word in template:
        if hasDigit(word):
            continue
        word = word.strip('.:*')
        if word in dictionary:
            wordset.append(word)
        elif all(char.isalpha() for char in word):
            splitted_words = wordninja.split(word)
            for sword in splitted_words:
                if len(sword) <= 2: continue
                wordset.append(sword)
    return wordset


class Logcluster:
    def __init__(self, logTemplate='', logTemplateDword='', logIDL=None):
        self.logTemplate = logTemplate
        self.logTemplateDword = logTemplateDword
        if logIDL is None:
            logIDL = []
        self.logIDL = logIDL


class PreNode:
    """ A node in prefix tree data structure
    """

    def __init__(self, token='', templateNo=0):
        self.logClust = None
        self.token = token
        self.templateNo = templateNo
        self.childD = dict()


class Node:
    def __init__(self, childD=None, depth=0, digitOrtoken=None):
        if childD is None:
            childD = dict()
        self.childD = childD
        self.depth = depth
        self.digitOrtoken = digitOrtoken


class LogParser:
    def __init__(self,  log_messages ,  log_format, indir='./', outdir='./result/', depth=4, st=0.4,
                 maxChild=100, rex=[], keep_para=True, tau=0.5 , cmp = 2):
        """
        Attributes
        ----------
            rex : regular expressions used in preprocessing (step1)
            path : the input path stores the input log file name
            depth : depth of all leaf nodes
            st : similarity threshold
            maxChild : max number of children of an internal node
            logName : the name of the input file containing raw log messages
            savePath : the output path stores the file containing structured logs
        """
        self.log_message = log_messages
        self.tau = tau
        self.path = indir
        self.depth = depth - 2
        self.st = st
        self.maxChild = maxChild
        self.logName = None
        self.savePath = outdir
        self.df_log = None
        self.log_format = log_format
        self.rex = rex
        self.keep_para = keep_para
        self.cmp = cmp

    def hasNumbers(self, s):
        return any(char.isdigit() for char in s)

    def treeSearch(self, rn, seq):
        retLogClust = None

        seqLen = len(seq)

        # 如果序列长度不在解析树中 返回结果为空
        # 我们想做的是即使长度不在这个解析树中 我们也找到与它含义接近的节点
        if seqLen not in rn.childD:
            return 0.0 ,  retLogClust
            # constLogMessL = [w for w in seq if w != '<*>']
            # matchCluster = self.PrefixTreeMatch(preRoot, constLogMessL, 0)
            # if matchCluster is None:
            #     return retLogClust
            # else:
            #     return matchCluster

        parentn = rn.childD[seqLen]

        currentDepth = 1
        for token in seq:
            if currentDepth >= self.depth or currentDepth > seqLen:
                break

            if token in parentn.childD:
                parentn = parentn.childD[token]
            elif '<*>' in parentn.childD:
                parentn = parentn.childD['<*>']
            else:
                return 0.0 ,  retLogClust  # 这里为什么返回？
            currentDepth += 1

        # 如果程序运行到这里说明匹配完成1
        logClustL = parentn.childD
        sim , retLogClust =  self.fastMatch(logClustL, seq)

        return sim ,  retLogClust

    def addSeqToPreTree(self, rootn, newCluster):
        parentn = rootn
        seq = newCluster.logTemplate
        seq = [w for w in seq if w != '<*>']

        for i in range(len(seq)):
            tokenInSeq = seq[i]
            # Match
            if tokenInSeq in parentn.childD:
                parentn.childD[tokenInSeq].templateNo += 1
            # Do not Match
            else:
                parentn.childD[tokenInSeq] = PreNode(token=tokenInSeq, templateNo=1)
            parentn = parentn.childD[tokenInSeq]

        if parentn.logClust is None:
            parentn.logClust = newCluster

    def addSeqToPrefixTree(self, rn, logClust):
        seqLen = len(logClust.logTemplate)
        if seqLen not in rn.childD:
            firtLayerNode = Node(depth=1, digitOrtoken=seqLen)
            rn.childD[seqLen] = firtLayerNode
        else:
            firtLayerNode = rn.childD[seqLen]

        parentn = firtLayerNode

        currentDepth = 1
        for token in logClust.logTemplate:

            # Add current log cluster to the leaf node
            if currentDepth >= self.depth or currentDepth > seqLen:
                if len(parentn.childD) == 0:
                    parentn.childD = [logClust]
                else:
                    parentn.childD.append(logClust)
                break

            # If token not matched in this layer of existing tree.
            if token not in parentn.childD:
                if not self.hasNumbers(token):
                    if '<*>' in parentn.childD:
                        if len(parentn.childD) < self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken=token)
                            parentn.childD[token] = newNode
                            parentn = newNode
                        else:
                            parentn = parentn.childD['<*>']
                    else:
                        if len(parentn.childD) + 1 < self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken=token)
                            parentn.childD[token] = newNode
                            parentn = newNode
                        elif len(parentn.childD) + 1 == self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken='<*>')
                            parentn.childD['<*>'] = newNode
                            parentn = newNode
                        else:
                            parentn = parentn.childD['<*>']

                else:
                    if '<*>' not in parentn.childD:
                        newNode = Node(depth=currentDepth + 1, digitOrtoken='<*>')
                        parentn.childD['<*>'] = newNode
                        parentn = newNode
                    else:
                        parentn = parentn.childD['<*>']

            # If the token is matched
            else:
                parentn = parentn.childD[token]

            currentDepth += 1

    # seq1 is template
    def seqDist(self, seq1, seq2):
        # assert len(seq1) == len(seq2)
        simTokens = 0
        numOfPar = 0

        for token1, token2 in zip(seq1, seq2):
            if token1 == '<*>':
                numOfPar += 1
                continue
            if token1 == token2:
                simTokens += 1
        if len(seq1) != 0:
            retVal = float(simTokens) / len(seq1)
            return retVal, numOfPar
        else:
            return 0 , numOfPar

    def fastMatch(self, logClustL, seq):
        retLogClust = None

        maxSim = -1.0
        maxNumOfPara = -1
        maxClust = None

        for logClust in logClustL:
            curSim, curNumOfPara = self.seqDist(logClust.logTemplate, seq)
            if curSim > maxSim or (curSim == maxSim and curNumOfPara > maxNumOfPara):
                maxSim = curSim
                maxNumOfPara = curNumOfPara
                maxClust = logClust

        if maxSim >= self.st:
            retLogClust = maxClust

        return maxSim , retLogClust

    def getTemplate(self, seq1, seq2):
        retVal = []
        # assert len(seq1) == len(seq2)
        if len(seq1) == len(seq2):
            i = 0
            for word in seq1:
                if word == seq2[i]:
                    retVal.append(word)
                else:
                    retVal.append('<*>')

                i += 1

            return retVal
        else:
            if not seq1:
                return retVal
            seq1 = seq1[::-1]
            i = 0
            for token in seq1:
                i += 1
                if token == seq1[-1]:
                    retVal.append(token)
                    seq1.pop()
                else:
                    retVal.append('<*>')
                if not seq1:
                    break
            if i < len(seq1):
                retVal.append('<*>')
            return retVal

    def outputResult(self, logClustL):
        log_templates = [0] * self.df_log.shape[0]
        log_templateids = [0] * self.df_log.shape[0]
        df_events = []
        for logClust in logClustL:
            template_str = ' '.join(logClust.logTemplate)
            occurrence = len(logClust.logIDL)

            template_id = hashlib.md5(template_str.encode('utf-8')).hexdigest()[0:8]
            # template_id = hashlib.md5(template_str.decode('utf-8')).hexdigest()[0:8]

            for logID in logClust.logIDL:
                logID -= 1
                # print(logID)
                log_templates[logID] = template_str
                log_templateids[logID] = template_id
            df_events.append([template_id, template_str, occurrence])

        # print (logClustL)
        df_event = pd.DataFrame(df_events, columns=['EventId', 'EventTemplate', 'Occurrences'])
        self.df_log['EventId'] = log_templateids
        self.df_log['EventTemplate'] = log_templates

        # print (1)
        # print (self.df_log['EventId'])
        # print (self.df_log['EventTemplate'])
        # print (self.keep_para)
        count = 0
        # if self.keep_para:
        #     # print (2)
        #     count = count + 1
        #     self.df_log["ParameterList"] = self.df_log.apply(self.get_parameter_list, axis=1)
        #     print (count)
        #
        # print (self.df_log["ParameterList"])
        self.df_log.to_csv(os.path.join(self.savePath, self.logName + '_structured.csv'), index=False)
        # print (3)

        occ_dict = dict(self.df_log['EventTemplate'].value_counts())
        df_event = pd.DataFrame()
        df_event['EventTemplate'] = self.df_log['EventTemplate'].unique()
        df_event['EventId'] = df_event['EventTemplate'].map(lambda x: hashlib.md5( x.encode('utf-8')).hexdigest()[0:8])
        df_event['Occurrences'] = df_event['EventTemplate'].map(occ_dict)
        df_event.to_csv(os.path.join(self.savePath, self.logName + '_templates.csv'), index=False,
                        columns=["EventId", "EventTemplate", "Occurrences"])

    def printTree(self, node, dep):
        pStr = ''
        for i in range(dep):
            pStr += '\t'

        if node.depth == 0:
            pStr += 'Root'
        elif node.depth == 1:
            pStr += '<' + str(node.digitOrtoken) + '>'
        else:
            pStr += node.digitOrtoken

        # print(pStr)

        if node.depth == self.depth:
            return 1
        for child in node.childD:
            self.printTree(node.childD[child], dep + 1)

    def PrefixTreeMatch(self, parentn, seq, idx , countMatch ):
        retLogClust = None
        length = len(seq)
        for i in range(idx, length):
            if seq[i] in parentn.childD:
                countMatch = countMatch + 1
                childn = parentn.childD[seq[i]]
                if (childn.logClust is not None):
                    constLM = [w for w in childn.logClust.logTemplate if w != '<*>']
                    if float(len(constLM)) >= self.tau * length:
                        return  float(len(constLM))/length , childn.logClust
                else:
                    return  countMatch/length , self.PrefixTreeMatch(childn, seq, i + 1 , countMatch )

        return countMatch/length , retLogClust

    def LCS(self, seq1, seq2):
        lengths = [[0 for j in range(len(seq2)+1)] for i in range(len(seq1)+1)]
        # row 0 and column 0 are initialized to 0 already
        for i in range(len(seq1)):
            for j in range(len(seq2)):
                if seq1[i] == seq2[j]:
                    lengths[i+1][j+1] = lengths[i][j] + 1
                else:
                    lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])

        # read the substring out from the matrix
        result = []
        lenOfSeq1, lenOfSeq2 = len(seq1), len(seq2)
        while lenOfSeq1!=0 and lenOfSeq2 != 0:
            if lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1-1][lenOfSeq2]:
                lenOfSeq1 -= 1
            elif lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1][lenOfSeq2-1]:
                lenOfSeq2 -= 1
            else:
                assert seq1[lenOfSeq1-1] == seq2[lenOfSeq2-1]
                result.insert(0,seq1[lenOfSeq1-1])
                lenOfSeq1 -= 1
                lenOfSeq2 -= 1
        return result

    def LCSMatch(self, logClustL, seq):
        retLogClust = None

        maxLen = -1
        maxlcs = []
        maxClust = None
        set_seq = set(seq)
        size_seq = len(seq)
        for logClust in logClustL:
            set_template = set( logClust.logTemplate )
#            set_template = set( dword_tep(logClust.logTemplate) )
            if len(set_seq & set_template) <= 0.5 * size_seq:
                continue
            # lcs = self.LCS(seq, logClust.logTemplateDword )
            lcs = self.LCS(seq, logClust.logTemplate)
            # print("lcs={} seq={} dword_tep(logClust.logTemplate)={}".format(lcs, seq, dword_tep(logClust.logTemplate)))

            if len(lcs) > maxLen or (len(lcs) == maxLen and len(logClust.logTemplate) < len(maxClust.logTemplate)):
                maxLen = len(lcs)
                maxlcs = lcs
                maxClust = logClust

        # LCS should be large then tau * len(itself)
        if float(maxLen) >= self.tau * size_seq:
            retLogClust = maxClust

        sim = 0
        if size_seq != 0:
            sim = maxLen*1.0/size_seq

        return  sim ,  retLogClust

    def removeSeqFromPreTree(self, rootn, newCluster):
        parentn = rootn
        seq = newCluster.logTemplate
        seq = [w for w in seq if w != '<*>']

        for tokenInSeq in seq:
            if tokenInSeq in parentn.childD:
                matchedNode = parentn.childD[tokenInSeq]
                if matchedNode.templateNo == 1:
                    del parentn.childD[tokenInSeq]
                    break
                else:
                    matchedNode.templateNo -= 1
                    parentn = matchedNode

    def getTemplateLCS(self, lcs, seq):
        retVal = []
        if not lcs:
            return retVal

        lcs = lcs[::-1]
        i = 0
        for token in seq:
            i += 1
            if token == lcs[-1]:
                retVal.append(token)
                lcs.pop()
            else:
                retVal.append('<*>')
            if not lcs:
                break
        if i < len(seq):
            retVal.append('<*>')
            i+=1
        return retVal

    def parse(self, logName):
        print('Parsing file: ' + os.path.join(self.path, logName))
        start_time = datetime.now()
        self.logName = logName
        rootNode = Node()
        logCluL = []
        print (self.tau)
        self.load_data()
        start_time = datetime.now()

        count = 0
        count_p = 0
        count_lcs_match = 0




        # for idx, line in self.df_log.iterrows():
        #     logID = line['LineId']
        #     logmessageL = self.preprocess(line['Content']).strip().split()
        #     simFix ,  matchCluster = self.treeSearch(rootNode, logmessageL)
        #     if matchCluster is None:
        #         count_p += 1
        #         constLogMessL = [w for w in logmessageL if w != '<*>']
        #         simLcs , matchCluster = self.LCSMatch(logCluL, constLogMessL)
        #         if matchCluster is None:
        #             newCluster = Logcluster(logTemplate=logmessageL, logTemplateDword= dword_tep(logmessageL) , logIDL=[logID])
        #             logCluL.append(newCluster)
        #             self.addSeqToPrefixTree(rootNode, newCluster)
        #         # Add the new log message to the existing cluster
        #         else:
        #             count_lcs_match += 1
        #             newTemplate = self.getTemplateLCS(self.LCS(logmessageL, matchCluster.logTemplate), matchCluster.logTemplate)
        #             if ' '.join(newTemplate) != ' '.join(matchCluster.logTemplate):
        #                 matchCluster.logTemplate = newTemplate
        #     if matchCluster:
        #             matchCluster.logIDL.append(logID)
        # count += 1
        # if count % 1000 == 0 or count == len(self.df_log):
        #     print('Processed {0:.1f}% of log lines.'.format(count * 100.0 / len(self.df_log)))
        # print("count_p={} count_lcs_match={}".format(count_p,count_lcs_match))


        for line_log in  self.log_message.items():
            logID = line_log[1]['LineId']
            logmessageL = line_log[1]['Content']
            simFix ,  matchCluster = self.treeSearch(rootNode, logmessageL)
            if matchCluster is None:
                count_p += 1
                constLogMessL = [w for w in logmessageL if w != '<*>']
                simLcs , matchCluster = self.LCSMatch(logCluL, constLogMessL)
                if matchCluster is None:
                    newCluster = Logcluster(logTemplate=logmessageL, logTemplateDword= dword_tep(logmessageL) , logIDL=[logID])
                    logCluL.append(newCluster)
                    self.addSeqToPrefixTree(rootNode, newCluster)
                # Add the new log message to the existing cluster
                else:
                    count_lcs_match += 1
                    newTemplate = self.getTemplateLCS(self.LCS(logmessageL, matchCluster.logTemplate), matchCluster.logTemplate)
                    if ' '.join(newTemplate) != ' '.join(matchCluster.logTemplate):
                        matchCluster.logTemplate = newTemplate
            if matchCluster :
                matchCluster.logIDL.append(logID)
        count += 1
        if count % 1000 == 0 or count == len(self.df_log):
            print('Processed {0:.1f}% of log lines.'.format(count * 100.0 / len(self.df_log)))
        print("count_p={} count_lcs_match={}".format(count_p,count_lcs_match))


        # for line_log in  self.log_message.items():
        #     logID = line_log[1]['LineId']
        #     logmessageL = line_log[1]['Content']
        #     simFix ,  matchCluster = self.treeSearch(rootNode, logmessageL)
        #     if matchCluster is None:
        #         count_p += 1
        #         constLogMessL = [w for w in logmessageL if w != '<*>']
        #         simLcs , matchCluster = self.LCSMatch(logCluL, constLogMessL)
        #         if matchCluster is None:
        #             newCluster = Logcluster(logTemplate=logmessageL, logTemplateDword= dword_tep(logmessageL) , logIDL=[logID])
        #             logCluL.append(newCluster)
        #             self.addSeqToPrefixTree(rootNode, newCluster)
        #         # Add the new log message to the existing cluster
        #         else:
        #             count_lcs_match += 1
        #             newTemplate = self.getTemplate(self.LCS(logmessageL, matchCluster.logTemplate), matchCluster.logTemplate)
        #             if ' '.join(newTemplate) != ' '.join(matchCluster.logTemplate):
        #                 matchCluster.logTemplate = newTemplate
        #     if matchCluster:
        #         constLogMessL = line_log[1]['dwords']
        #         simLcs , matchClusterLCS = self.LCSMatch(logCluL, constLogMessL)
        #         if(simLcs >= self.cmp ) :
        #             count_lcs_match += 1
        #             newTemplate = self.getTemplate(self.LCS(logmessageL, matchClusterLCS.logTemplate), matchClusterLCS.logTemplate)
        #             if ' '.join(newTemplate) != ' '.join(matchClusterLCS.logTemplate):
        #                 matchClusterLCS.logTemplate = newTemplate
        #             matchClusterLCS.logIDL.append(logID)
        #         else:
        #             matchCluster.logIDL.append(logID)
        # count += 1
        # if count % 1000 == 0 or count == len(self.df_log):
        #     print('Processed {0:.1f}% of log lines.'.format(count * 100.0 / len(self.df_log)))
        # print("count_p={} count_lcs_match={}".format(count_p,count_lcs_match))

        # for idx, line in self.df_log.iterrows():
        #     logID = line['LineId']
        #     logmessageL = self.preprocess(line['Content']).strip().split()
        #     # logmessageL = filter(lambda x: x != '', re.split('[\s=:,]', self.preprocess(line['Content'])))
        #     # print (count)
        #     # print (   logmessageL)
        #     #固定深度树匹配的结果
        #     simFix ,  matchCluster = self.treeSearch(rootNode, logmessageL)
        #     # print (fixSim)
        #     # #前缀树匹配的结果
        #     # constLogMessL = [w for w in logmessageL if w != '<*>']
        #     # preSim , matchClusterPre = self.PrefixTreeMatch(rootPreNode , constLogMessL, 0 , 0.1)
        #     # print (preSim)
        #     #LCS
        #     # print (lcsSim)
        #     # constLogMessL = [w for w in logmessageL if w != '<*>']
        #     # simLcs , matchCluster_lcs = self.LCSMatch(logCluL, constLogMessL)
        #     # # 1
        #     # if matchCluster is None and matchCluster_lcs is None:
        #     #     newCluster = Logcluster(logTemplate=logmessageL, logIDL=[logID])
        #     #     logCluL.append(newCluster)
        #     #     self.addSeqToPrefixTree(rootNode, newCluster)
        #     # elif matchCluster is None and matchCluster_lcs is not None
        #     # Match no existing log cluste
        #     if matchCluster is None:
        #         count_p += 1
        #         constLogMessL = [w for w in logmessageL if w != '<*>']
        #         matchCluster = self.LCSMatch(logCluL, constLogMessL)
        #         if matchCluster is None:
        #             newCluster = Logcluster(logTemplate=logmessageL, logIDL=[logID])
        #             logCluL.append(newCluster)
        #             self.addSeqToPrefixTree(rootNode, newCluster)
        #         # Add the new log message to the existing cluster
        #         else:
        #             newTemplate = self.getTemplate(self.LCS(logmessageL, matchCluster.logTemplate), matchCluster.logTemplate)
        #             if ' '.join(newTemplate) != ' '.join(matchCluster.logTemplate):
        #                 matchCluster.logTemplate = newTemplate
        #     if matchCluster:
        #         matchCluster.logIDL.append(logID)
        #     # if matchCluster:
        #     #     newTemplate = self.getTemplate(logmessageL, matchCluster.logTemplate)
        #     #     matchCluster.logIDL.append(logID)
        #     #     if ' '.join(newTemplate) != ' '.join(matchCluster.logTemplate): 
        #     #         matchCluster.logTemplate = newTemplate

            # count += 1
            # if count % 1000 == 0 or count == len(self.df_log):
            #     print('Processed {0:.1f}% of log lines.'.format(count * 100.0 / len(self.df_log)))



        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)
        # print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))
        self.outputResult(logCluL)

        # print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))

    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)
        # print ("headers")
        # print (headers)
        # print ("regex")
        # print (regex)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logName), regex, headers, self.log_format)
        # print (self.df_log)

    def preprocess(self, line):
        for currentRex in self.rex:
            line = re.sub(currentRex, '<*>', line)
        return line

    def log_to_dataframe(self, log_file, regex, headers, logformat):
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

    def generate_logformat_regex(self, logformat):
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

    # def get_parameter_list(self, row):
    #     template_regex = re.sub(r"<.{1,5}>", "<*>", row["EventTemplate"])
    #     if "<*>" not in template_regex: return []
    #     template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template_regex)
    #     template_regex = re.sub(r'\\ +', r'\s+', template_regex)
    #     template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
    #     parameter_list = re.findall(template_regex, row["Content"])
    #     parameter_list = parameter_list[0] if parameter_list else ()
    #     parameter_list = list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list]
    #     return parameter_list

    def get_parameter_list(self, row):
        # print (row)
        template_regex = re.sub(r"\s<.{1,5}>\s", "<*>", row["EventTemplate"])
        if "<*>" not in template_regex: return []
        template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template_regex)
        template_regex = re.sub(r'\\ +', r'[^A-Za-z0-9]+', template_regex)
        template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
        parameter_list = re.findall(template_regex, row["Content"])
        parameter_list = parameter_list[0] if parameter_list else ()
        parameter_list = list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list]
        parameter_list = [para.strip(string.punctuation).strip(' ') for para in parameter_list]
        return parameter_list