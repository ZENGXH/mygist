from __future__ import with_statement
import sys
import subprocess
import json
from helper import *
import os.path


## update itself
## parse log itself
class ExpContainer(object):
    def __init__(self, exp_config):
        self.title = exp_config['title']
        self.server_path = exp_config['server_path']
        self.average_interval =  exp_config['average_interval'] 
        self.need_update = exp_config['need_update'] 
        self.skill_score = exp_config['skill_score']
        self.path = self.title #exp_config['path']
        self.load_score_from_log = exp_config.get('load_score_from_log', True)
        self.iter_list = []
        self.epoch_list = []
        self.train_loss_list = []
        self.valid_loss_list = []
        self.train_log = ""
        self.score_dict_test = {}
        self.need_sorted = exp_config.get('need_sorted', 0)
        self.score_dict_valid = {}
        if self.need_update:
            self.UpdateData()

    def UpdateData(self):
        if not os.path.isdir(self.title):
            os.mkdir(self.title)
        self.rsync(self.server_path+'/train', self.title)
        print 'UpdateData done'
     
    def rsync(self, path, dest):
        cmd = ["rsync -avrP %s %s/" % (path, dest)]
        print 'cmd ', cmd
        return subprocess.call(cmd, shell=True)


    def GetTrainLog(self):
        # bad! 
        log = 'log' ## ????what
        title = self.title 
        path = self.path
        print '[GetTrainLog] of ', title
        fileList, found = Iglob(path, "*log*")
        print '[ReadFileList] find log in %s get list %d '%(path, len(fileList))
        print fileList
        if len(fileList) == 1:
            self.UpdateData()
            fileList, found = Iglob(path, "*log*")
        assert(len(fileList) == 1)
        target_file = fileList[0]

        loss_file = open(target_file, 'r')
        lines = loss_file.readlines()
        loss_file.close()
        # train_log = ReadFileList(exp_info, average_interval)
        """ 
        exp_info['title']+'/train/train.log'
        if not os.path.isfile(train_log):
            print 'log not found`'
            return False, ''
        log = open(train_log, 'r')
        lines = log.readlines()
        """
        self.train_log = lines 
        return True

    def LoadExperiment(self):
        ## load self: iter_list, train_loss_list, valid_loss_list, epoch_list, score_dict_cat
        ## load loss and skill_score
        title = self.title 
        
        print 'LoadExperimentSet'

        ## LoadTrainLog
        is_suc = self.GetTrainLog()
        # exp_dict_tmp['train_log'] = train_log
        if not is_suc:
            return
        else:
            print '[LoadExperiment] read log of ', self.title
            self.ReadTrainLog()

        ## LoadSkillScore 
        if self.load_score_from_log:
            print 'LoadSkillScore'
            self.LoadSkillScore()
        else:
            raise NotImplementedError

        # exp_dict[exp_info['id']] = exp_dict_tmp
        # score_dict_cat = [score_dict_test, score_dict_valid]

    def ReadTrainLog(self): 
        lines = self.train_log
        average_interval = self.average_interval
        ## return iter_list, train_loss_list, valid_loss_list
        iter_num = 0
        acc_train_loss = 0
        for ind, line in enumerate(lines):
            if "INFO - Validation Cost:" not in line and "Minibatch Cost:" not in line:
                continue
            if "Validation Cost" in line:
                valid_loss_num = float(line.split('Validation Cost: ')[1].split(' Time Spent:')[0])
                # remove the repeat one
                if len(self.valid_loss_list) and valid_loss_num == self.valid_loss_list[-1]:
                    continue
                self.valid_loss_list.append(valid_loss_num) 
                if 'epoch' in line:
                    current_epoch = int(line.split('epoch: ')[1].split(',')[0])
                    self.epoch_list.append(current_epoch)
                else:
                    self.epoch_list.append(len(self.epoch_list) + 1)
            elif "Minibatch" in line:
                iter_num += 1
                train_loss_num = float(line.split('Minibatch Cost: ')[1].split(' Time Spent:')[0].strip(':'))
            if math.isnan(train_loss_num) or 9999 < train_loss_num:
                train_loss_num = 9999
            acc_train_loss += train_loss_num
            if 0 == (int(iter_num) % average_interval):
                self.train_loss_list.append(float(acc_train_loss / average_interval))
                self.iter_list.append(int(iter_num))
                acc_train_loss = 0
        print '[ReadTrainLog] return iter %d, epoch %d'%(len(self.iter_list), len(self.epoch_list))
        assert(len(self.iter_list) == len(self.train_loss_list))
        assert(len(self.epoch_list) == len(self.valid_loss_list))

    # encoding: utf-8
    def LoadSkillScore(self):
        exp_name = self.title
        lines = self.train_log
        skill_score = self.skill_score

        score_dict_test = {}
        score_dict_valid = {}
        epoch_list = []
        for score_name in skill_score:
            score_dict_test[score_name] = []
            score_dict_valid[score_name] = []

        # score_name = ''
        for id, line in enumerate(lines):
            if '[Test]' in line:
                current_epoch = int(line.split('epoch: ')[1].split(',')[0])
                if current_epoch not in epoch_list:
                    epoch_list.append(int(line.split('epoch: ')[1].split(',')[0]))

                if 'score' in line:
                    score_name = line.split('score: ')[1]
                    value = score_name.split(':')[1].split('end')[0]
                    score_name = score_name.split(':')[0]
                    test_name = 'skill_score_%s_epoch%d'%(score_name, current_epoch)
                    # get scorevalue    
                    buf = value
                    i = id + 1
                    next_line = lines[i]
                    if 'end' not in line:
                        while i < len(lines):
                            i = i + 1
                            buf += next_line
                            if 'end' in next_line:
                                break
                            next_line = lines[i]
                    buf = buf.split('end')[0].split(' ')
                    buf =[x.strip('\n').strip('[').strip(']') for x in buf if x not in [' ','[', ''] ]
                    buf = [float(x) for x in buf if x not in ['']] 
                    
                    if 'test_data' in line:
                        test_name = 'test_' + test_name
                        score_dict_test[score_name].append(
                            {
                                "test_name": test_name,    
                                "score": buf,
                                "exp_name": exp_name 
                            })
                    elif 'valid_data' in line:
                        test_name = 'valid_' + test_name
                        score_dict_valid[score_name].append(
                            {
                                "test_name": test_name,    
                                "score": buf,
                                "exp_name": exp_name 
                            })   
        self.score_dict_test = score_dict_test
        self.score_dict_valid = score_dict_valid

    def DrawSkillScore(self, savepath, datatype):
        print 'DrawSkillScore_ for ', self.title, ' save in ', savepath
        for score_name in self.skill_score:
            self.DrawSkillScore_(savepath, score_name, datatype)

    def DrawSkillScore_(self, savepath, score_name, datatype='test_data'):
        # def DrawSkillScore(savepath, exp_name, score_set, score_name, opt={}):
        """
            score_set: 
            [
                {
                    (opt)"test_time": .. ,
                    "score": [n,n,n], 
                    'exp_name': ""
                },    
            ]

        """
        exp_name = self.title
        if datatype == 'test_data':
            score_set = self.score_dict_test[score_name]
        else:
            score_set = self.score_dict_valid[score_name]
            # is_all=opt.get('is_all', False)
        if self.need_sorted and score_set[0].get('test_time', None) is not None: # do sorted
            score_set = sorted(score_set, key=lambda y: y['test_time'])
            
        plt.xlabel('#time step')
        plt.ylabel(score_name)
        for i in range(len(score_set)):
            score_ele = score_set[i]
            if(len(score_ele) == 0) or score_ele.get('score', None) is None:
                print filename + 'empty'
                continue
            test_name = score_ele['test_name']
            score = score_ele['score']
            exp_name_cur = score_ele['exp_name']
            #if is_all:
            #    label_name = exp_name_cur + '_' + test_name 
            #else:
            label_name = test_name
            if len(score_ele) == 1:
                plt.plot(1, score_ele[0], 'o', label=label_name)
                continue

            plt.plot(range(len(score)), score[:], label=label_name)
            print 'ploting ', exp_name_cur, test_name , '\t ', score_name, score[-1]
            plt.text(len(score), score[-1] , str("%.4f(%s)" % (score[-1], test_name.split('_')[0]))) 

        lgd = plt.legend(bbox_to_anchor=(0.5, 1.35),  loc='upper center', borderaxespad=0., fontsize='x-small', mode='expand', ncol=2)

        plt.grid()
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
        fig_name = savepath + '/' + exp_name + '_' + score_name + '.pdf'
        print 'save in ',fig_name 
        plt.savefig(fig_name, bbox_extra_artist=(lgd,), bbox_inches='tight')
        plt.clf()

    def GetIterList(self):
        result = {
                    "title": self.title,
                    "iter_list": self.iter_list
                 }
        return result
    def GetTrainLoss(self):
        return {
                    "title": self.title,
                    "x": self.iter_list,
                    "y": self.train_loss_list
                }
    def GetValidLoss(self):
        return {
                    "title": self.title,
                    "x": self.epoch_list,
                    "y": self.valid_loss_list
                }
    def GetBestSkillScore(self, score_name, mode='h'):
        """
          score_dict_test[score_name].append(
                            {
                                "test_name": test_name,    
                                "score": buf,
                                "exp_name": exp_name 
                            })
        """
        max_score = -1
        min_score = 1e5
        max_id = 0
        min_id = 0
        score_list = self.score_dict_test[score_name]
        for id, test in enumerate(score_list):
            if test['score'][-1] > max_score:
                max_score = test['score'][-1]
                max_id = id

            if test['score'][-1] < min_score:
                min_score = test['score'][-1]
                min_id = id
        print 'for ', self.title, ' score ', score_name, max_score, min_score
        if mode == 'h':
            return max_id, max_score
        else:
            return min_id, min_score
