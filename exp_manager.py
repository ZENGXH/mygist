from __future__ import with_statement
import sys
import subprocess
import json
from helper import *
import os.path
from exp_container import ExpContainer 


class ExpManeger(object):
    def __init__(self, config):
        config_file = json.loads(open(config).read())
        self.address_book = config_file['address_book']
        self.display = config_file.get('display', 0)
        self.savepath = config_file.get('savepath', '.')
        self.exp_name = config_file.get('exp_name', 'plot')
        self.filename = config_file.get('setID', 'default_set')
        self.need_update = config_file.get('need_update', False)
        self.is_draw_skill = config_file.get('is_draw_skill', True)
        self.average_interval = config_file.get('average_interval', 10)
        self.expdict = {}
        self.experiment_set = config_file['experiments']
        self.skill_score = ['CSI', 'FAR', 'correlation', 'POD', 'RMSE', 'Rain RMSE']  # 'RMSE', 'rain RMSE'] 
        self.all_title = []
        self.load_score_from_log = config_file.get('load_score_from_log', True)
        self.all_score_dict_test = {}
        self.all_score_dict_valid = {}
        self.all_iter_list = {} 
        self.all_epoch_list = {}
        self.all_train_loss_list = {}
        self.all_valid_loss_list = {}
        
        self.exp_container_list = []
        if not os.path.isdir(self.savepath):
            os.mkdir(self.savepath)
        self.LoadExperimentSet()

    def LoadExperimentSet(self):
        address_book = self.address_book
        expSet = self.experiment_set
        skill_score = self.skill_score
        need_update = self.need_update
        average_interval = self.average_interval

        for i in range(len(expSet)):
            exp_info = expSet[i]
            exp_info['id'] = i
            title = exp_info['title']
            print '[LoadExperimentSet] get title ', title
            server_root = address_book[exp_info['address_id']]['server'] + ':' + address_book[exp_info['address_id']]['root']
            # print 'server_root', server_root
            exp_info['server_path'] = server_root + '/' + exp_info['title']
            exp_info['average_interval'] = self.average_interval 
            exp_info['need_update'] = self.need_update
            exp_info['skill_score'] = self.skill_score
            exp_container = ExpContainer(exp_info)
            exp_container.LoadExperiment()
            self.exp_container_list.append(exp_container)

            # print 'exp_dict, ', exp_dict
            #self.all_iter_list[title] = iter_list
            #self.all_epoch_list[title] = epoch_list
            #self.all_train_loss_list[title] = train_loss_list
            #self.all_valid_loss_list[title] = valid_loss_list
            #self.all_epoch_list[title] = range(len(valid_loss_list))
            #self.all_title.append(title)
            #self.all_score_dict_test[title] = score_dict_cat[0]
            #self.all_score_dict_valid[title] = score_dict_cat[1]
                

    def DrawLossAll(self):
        result_test = []
        result_valid = []    
        for exp in self.exp_container_list:
            result_test.append(exp.GetTrainLoss())
            result_valid.append(exp.GetValidLoss())
            print '[DrawLossAll] DrawLoss', self.filename, 'test'
        self.Draw(result_test, filename='test_' + self.filename)
        self.Draw(result_valid, filename = 'valid' + self.filename)
    def DrawSkillScoreAll(self):
        for exp in self.exp_container_list:
            exp.DrawSkillScore(self.savepath, 'valid_data')
            exp.DrawSkillScore(self.savepath, 'test_data')

    def Draw(self, result_set, filename, plot_title='loss', labelx='x', labely='y'): 
        xmax = 0
        ymax = 0
        ymin = 1000
        for result in result_set:
            title = result['title']
            x_list = result["x"]
            y_list = result["y"] 
            starting_point = 0
            if(len(x_list) == 0) :
                print('empty list in %s'%title)
                continue
            xmax = max(max(x_list), xmax)
            ymin = min(min(y_list), ymin)
            ymax = max(max(y_list), ymax)
            print '[Draw] title %s, #x %d, #y %d range %.4f %.4f'%(title, len(x_list), len(y_list), ymin, ymax)
            plt.plot(x_list[:], y_list[:], label=title)
            plt.text(x_list[-1], y_list[-1], title, fontsize=5)
        plt.xlabel(labelx)
        plt.ylabel(labely)
        pylab.xlim([0, int(xmax)])
        pylab.ylim([ymin*(0.9), ymax*(1.1)])
        plt.grid()
        plt.title(plot_title)
        lgd = plt.legend(bbox_to_anchor=(0.5, 1.45),  
                loc='upper center', borderaxespad=0., 
                fontsize='x-small', mode='expand', ncol=2)

        plt.savefig('%s/%s.pdf'%(self.savepath, filename), 
                bbox_extra_artist=(lgd,), bbox_inches='tight')

        print 'save in %s/%s'%(self.savepath, filename)
        if self.display:
            print 'display'
            pylab.show()
        plt.clf()

    def GetBestSkillScore(self, score_name, mode='h'):
        best_epoch = 0
        best_score = 0
        result = {}
        for exp_container in self.exp_container_list:
            id, score, score_list = exp_container.GetBestSkillScore(score_name, mode)
            result['title'] = exp_container.title
            if (score > best_score and mode == 'h') or (score < best_score and mode == 'l'):
                best_epoch = id
                best_exp = exp.title
                best_score_list = score_list
        print 'best', score_name , best_exp, best_epoch, best_score
        return best_score_list

    def DrawBestSkillScore(self, score_name='all'):
        if score_name == 'all':
            for score_name_ in self.skill_score:
                return self.DrawBestSkillScore(score_name_)
        elif score_name in ['correlation', 'POD', 'CSI']:
            mode = 'h'
        elif score_name in ['RMSE', 'Rain RMSE', 'FAR']:
            mode = 'l'
        else:
            print "score_name %s not recognized"%score_name
            return
        print '[%s]'%score_name
        result_list = []
        for exp_container in self.exp_container_list:
            id, score, score_list = exp_container.GetBestSkillScore(score_name, mode)
            result_list.append( 
                        {
                            "title": exp_container.title + 'epoch%d'%(id),
                            "y": score_list,
                            "x": range(len(score_list))
                        })
            print '%s epoch %d %.3f'%(exp_container.title, id, score)
        self.Draw(result_list, 'best%s'%score_name,  'best%s'%score_name, '#step', score_name) 

    def OverAllEval(self):
        # self.skill_score = ['CSI', 'FAR', 'correlation', 'POD', 'RMSE', 'Rain RMSE']  # 'RMSE', 'rain RMSE'] 
        score_weight = {
                    'CSI': 0.1,
                    'FAR': -0.1,
                    'correlation': 0.3,
                    'POD': 0.3
                    
            }
        best_score = 0
        best_exp = ''
        for exp_container in self.exp_container_list:
            if exp_container.final_test_epoch == 0:
                continue
            epoch, evalscore = exp_container.OverAllEval(score_weight, 'test')
            # print exp_container.title, evalscore
            if best_score < evalscore:
                best_exp = exp_container.title
                best_score = evalscore
        print '<<test_data OverAllEval>> get %s %.4f \n'%(best_exp, best_score)
        
        best_score = 0
        for exp_container in self.exp_container_list:
            if exp_container.final_test_epoch == 0:
                continue
            epoch, evalscore = exp_container.OverAllEval(score_weight, 'valid')
            # print exp_container.title, evalscore
            if best_score < evalscore:
                best_exp = exp_container.title
                best_score = evalscore
        print '<<valid_data OverAllEval>> get %s %.4f \n'%(best_exp, best_score)
    
        best_score = -1
        for exp_container in self.exp_container_list:
            if exp_container.final_test_epoch == 0:
                continue
            epoch, evalscore = exp_container.OverAllEval(-1.0/2913838, 'test', 'Rain RMSE')
            if best_score < evalscore:
                best_exp = exp_container.title
                best_score = evalscore
        print '<<test_data RRMSE>> get %s %.4f \n'%(best_exp, best_score)
        
        best_score = -1
        for exp_container in self.exp_container_list: 
            if exp_container.final_test_epoch == 0:
                continue
            epoch, evalscore = exp_container.OverAllEval(-1.0/2913838, 'valid', 'Rain RMSE')
            if best_score < evalscore:
                best_exp = exp_container.title
                best_score = evalscore
        print '<<valid_data RRMSE>> get %s %.4f \n'%(best_exp, best_score)
