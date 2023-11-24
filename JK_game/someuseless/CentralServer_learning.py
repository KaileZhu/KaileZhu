#这个是联合版本，需要强化学习的内容，不建议跑

from flask import Flask, render_template, session, request
import json
import numpy as np
from training.Arch.arch_model import ArchModel
import time
from u import *
import torch
from planning.src.http import dec_data
from planning.src.central_master import central_master
import os
import matplotlib
#import matplotlib.pyplot as plt
matplotlib.use('Agg') #
from matplotlib import pyplot as plt #

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
task, task_context, task_now, task_num, task_num_now = get_task('outputdataforbeihang_0_0.txt')
ServerAPP = Flask(__name__)
#net = ArchModel().cuda()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = ArchModel().to(device)
#net.load_state_dict(torch.load('arch.pth'))
net.load_state_dict(torch.load('arch.pth',map_location='cpu'))

Central_master=central_master()
Central_master.get_initial_data()
red = ['Red.RedForce_RedCommander#0', 'Red.RedForce_RedCommander#0_RedCommanderMiddle#0',
                'Red.RedForce_RedCommander#0_RedAirArtillery#0', 'Red.RedForce_RedCommander#0_RedAirArtillery#1',
                   'Red.RedForce_RedCommander#0_RedAirArtillery#2', 'Red.RedForce_RedCommander#0_RedAirArtillery#3',
                   'Red.RedForce_RedCommander#0_RedRocketGun#0', 'Red.RedForce_RedCommander#0_RedRocketGun#1',
                'Red.RedForce_RedCommander#0_RedRocketGun#2', 'Red.RedForce_RedCommander#0_RedRocketGun#3',
                'Red.RedForce_RedCommander#0_RedRocketGun#4', 'Red.RedForce_RedCommander#0_RedSAUGV#0', 'Red.RedForce_RedCommander#0_RedSAUGV#1',
            'Red.RedForce_RedCommander#0_RedSAUGV#2', 'Red.RedForce_RedCommander#0_RedSAUGV#3',
            'Red.RedForce_RedCommander#0_RedRSUAVehicle#0', 'Red.RedForce_RedCommander#0_RedRSUAVehicle#1',
                  'Red.RedForce_RedCommander#0_RedRSUAVehicle#2', 'Red.RedForce_RedCommander#0_RedRSUAVehicle#0.RedRSUAV#0',
            'Red.RedForce_RedCommander#0_RedRSUAVehicle#1.RedRSUAV#0','Red.RedForce_RedCommander#0_RedRSUAVehicle#2.RedRSUAV#0',
            'Red.RedForce_RedCommander#0_RedFSUAVehicle#0', 'Red.RedForce_RedCommander#0_RedFSUAVehicle#1',
                  'Red.RedForce_RedCommander#0_RedFSUAVehicle#2', 'Red.RedForce_RedCommander#0_RedFSUAVehicle#0.RedFSUAV#0', 'Red.RedForce_RedCommander#0_RedFSUAVehicle#1.RedFSUAV#0',
            'Red.RedForce_RedCommander#0_RedFSUAVehicle#2.RedFSUAV#0', 'Red.RedForce_RedCommander#0_RedArCMUGV#0',
            'Red.RedForce_RedCommander#0_RedArCMUGV#1', 'Red.RedForce_RedCommander#0_RedInCMUGV#0', 'Red.RedForce_RedCommander#0_RedAJUGV#0',
              'Red.RedForce_RedCommander#0_RedArCMUGV#0.RedArCMissile#0', 'Red.RedForce_RedCommander#0_RedArCMUGV#0.RedArCMissile#1',
                'Red.RedForce_RedCommander#0_RedArCMUGV#0.RedArCMissile#2', 'Red.RedForce_RedCommander#0_RedArCMUGV#0.RedArCMissile#3',
                'Red.RedForce_RedCommander#0_RedArCMUGV#0.RedArCMissile#4', 'Red.RedForce_RedCommander#0_RedArCMUGV#0.RedArCMissile#5',
                'Red.RedForce_RedCommander#0_RedArCMUGV#1.RedArCMissile#0', 'Red.RedForce_RedCommander#0_RedArCMUGV#1.RedArCMissile#1',
                'Red.RedForce_RedCommander#0_RedArCMUGV#1.RedArCMissile#2', 'Red.RedForce_RedCommander#0_RedArCMUGV#1.RedArCMissile#3',
                'Red.RedForce_RedCommander#0_RedArCMUGV#1.RedArCMissile#4', 'Red.RedForce_RedCommander#0_RedArCMUGV#1.RedArCMissile#5',
                'Red.RedForce_RedCommander#0_RedInCMUGV#0.RedInCMissile#0', 'Red.RedForce_RedCommander#0_RedInCMUGV#0.RedInCMissile#1',
                'Red.RedForce_RedCommander#0_RedInCMUGV#0.RedInCMissile#2', 'Red.RedForce_RedCommander#0_RedInCMUGV#0.RedInCMissile#3',
                'Red.RedForce_RedCommander#0_RedInCMUGV#0.RedInCMissile#4', 'Red.RedForce_RedCommander#0_RedInCMUGV#0.RedInCMissile#5',
                'Red.RedForce_RedCommander#0_RedInCMUGV#0.RedInCMissile#6', 'Red.RedForce_RedCommander#0_RedInCMUGV#0.RedInCMissile#7',
                'Red.RedForce_RedCommander#0_RedInCMUGV#0.RedInCMissile#8', 'Red.RedForce_RedCommander#0_RedInCMUGV#0.RedInCMissile#9']

blue= ['Blue.BlueForce_BlueCommander#0', 'Blue.BlueForce_BlueCommander#0_BlueUAVehicle#0',
                 'Blue.BlueForce_BlueCommander#0_BlueCommandMiddle#0', 'Blue.BlueForce_BlueCommander#0_BlueCommandMiddle#1',
                     'Blue.BlueForce_BlueCommander#0_BlueCommandMiddle#2', 'Blue.BlueForce_BlueCommander#0_BlueCommandMiddle#3',
                     'Blue.BlueForce_BlueCommander#0_BlueInfantry#0', 'Blue.BlueForce_BlueCommander#0_BlueInfantry#1',
                'Blue.BlueForce_BlueCommander#0_BlueInfantry#2', 'Blue.BlueForce_BlueCommander#0_BlueInfantry#3',
                'Blue.BlueForce_BlueCommander#0_BlueArtillery#0', 'Blue.BlueForce_BlueCommander#0_BlueArtillery#1',
                 'Blue.BlueForce_BlueCommander#0_BlueArtillery#2',
                 'Blue.BlueForce_BlueCommander#0_BlueArchibald#0', 'Blue.BlueForce_BlueCommander#0_BlueArchibald#1', 'Blue.BlueForce_BlueCommander#0_BlueADLanucher#0', 'Blue.BlueForce_BlueCommander#0_BlueADLanucher#1',
                  'Blue.BlueForce_BlueCommander#0_BlueADLanucher#2', 'Blue.BlueForce_BlueCommander#0_BlueCommNode#0',
                'Blue.BlueForce_BlueCommander#0_BlueRadar#0', 'Blue.BlueForce_BlueCommander#0_BlueRadar#1'
        ,'Blue.BlueForce_BlueCommander#0_BlueUAV#0']

final_location = [[], [], [], [], [], []]
final_flag = [0, 0, 0, 0, 0, 0]
detect_index = [18, 19, 20, 24, 25, 26]
al_task = []
reward_list = []

global is_finish
is_finish = 'False'

@ServerAPP.route('/', methods=['POST'])
def mainentry():
    data = request.json
    print(data)
    return data

@ServerAPP.route('/initialsituation', methods=['POST'])
def initialsituation():
    situationDataDict = request.json
    print(situationDataDict)
    response = 'Copy that!'
    return response

@ServerAPP.route('/decision', methods=['POST'])
def getSituation():  # situation，次级接口，接受command数据
    command= request.json  # 读取command数据
    #command = json.dumps(situationDataDict)  # 将数据修改，这里实际上是最后一步，
    #这里读取输入的数据， 然后进行一个对command 的分析
    #function1
    print(command)
    if 'ZF_choose_function' in command.keys():
        #输出智能体信息
        Central_master.Data_manager.agent_data
        #print(Data_manager.agent_data)
        return json.dumps(Central_master.Data_manager.agent_data)
    #function2
    if 'ZF_combination' in command.keys():
        #这部分战法属于预先完成的内容，所以可以直接查表
        return json.dumps(dec_data.zanfa_data[command['ZF_combination']])
    #function3
    if 'ZF_detail' in command.keys():
        #这里需要输入战法的内容，进行对于的计算，最终返回子任务序列，
        Central_master.get_input_data(command['ZF_detail'])
        subtask_list=Central_master.out_put_subtask_data()
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(Central_master.Poset_product.final_poset)
        print(Central_master.Poset_product.final_task_data_list)
        #Centrol_master.get_input_data(command['ZF_detail'])
        #return json.dumps(Centrol_master.poset['action_map'])
        #这里返回的是子任务，以及子任务对于的4个维度的参数

        return json.dumps(subtask_list)
    if 'ZF_calculate' in command.keys():
        time_limit=15
        Central_master.determine_step(time_limit)
        f = open('outputdataforbeihang_0.txt', 'w')
        f.write(str(Central_master.current_solution))
        f.close()

        return json.dumps(Central_master.second_level_calculate_time)
    #function4-5


    if 'Gantt_graph' in command.keys():
        if 'first' == command['Gantt_graph']:
            gantt_data=Central_master.get_off_line_gantt_graph()

            return json.dumps(gantt_data)
        if 'final' == command['Gantt_graph']:
            #这里需要获取来着北航的message_data
            message_data = []
            with open('subtask.txt', 'r') as f:
                content = f.readlines()
                for idx, cur in enumerate(content):
                    if idx == len(content) - 1:
                        message_data.append(eval(cur))
                    else:
                        message_data.append(eval(cur)[0])
            gantt_data_dic=Central_master.online_gantt_graph(message_data)
            return json.dumps(gantt_data_dic)


@ServerAPP.route('/situation', methods=['POST'])
def getSituation_rl():
    situationDataDict = request.json
    print(situationDataDict)
    global task, task_context, task_now, task_num, task_num_now, task_num_list0

    global is_finish
    if 'finish' in situationDataDict.keys():
        is_finish = situationDataDict['finish']

    # if situationDataDict['step'] == 1:
    #     # 输出first_data
    #     with open("first_data.json", "w") as f:
    #         f.write(json.dumps(situationDataDict))
    #
    #
    # if situationDataDict['step'] == 30 or situationDataDict['step'] == 70:
    #     # 得到已完成任务列表
    #     remain_task_list = []
    #     for i in range(53):
    #         for j in range(len(task[i])):
    #             if j < task_num_now[i]:
    #                 continue
    #             if task[i][j] not in remain_task_list and task[i][j] !=100 and task[i][j] != -1:
    #                 remain_task_list.append(task[i][j])
    #     finish_task_list = []
    #     for task_n in task_num_list0:
    #         if task_n in remain_task_list:
    #             continue
    #         finish_task_list.append(task_n)
    #
    #     # 重规划
    #     print(finish_task_list)
    #     new_task_list = Central_master.online_replanning(situationDataDict, finish_task_list, is_beihang=True)
    #     task, task_context, task_now, task_num, task_num_now = get_txt_from_replan(new_task_list)
    #     print('--重规划已完成--')

    # 任务预处理
    index = 0
    life = [1 for _ in range(53)]
    for id, entities in situationDataDict['redSituation'].items():
        for entity in entities:
            if entity['life'] <= 0:
                life[index] = 0
            index += 1

    for i in range(53):
        if life[i] == 0:
            task_now[i] = -1

    for i in range(53):
        if task_now[i] == 100 or task_now[i] == -1:
            pass
        else:
            if is_task_done(situationDataDict, task_context[i][task_num_now[i]], red[i]) or task_now[i] in al_task:
                task_num_now[i] += 1
                al_task.append(task_now[i])
                if task_num_now[i] == task_num[i]:
                    task_now[i] = 100
                else:
                    task_now[i] = task[i][task_num_now[i]]

    task_now_constrain = get_task_constrain(task_now)
    task_set = list(set(task_now_constrain))
    if 100 in task_set:
        task_set.remove(100)
    if -1 in task_set:
        task_set.remove(-1)

    logger(situationDataDict, task_set, reward_list)

    command = decision_process(situationDataDict, task_now_constrain)
    command = blue_action(situationDataDict, command)

    for id, entities in situationDataDict['redSituation'].items():
        for entity in entities:
            if entity['name'] in red[2:6]:
                action = {}
                action['agentname'] = entity['name']
                action['targetname'] = ['Blue.BlueForce_BlueCommander#0_BlueUAV#0']
                command['fireactions'].append(action)

    # action = {'agentname': 'Red.RedForce_RedCommander#0_RedSAUGV#1', 'longitude': 79.71085258130125,
    #           'latitude': 30.36712195205492, 'altitude': 0}
    # command = {
    #     'moveactions': [action],
    #     'fireactions': [],
    #     'scoutactions': []
    # }
    #
    # print(command)

    return command


'''替换getSituation2'''
@ServerAPP.route('/gaming', methods=['POST'])
def getSituation2():  # situation，次级接口，接受command数据
    global is_finish
    command = request.json  # 读取command数据
    if 'all_info' == command:
        red_war_info_dict_list = []
        with open('red_damage.txt', 'r') as f:
            content = f.readlines()
            for idx, cur in enumerate(content):
                red_war_info_dict_list.append(eval(cur))

        blue_war_info_dict_list = []
        with open('blue_damage.txt', 'r') as f:
            content = f.readlines()
            for idx, cur in enumerate(content):
                blue_war_info_dict_list.append(eval(cur))

        red_serch_info_dict_list = []
        with open('detect.txt', 'r') as f:
            content = f.readlines()
            for idx, cur in enumerate(content):
                red_serch_info_dict_list.append(eval(cur))

        red_blue_ability_dict_list = []
        with open('strength.txt', 'r') as f:
            content = f.readlines()
            for idx, cur in enumerate(content):
                red_blue_ability_dict_list.append(eval(cur))

        red_blue_value_dict_list = []
        with open('value.txt', 'r') as f:
            content = f.readlines()
            for idx, cur in enumerate(content):
                red_blue_value_dict_list.append(eval(cur))

        red_executing_subtask_dict_list = []
        with open('subtask.txt', 'r') as f:
            content = f.readlines()
            for idx, cur in enumerate(content):
                red_executing_subtask_dict_list.append(eval(cur))

        red_reward_dict_list = []
        with open('reward.txt', 'r') as f:
            content = f.readlines()
            for idx, cur in enumerate(content):
                red_reward_dict_list.append(eval(cur))

        return json.dumps(red_war_info_dict_list), json.dumps(blue_war_info_dict_list), \
               json.dumps(red_serch_info_dict_list), json.dumps(red_blue_ability_dict_list), \
               json.dumps(red_blue_value_dict_list), json.dumps(red_executing_subtask_dict_list), \
               json.dumps(red_reward_dict_list), json.dumps(is_finish)

def decision_process(situation, task_now):
    command = net(situation, task_now, task_context, task_num_now)
    return command


if __name__ == '__main__':
    # task, task_context, task_now, task_num, task_num_now = get_task('outputdataforbeihang_0_0.txt')
    ServerAPP.run(host='localhost', port=9999, debug=True)