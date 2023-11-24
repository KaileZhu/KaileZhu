#learning banbeng
from flask import Flask, request
import json
import numpy as np

import time
from u import *
from u_2 import *
from u_3 import *
from action_mapping import *
# import torch
from planning.src.http import dec_data
from planning.src.central_master import central_master
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

ServerAPP = Flask(__name__)
# from training.Arch.arch_model import ArchModel
# net = ArchModel().cuda()
# net.load_state_dict(torch.load('arch.pth'))
Central_master=central_master()

logger = Logger()
logger2 = Logger2()
logger3 = Logger3()


global task, task_context, task_now, task_num, task_num_now, task_num_list0
global is_replan
is_replan = False

# task, task_context, task_now, task_num, task_num_now, task_num_list0 = get_task('outputdataforbeihang_0_0.txt')
# from task_list_ZYBJ import *
# task, task_context, task_now, task_num, task_num_now, = get_txt_from_replan(task_list00)  # 演示视频所用任务
# task, task_context, task_now, task_num, task_num_now = get_txt_from_replan(task_list00)  # 全面进攻
# task, task_context, task_now, task_num, task_num_now, = get_txt_from_replan(task_list_fangshoufanji)  # 防守反击
# task, task_context, task_now, task_num, task_num_now, = get_txt_from_replan(task_list_weidiandayuan)  # 围点打援
# task, task_context, task_now, task_num, task_num_now, = get_txt_from_replan(task_list_bubuweiying)  # 步步为营
# task, task_context, task_now, task_num, task_num_now, = get_txt_from_replan(task_list_zhanchangsaomiao)  # 占场扫描
# task, task_context, task_now, task_num, task_num_now, = get_txt_from_replan(task_list_zhongdianfangyu)  # 重点防御

# entities updated
red1 = [
     'Red.RedForce_RedCommander#0',
     'Red.RedForce_RedCommander#0_RedCommanderMiddle#0',
     'Red.RedForce_RedCommander#0_RedAirArtillery#0',
     'Red.RedForce_RedCommander#0_RedAirArtillery#1',
     'Red.RedForce_RedCommander#0_RedAirArtillery#2',
     'Red.RedForce_RedCommander#0_RedAirArtillery#3',
     'Red.RedForce_RedCommander#0_RedRocketGun#0',
     'Red.RedForce_RedCommander#0_RedRocketGun#1',
     'Red.RedForce_RedCommander#0_RedRocketGun#2',
     'Red.RedForce_RedCommander#0_RedRocketGun#3',
     'Red.RedForce_RedCommander#0_RedRocketGun#4',
     'Red.RedForce_RedCommander#0_RedSAUGV#0',
     'Red.RedForce_RedCommander#0_RedSAUGV#1',
     'Red.RedForce_RedCommander#0_RedSAUGV#2',
     'Red.RedForce_RedCommander#0_RedSAUGV#3',
     'Red.RedForce_RedCommander#0_RedSAUGV#4',
     'Red.RedForce_RedCommander#0_RedSAUGV#5',
     'Red.RedForce_RedCommander#0_RedSAUGV#6',
     'Red.RedForce_RedCommander#0_RedSAUGV#7',
     'Red.RedForce_RedCommander#0_RedRSUAVehicle#0',
     'Red.RedForce_RedCommander#0_RedRSUAVehicle#1',
     'Red.RedForce_RedCommander#0_RedRSUAVehicle#2',
     'Red.RedForce_RedCommander#0_RedRSUAVehicle#0.RedRSUAV#0',
     'Red.RedForce_RedCommander#0_RedRSUAVehicle#1.RedRSUAV#0',
     'Red.RedForce_RedCommander#0_RedRSUAVehicle#2.RedRSUAV#0',
     'Red.RedForce_RedCommander#0_RedFSUAVehicle#0',
     'Red.RedForce_RedCommander#0_RedFSUAVehicle#1',
     'Red.RedForce_RedCommander#0_RedFSUAVehicle#2',
     'Red.RedForce_RedCommander#0_RedFSUAVehicle#3',
     'Red.RedForce_RedCommander#0_RedFSUAVehicle#4',
     'Red.RedForce_RedCommander#0_RedFSUAVehicle#5',
     'Red.RedForce_RedCommander#0_RedFSUAVehicle#6',
     'Red.RedForce_RedCommander#0_RedFSUAVehicle#7',
     'Red.RedForce_RedCommander#0_RedFSUAVehicle#8',
     'Red.RedForce_RedCommander#0_RedFSUAVehicle#9',
     'Red.RedForce_RedCommander#0_RedFSUAVehicle#0.RedFSUAV#0',
     'Red.RedForce_RedCommander#0_RedFSUAVehicle#1.RedFSUAV#0',
     'Red.RedForce_RedCommander#0_RedFSUAVehicle#2.RedFSUAV#0',
     'Red.RedForce_RedCommander#0_RedFSUAVehicle#3.RedFSUAV#0',
     'Red.RedForce_RedCommander#0_RedFSUAVehicle#4.RedFSUAV#0',
     'Red.RedForce_RedCommander#0_RedFSUAVehicle#5.RedFSUAV#0',
     'Red.RedForce_RedCommander#0_RedFSUAVehicle#6.RedFSUAV#0',
     'Red.RedForce_RedCommander#0_RedFSUAVehicle#7.RedFSUAV#0',
     'Red.RedForce_RedCommander#0_RedFSUAVehicle#8.RedFSUAV#0',
     'Red.RedForce_RedCommander#0_RedFSUAVehicle#9.RedFSUAV#0',
     'Red.RedForce_RedCommander#0_RedArCMUGV#0',
     'Red.RedForce_RedCommander#0_RedArCMUGV#1',
     'Red.RedForce_RedCommander#0_RedInCMUGV#0',
     'Red.RedForce_RedCommander#0_RedAJUGV#0',
     'Red.RedForce_RedCommander#0_RedArCMUGV#0.RedArCMissile#0',
     'Red.RedForce_RedCommander#0_RedArCMUGV#0.RedArCMissile#1',
     'Red.RedForce_RedCommander#0_RedArCMUGV#0.RedArCMissile#2',
     'Red.RedForce_RedCommander#0_RedArCMUGV#0.RedArCMissile#3',
     'Red.RedForce_RedCommander#0_RedArCMUGV#0.RedArCMissile#4',
     'Red.RedForce_RedCommander#0_RedArCMUGV#0.RedArCMissile#5',
     'Red.RedForce_RedCommander#0_RedArCMUGV#1.RedArCMissile#0',
     'Red.RedForce_RedCommander#0_RedArCMUGV#1.RedArCMissile#1',
     'Red.RedForce_RedCommander#0_RedArCMUGV#1.RedArCMissile#2',
     'Red.RedForce_RedCommander#0_RedArCMUGV#1.RedArCMissile#3',
     'Red.RedForce_RedCommander#0_RedArCMUGV#1.RedArCMissile#4',
     'Red.RedForce_RedCommander#0_RedArCMUGV#1.RedArCMissile#5',
     'Red.RedForce_RedCommander#0_RedInCMUGV#0.RedInCMissile#0',
     'Red.RedForce_RedCommander#0_RedInCMUGV#0.RedInCMissile#1',
     'Red.RedForce_RedCommander#0_RedInCMUGV#0.RedInCMissile#2',
     'Red.RedForce_RedCommander#0_RedInCMUGV#0.RedInCMissile#3',
     'Red.RedForce_RedCommander#0_RedInCMUGV#0.RedInCMissile#4',
     'Red.RedForce_RedCommander#0_RedInCMUGV#0.RedInCMissile#5',
     'Red.RedForce_RedCommander#0_RedInCMUGV#0.RedInCMissile#6',
     'Red.RedForce_RedCommander#0_RedInCMUGV#0.RedInCMissile#7',
     'Red.RedForce_RedCommander#0_RedInCMUGV#0.RedInCMissile#8',
     'Red.RedForce_RedCommander#0_RedInCMUGV#0.RedInCMissile#9',
     'Red.RedForce_RedCommander#0_RedsuicideUAV#0',
     'Red.RedForce_RedCommander#0_RedsuicideUAV#1',
     'Red.RedForce_RedCommander#0_RedsuicideUAV#2',
     'Red.RedForce_RedCommander#0_RedsuicideUAV#3',
     'Red.RedForce_RedCommander#0_RedsuicideUAV#4',
     'Red.RedForce_RedCommander#0_RedsuicideUAV#5',
     'Red.RedForce_RedCommander#0_RedsuicideUAV#6',
     'Red.RedForce_RedCommander#0_RedsuicideUAV#7',
     'Red.RedForce_RedCommander#0_RedsuicideUAV#8',
     'Red.RedForce_RedCommander#0_RedsuicideUAV#9',
]

blue1 = [
     'Blue.BlueForce_BlueCommander#0',
     'Blue.BlueForce_BlueCommander#0_BlueUAVehicle#0',
     'Blue.BlueForce_BlueCommander#0_BlueCommandMiddle#0',
     'Blue.BlueForce_BlueCommander#0_BlueCommandMiddle#1',
     'Blue.BlueForce_BlueCommander#0_BlueCommandMiddle#2',
     'Blue.BlueForce_BlueCommander#0_BlueCommandMiddle#3',
     'Blue.BlueForce_BlueCommander#0_BlueInfantry#0',
     'Blue.BlueForce_BlueCommander#0_BlueInfantry#1',
     'Blue.BlueForce_BlueCommander#0_BlueInfantry#2',
     'Blue.BlueForce_BlueCommander#0_BlueInfantry#3',
     'Blue.BlueForce_BlueCommander#0_BlueInfantry#4',
     'Blue.BlueForce_BlueCommander#0_BlueInfantry#5',
     'Blue.BlueForce_BlueCommander#0_BlueInfantry#6',
     'Blue.BlueForce_BlueCommander#0_BlueInfantry#7',
     'Blue.BlueForce_BlueCommander#0_BlueInfantry#8',
     'Blue.BlueForce_BlueCommander#0_BlueInfantry#9',
     'Blue.BlueForce_BlueCommander#0_BlueInfantry#10',
     'Blue.BlueForce_BlueCommander#0_BlueInfantry#11',
     'Blue.BlueForce_BlueCommander#0_BlueArtillery#0',
     'Blue.BlueForce_BlueCommander#0_BlueArtillery#1',
     'Blue.BlueForce_BlueCommander#0_BlueArtillery#2',
     'Blue.BlueForce_BlueCommander#0_BlueArtillery#3',
     'Blue.BlueForce_BlueCommander#0_BlueArtillery#4',
     'Blue.BlueForce_BlueCommander#0_BlueArtillery#5',
     'Blue.BlueForce_BlueCommander#0_BlueArtillery#6',
     'Blue.BlueForce_BlueCommander#0_BlueArtillery#7',
     'Blue.BlueForce_BlueCommander#0_BlueArchibald#0',
     'Blue.BlueForce_BlueCommander#0_BlueArchibald#1',
     'Blue.BlueForce_BlueCommander#0_BlueADLanucher#0',
     'Blue.BlueForce_BlueCommander#0_BlueADLanucher#1',
     'Blue.BlueForce_BlueCommander#0_BlueADLanucher#2',
     'Blue.BlueForce_BlueCommander#0_BlueCommNode#0',
     'Blue.BlueForce_BlueCommander#0_BlueRadar#0',
     'Blue.BlueForce_BlueCommander#0_BlueRadar#1',
     'Blue.BlueForce_BlueCommander#0_BlueUAV#0',
     'Blue.BlueForce_BlueCommander#0_BlueUAV#1',
     'Blue.BlueForce_BlueCommander#0_BlueUAV#2',
     'Blue.BlueForce_BlueCommander#0_BlueUAV#3',
     'Blue.BlueForce_BlueCommander#0_BlueUAV#4',
     'Blue.BlueForce_BlueCommander#0_BlueUAV#5'
]

al_task = []
reward_list = []

# 红方作战智能体数量，与任务规划结果行数匹配
red_agent_num1 = 81  # 此处和两个u都要改 81个

red = red1
red_agent_num = red_agent_num1

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
    print('这里是输入的参数',command)
    if 'calculate_content' in command.keys():
        #{'场景选择': '进攻场景', '算法选择': '两级合同网', '算法路径':
        # 'planning\\src\\ltl_mas\\tools\\Ct_N.py', '偏好选择': '执行速度优先', '战法选择': '占场扫描', '条例选择': ['禁止攻击']}
        input=command['calculate_content']
        print(input)
        background=input['场景选择']
        #background='想定场景1'
        calculate_type=input['算法选择']  #calculate_type='两级合同网'
        path_calculate=input['算法路径']  #path_calculate=''
        prefer_type=input['偏好选择']   #prefer_type='最短时间'
        war_type=input['战法选择']
        stick=input['条例选择']     #stick='无限制'
        begin_time=time.time()
        #初始化背景参数
        output=Central_master.pre_design_the_subtask_structure(background,calculate_type,path_calculate,prefer_type,war_type,stick)
        #输入算法，偏好，战法，条例
        Central_master.first_level_calculate_time=time.time()-begin_time
        # dec_data.zanfa_data[command['ZF_combination']]
        print('output', output)
        print('--------------------------')
        print('完成中枢决策层计算！')
        print('中枢决策层计算时间为：', Central_master.first_level_calculate_time)
        file = open('logger/common/time.txt', mode='a')
        file.write('中枢决策层计算时间为：' + str(Central_master.first_level_calculate_time) + ',\n')
        file.close()
        print('output',output)
        old_output=dec_data.zanfa_data[input]
        old_output['zanfa_example']=output
        return json.dumps(output)

    if 'calculate_subtask' in command.keys():
        input = command['calculate_subtask']
        print('first_error',input)
        input['偏好选择']='最短时间'
        alg_param=int(input['算法参数']) #  alg_param=15
        input,prefer=Central_master.change_goal_task_pair(input)
        output_data=Central_master.calculate_subtask_value(input,Central_master.task.stick,Central_master.task.war_type)
        f = open('subtask_data1.txt', 'w')#路径改到gui里面吧
        f.write(str(output_data))
        f.close()
        print('final',output_data)
        #理论上这部分是直接保存在 subtask_data.py文件里？
        #为啥这部分会自动刷新？？
        return json.dumps(output_data)

    if 'calculate_execute' in command.keys():
        input = command['calculate_execute']
        Central_master.alg_param=int(input['算法参数'])
        print(input)
        Central_master.determine_step()
        solution = Central_master.current_solution
        f = open('全面进攻.txt', 'w')
        f.write(str(Central_master.current_solution))
        global task, task_context, task_now, task_num, task_num_now, task_num_list0
        if Central_master.background == '想定场景1':
            task, task_context, task_now, task_num, task_num_now, task_num_list0 = get_txt_from_replan(solution, is_first=True)
        # print(task, task_context, task_now, task_num, task_num_now, task_num_list0)
        f.close()
        #现在这里的问题是，运行之后 应该到不了1了
        #但是决策的过程已经完成了
        return json.dumps(1)

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
        #Centrol_master.get_input_data(command['ZF_detail'])
        #return json.dumps(Centrol_master.poset['action_map'])
        #这里返回的是子任务，以及子任务对于的4个维度的参数

        return json.dumps(subtask_list)
    if 'ZF_calculate' in command.keys():
        time_limit=15
        Central_master.determine_step(time_limit)
        f = open('全面进攻.txt', 'w')
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

def save_dict(sitution):
    step = sitution["step"]
    file_name = 'logger/situation/'+str(step)+'.json'
    with open(file_name, 'w') as f:
        f.write(json.dumps(sitution))

@ServerAPP.route('/situation', methods=['POST'])
def getSituation_rl():
    situationDataDict = request.json

    print(situationDataDict)
    save_dict(situationDataDict)

    global task, task_context, task_now, task_num, task_num_now, task_num_list0
    global is_replan

    # print(Central_master.task.stick)
    stick = Central_master.task.stick

    # print(situationDataDict['step'])
    # print(task)
    # print(task_context)
    # print(task_now)
    # print(task_num)
    # print(task_num_now)
    # print(task_num_list0)
    # print(Central_master.background)
    if Central_master.background == '想定场景1':
        # print('start')
        time1 = time.time()
        red_staus = get_red_staus(situationDataDict)
        rate = red_staus['red_damage_rate']

        # if situationDataDict['step'] == 30 or situationDataDict['step'] == 70:
        if (rate > 0.2 and is_replan == False) or situationDataDict['step'] == 70:
            is_replan = True
            # 获取已完成任务列表
            print('当前红方毁伤率', rate)
            print('检测到战场环境发生变化，子网估计智能体，任务变更情况:')
            remain_task_list = []
            for i in range(81):
                for j in range(len(task[i])):
                    if j < task_num_now[i]:
                        continue
                    if task[i][j] not in remain_task_list and task[i][j] != 100 and task[i][j] != -1:
                        remain_task_list.append(task[i][j])
            finish_task_list = []
            for task_n in task_num_list0:
                if task_n in remain_task_list:
                    continue
                finish_task_list.append(task_n)
            print('已完成任务',finish_task_list,'对以下任务进行重新规划：',remain_task_list)
            print('传递给上层合同网，开始重新规划')
            # 重规划
            new_task_list = Central_master.online_replanning(situationDataDict, finish_task_list, is_beihang=True)
            task, task_context, task_now, task_num, task_num_now, _ = get_txt_from_replan(new_task_list)

            print('--重规划已完成--')

        # 任务预处理
        # 获取智能体是否死亡 get life for entities
        index = 0
        life = [1 for _ in range(red_agent_num)]
        for id, entities in situationDataDict['redSituation'].items():
            for entity in entities:
                # print(entity['life'])
                if entity['life'] <= 0:
                    life[index] = 0
                index += 1

        # 如智能体死亡，将任务列表置为-1
        for i in range(red_agent_num):
            if life[i] == 0:
                task_now[i] = -1

        # 判断所有智能体任务完成情况
        for i in range(red_agent_num):
            if task_now[i] == 100 or task_now[i] == -1:
                pass
            else:
                if is_task_done(situationDataDict, task_context[i][task_num_now[i]], red[i]) or task_now[i] in al_task:  # todo: change for now location mapping
                    task_num_now[i] += 1
                    al_task.append(task_now[i])  # 同时完成的几个任务无需重复判断
                    # print(task_num_now, task_num)
                    if task_num_now[i] == task_num[i]:
                        task_now[i] = 100
                    else:
                        task_now[i] = task[i][task_num_now[i]]

        logger.update(situationDataDict, reward_list)
        command = red_action_2(situationDataDict, task_now, task_context, task_num_now, stick)
        command = blue_action(situationDataDict, command)

        command_printer(command)  # 输出动作和映射关系

        time2 = time.time()
        print("单步博弈结束，总共用时为", time2-time1)
        return command
    elif Central_master.background == '想定场景2':
        print(situationDataDict)
        command = {"moveactions": [],
                   "scoutactions": [],
                   "fireactions": []}
        command = red_actions_2(situationDataDict, command)
        command = blue_actions_2(situationDataDict, command)
        # print(command)
        logger2.update(situationDataDict, reward_list)
        return command
    elif Central_master.background == '想定场景3':
        situationDataDict = request.json  # 读取从端口发来的信息。
        print(situationDataDict)
        command = {"moveactions": [],
                   "scoutactions": [],
                   "fireactions": []}
        command = red_actions_3(situationDataDict, command)
        command = blue_actions_3(situationDataDict, command)
        print(command)
        logger3.update(situationDataDict, reward_list)
        return command  # 返回请求数据


@ServerAPP.route('/gaming', methods=['POST'])
def getSituation2():  # situation，次级接口，接受command数据
    command = request.json  # 读取command数据
    #command = json.dumps(situationDataDict)  # 将数据修改，
    #这里读取输入的数据， 然后进行一个调用，然后返回
    # function1
    if 'red_war_infor' ==command:
        dict_list = []
        with open('red_damage.txt', 'r') as f:
            content = f.readlines()
            for idx, cur in enumerate(content):
                if idx == len(content) - 1:
                    dict_list.append(eval(cur))
                else:
                    dict_list.append(eval(cur)[0])
        return json.dumps(dict_list)

    if 'blue_war_infor' == command:
        dict_list = []
        with open('blue_damage.txt', 'r') as f:
            content = f.readlines()
            for idx, cur in enumerate(content):
                if idx == len(content) - 1:
                    dict_list.append(eval(cur))
                else:
                    dict_list.append(eval(cur)[0])
        return json.dumps(dict_list)

    if 'red_search_infor' == command:
        dict_list = []
        with open('detect', 'r') as f:
            content = f.readlines()
            for idx, cur in enumerate(content):
                if idx == len(content) - 1:
                    dict_list.append(eval(cur))
                else:
                    dict_list.append(eval(cur)[0])
        return json.dumps(dict_list)

    if 'red_blue_ability' == command:
        dict_list = []
        with open('strength.txt', 'r') as f:
            content = f.readlines()
            for idx, cur in enumerate(content):
                if idx == len(content) - 1:
                    dict_list.append(eval(cur))
                else:
                    dict_list.append(eval(cur)[0])
        return json.dumps(dict_list)

    if 'red_blue_value'== command:
        dict_list = []
        with open('value.txt', 'r') as f:
            content = f.readlines()
            for idx, cur in enumerate(content):
                if idx == len(content) - 1:
                    dict_list.append(eval(cur))
                else:
                    dict_list.append(eval(cur)[0])
        return json.dumps(dict_list)

    if 'red_executing_subtask'== command:
        dict_list = []
        with open('subtask.txt', 'r') as f:
            content = f.readlines()
            for idx, cur in enumerate(content):
                if idx == len(content) - 1:
                    dict_list.append(eval(cur))
                else:
                    dict_list.append(eval(cur)[0])
        return json.dumps(dict_list)

    if 'red_reward' == command:
        dict_list = []
        with open('reward.txt', 'r') as f:
            content = f.readlines()
            for idx, cur in enumerate(content):
                if idx == len(content) - 1:
                    dict_list.append(eval(cur))
                else:
                    dict_list.append(eval(cur)[0])
        return json.dumps(dict_list)


if __name__ == '__main__':
    ServerAPP.run(host='127.0.0.1', port=9999, debug=False)