from flask import Flask, render_template, session, request
import json
import numpy as np
from training.Arch.arch_model import ArchModel
import time
from u import *
import torch
import planning.src.http.dec_data
from planning.src.central_master import central_master
ServerAPP = Flask(__name__)
net = ArchModel().cuda()
net.load_state_dict(torch.load('arch.pth'))
Central_master=central_master()
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
task, task_context, task_now, task_num, task_num_now = [], [], [], [], []
reward_list = []

@ServerAPP.route('/', methods=['POST'])
def mainentry():
    data = request.json
    print(data)
    return data

@ServerAPP.route('/situation', methods=['POST'])
def getSituation_generate_info():
    situationDataDict = request.json
    print(situationDataDict)

    if situationDataDict['step'] == 1:
        print(1)

    print("finished generate! ")
    return