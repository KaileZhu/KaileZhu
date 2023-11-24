#！/usr/bin/env python3

'''
@Date   : 2023/11 -->  
@Authors: Junjie Wang, 
@Contact: pkuwjj1998@163.com
@Version: 1.0
@Descrip: Based on the given LTL task formula, assign the collaborative subtasks to
        multi-agent system efficiently, and guarantee asymptotic optimization.
'''

import os
import re
import time

import numpy as np
import networkx as nx
import matplotlib.patches as patches

from PIL import Image
from matplotlib import pyplot as plt

from MAS_TL_STAP import *
from MAS_TL_STAP.poset_builder import Buchi_poset_builder
from MAS_TL_STAP.B_A_B2 import Branch_And_Bound
from MAS_TL_STAP.poset_product import Poset_producter


if __name__ == '__main__':
    #=============================================
    # Initialization of the task and environment
    #=============================================
    path = os.path.abspath(os.path.dirname(
                            os.path.dirname(__file__)))
    path_file = f'{path}/scenes/scene_05.yaml'
    path_img = f'{path}/figures/zybj_dl.png'
    buchi, agents, subtasks, regions = None, list(), list(), dict()
    res = 50
    init_node = (50, 50)
    # get the initial setting from yaml file
    scene = InputData()
    TASK, AGENT, REGION = scene.read_from_yaml(path_file)
    #poset = Buchi_poset_builder(TASK)
    begin_time=time.time()
    Poset_product = Poset_producter( TASK)
    Poset_product.generate_poset()
    Poset_product.prodocter()
    poset = Poset_product.final_poset
    end_time=time.time()
    print('任务分析的执行时间为:',end_time-begin_time)
    # generate the grid map according to the figure
    motionmap = MotionFts(Image.open(path_img), res, init_node)
    motionmap.add_full_regions(REGION)
    motionmap.full_regions_dijkstra()
    agents = [Agent(at, scene.position, motionmap) for at in AGENT]

    # define a regular expression pattern to match the desired substrings
    pattern = re.compile(r'\b(\w+)_([a-zA-Z0-9]+)\b')
    # find all matches in the input string
    matches = poset['action_map']
    # extracted components
    subtasks.extend([(i, matches[i][1], matches[i][-1])
                    for i in range(len(matches))])

    #=========================
    # Calculation process
    #=========================
    begin_time=time.time()
    solver = Branch_And_Bound(poset, subtasks, scene)
    solver.Begin_branch_search2(3, up_bound_method='greedy',
                                low_bound_method='i+j', search_method='DFS')
    # print('solver_low_bound_list',solver.low_bound_list)
    solver.get_time_table_of_best_solution(solver.best_solution)
    end_time=time.time()
    print('分支定界法计算时间为：',end_time-begin_time)
    #100s最优是 240   10s 值是260
    Poset_product.gantt_plotter(poset,solver.best_solution,solver.task_time_table)
    for age in agents:
        for plan in solver.best_solution[age.id]:
            task = [{
                'task': plan[0][1],
                'task_id': plan[0][0],
                'act': plan[1],
                'dur': scene.task_type[plan[0][1]][0],
                'reg': plan[0][2],
                'pos': age.regions[plan[0][2]]
            }]
            age.add_new_tasks(task)

    #==========================
    # Show the results
    #==========================
    colorbar = plt.get_cmap('tab20')(range(20))
    t_max = max([len(a.path) for a in agents])
    mode = 'figure'  # 'figure' or 'video'

    # create figure and axes
    fig, ax = plt.subplots(figsize=(12,12))
    if mode == 'figure':
        # display the image
        ax.imshow(motionmap.img, origin='lower', alpha=0.2)
        ax.axis('on')
        ax.grid(ls='--', zorder=0)
        # for node_s, node_t in motionmap.edges():
        #     x = [node_s[0], node_t[0]]
        #     y = [node_s[1], node_t[1]]
        #     ax.plot(x, y, linewidth=0.5)
        # ax.scatter( [x for x,_ in motionmap.nodes()], 
        #             [y for _,y in motionmap.nodes()], 
        #             s=10, color='black')
        # draw regions
        for pos, label in motionmap.regions.items():
            ax.add_patch(plt.Rectangle((pos[0]-label['range']/2, 
                                    pos[1]-label['range']/2), 
                                    width=label['range'], height=label['range'], 
                                    facecolor=label['color'], edgecolor='black', 
                                    linestyle='-', alpha=0.6))
            ax.text(x=pos[0], y=pos[1]+res/8, s=label['name'], 
                    fontsize=8, color='black', weight='normal',
                    verticalalignment='center', horizontalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        # draw agents
        for age in agents:
            points = np.array([[p[0][0] for p in age.path], 
                                [p[0][1] for p in age.path]])
            points_f = np.array([[p[0][0] for p in age.path[:-1]], 
                                [p[0][1] for p in age.path[:-1]]])
            points_t = np.array([[p[0][0] for p in age.path[1:]], 
                                [p[0][1] for p in age.path[1:]]])
            arrows = points_t - points_f
            ax.scatter(age.init_pos[0], age.init_pos[1], 
                    s=100, c=colorbar[age.id], marker='*')
            ax.quiver(points_f[0], points_f[1], arrows[0], arrows[1], 
                    color=colorbar[age.id], scale=1000, width=0.003, 
                    label='${A_{%s}}$' %{age.id},)
        ax.legend()
        plt.savefig(f'{path}/figures/demo_01.png', dpi=200, bbox_inches='tight')
        plt.show()

    elif mode == 'video':
        for t in range(t_max):
            # display the image
            ax.imshow(motionmap.img, origin='lower', alpha=0.2)
            ax.axis('on')
            ax.grid(ls='--', zorder=0)
            # draw regions
            for pos, label in motionmap.regions.items():
                ax.add_patch(plt.Rectangle((pos[0]-label['range']/2, 
                                        pos[1]-label['range']/2), 
                                        width=label['range'], height=label['range'], 
                                        facecolor=label['color'], edgecolor='black', 
                                        linestyle='-', alpha=0.6))
                ax.text(x=pos[0], y=pos[1]+res/8, s=label['name'], 
                        fontsize=8, color='black', weight='normal',
                        verticalalignment='center', horizontalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            # draw agents
            for age in agents:
                if t < len(age.path):
                    x, y = age.path[t][0][0], age.path[t][0][1]
                    ax.scatter(x, y, s=100, c=colorbar[age.id], marker='*', 
                            label='${A_{%s}}$' %{age.id},)
                    if age.path[t][1] != None:
                        ax.text(x, y+5, s=age.path[t][1], fontsize=8, 
                                color='black', weight='normal', alpha=0.5,
                                verticalalignment='center', horizontalalignment='center',
                                bbox=dict(boxstyle='round', facecolor='white', ))
                else:
                    tt = len(age.path) - 1
                    ax.scatter(age.path[tt][0][0], age.path[tt][0][1], 
                            s=100, c=colorbar[age.id], marker='*', 
                            label='${A_{%s}}$' %{age.id},)
            ax.legend()
            path = os.path.abspath(os.path.dirname(
                                    os.path.dirname(
                                        os.path.dirname(__file__))))
            plt.savefig(f'{path}/demos/demo_01/{t}.png', dpi=200)
            ax.cla()