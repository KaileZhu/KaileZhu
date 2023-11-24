import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from data.input_data.task_data import task_type
import numpy as np

agent_data_s0=[(0,'b1','UAV'),
                    (1,'b1','UAV'),
                    (2,'b1','UAV'),
                    (3,'b1','UAV'),
                    (4,'b1','UAV'),
                    (5, 'b1', 'UAV'),

                    (6,'b1','UGV_l'),
                    (7,'b1','UGV_l'),
                    (8,'b1','UGV_l'),

                    (9,'b1','UGV_s'),
                    (10,'b1','UGV_s'),
                    (11,'b1','UGV_s'),
                       ]
agent_data_s1=[(0,'b1','UAV'),
                    (1,'b1','UAV'),
                    (2,'b1','UAV'),
                    (3,'b1','UAV'),
                    (4,'b1','UGV_l'),
                    (5,'b1','UGV_l'),
                    (6,'b1','UGV_s'),
                    (7,'b1','UGV_s')
                    ]
agent_data_s2=[(0,'b1','UAV'),
                    (1,'b1','UAV'),
                    (2,'b1','UAV'),
                    (3,'b1','UAV'),
                    (4,'b1','UAV'),
                    (5, 'b1', 'UAV'),

                    (6,'b1','UGV_l'),
                    (7,'b1','UGV_l'),
                    (8,'b1','UGV_l'),

                    (9,'b1','UGV_s'),
                    (10,'b1','UGV_s'),
                    (11,'b1','UGV_s'),
                       ]
agent_data_s3=[(0,'b1','UAV'),
                    (1,'b1','UAV'),
                    (2,'b1','UAV'),
                    (3,'b1','UAV'),
                    (4,'b1','UAV'),
                    (5,'b1','UAV'),
                    (6,'b1','UAV'),
                    (7,'b1','UAV'),
                    (8,'b1','UAV'),
                    (9,'b1','UAV'),

                    (10,'b1','UGV_l'),
                    (11,'b1','UGV_l'),
                    (12,'b1','UGV_l'),
                    (13,'b1','UGV_l'),
                    (14,'b1','UGV_l'),

                    (15,'b1','UGV_s'),
                    (16,'b1','UGV_s'),
                    (17,'b1','UGV_s'),
                    (18,'b1','UGV_s'),
                    (19,'b1','UGV_s'),
                       ]


def gantt_plotter(task_number,situation,node_time ,adapt_time_table):
	#plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
	plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
	node_name='/home/LZS/LTL/git/LTL_MAS_C-action/data/output_data/task'+task_number+'/solution_s'+situation+'.npy'
	node=np.load(node_name)
	#node=np.load('/home/LZS/LTL/git/LTL_MAS_C-action/data/output_data/task2/solution_s3.npy')
	time_name='/home/LZS/LTL/git/LTL_MAS_C-action/data/output_data/task'+task_number+'/time_table_s'+situation+'.npy'
	#time=np.load(time_name)
	#time=np.load('/home/LZS/LTL/git/LTL_MAS_C-action/data/output_data/task2/time_table_s3.npy')
	task_data_list_name='/home/LZS/LTL/git/LTL_MAS_C-action/data/output_data/task'+task_number+'/task_data_list.npy'
	task_data_list=np.load(task_data_list_name)
	#task_data_list=np.load('/home/LZS/LTL/git/LTL_MAS_C-action/data/output_data/task2/task_data_list.npy')
	plt.figure(figsize=(21, 12), dpi=80)
	i=1
	color_dic={}
	color = ['b', 'c', 'pink', 'r', 'y','m', 'g', 'aqua', 'brown','tan']
	hatch_list=[]
	for i in range(len(task_data_list)):
		if not task_data_list[i][1]=='error':
			if i < len(color)-1:
				color_dic[task_data_list[i][1]]=color[i]
			else:
				color_dic[task_data_list[i][1]]=color[i-10]
				hatch_list.append(task_data_list[i][1])

	i=1
	for agent in adapt_time_table:
		y=[i,i]
		for task in agent:
			if task[0] == 'motion':
				x=[task[1]+1,task[2]-1]
				plt.plot(x,y,color='r',linewidth=7)
			elif task[0] == 'cooperate':
				x=[task[1]+1,task[2]-1]
				plt.plot(x,y,color='g',linewidth=7)
			elif task[0]=='pretask':
				x=[task[1]+1,task[2]-1]
				plt.plot(x,y,color='b',linewidth=7)
					#plt.barh(i,time[task[0][0]][2]-time[task[0][0]][1],left=time[task[0][0]][1],color=color[task[0][0]])
				#print('left=',time[task[0][0]][2])
		i=i+1
	i=1
	color_dic['error']=color[i]
	for agent in node_time:
		for task in agent:
			if not task[0]=='error':
				if task[0] in hatch_list:
					plt.barh(i,task[2]-task[1],left=task[1],color=color_dic[task[0]],hatch='/')
				else:
					plt.barh(i,task[2]-task[1],left=task[1],color=color_dic[task[0]])
					#plt.barh(i,time[task[0][0]][2]-time[task[0][0]][1],left=time[task[0][0]][1],color=color[task[0][0]])
				#print('left=',time[task[0][0]][2])
		i=i+1

	plt.xlim(0,1800)
	x_tick=np.linspace(0,1800,19)
	#len(node)
	y_tick=np.linspace(1,len(node),len(node))
	if situation=='0':
		index_list=[agent[2]+' '+str(agent[0]+1) for agent in  agent_data_s2]
	if situation=='1':
		index_list=[agent[2]+' '+str(agent[0]+1) for agent in  agent_data_s1]
	if situation=='2':
		index_list=[agent[2]+' '+str(agent[0]+1) for agent in  agent_data_s2]
	if situation=='3':
		index_list=[agent[2]+' '+str(agent[0]+1) for agent in  agent_data_s3]
	index_list=[]
	for agent in agent_data_s2:
		if agent[2]=='UAV':
			index_list.append(agent[2]+' '+str(agent[0]+1))
		if agent[2]=='UGV_l':
			index_list.append(agent[2]+' '+str(agent[0]-5))
		if agent[2]=='UGV_s':
			index_list.append(agent[2]+' '+str(agent[0]-8))
	plt.xticks(x_tick,fontsize=20)
	plt.yticks(y_tick,index_list,fontsize=20)
	plt.title("Task assigment Gantt graph",fontsize=30)

	#labels=['(1,blowp1)','(2,shootp1)','(3,sweepp1)','(4,shootp12)','(5,sweepp12)','(6,blowp7)','(7,shootp7)','(8,washp7)']

	#tuli:
	patches = []
	for i in range(len(color)):
		patches.append(mpatches.Patch(color=color[i], label="{:s}".format(task_data_list[i][1])))
	for i in range(len(color), len(task_data_list)):
		patches.append(mpatches.Patch(color=color[i - 10], label="{:s}".format(task_data_list[i][1]), hatch='/'))
	plt.legend(loc='right',handles=patches,fontsize='20')

	plt.xlabel("time/s",fontsize=30)
	plt.ylabel("agent ID",fontsize=30)
	# 网格线，此图使用不好看，注释掉
	# plt.grid(linestyle="--",alpha=0.5)
	plt.show()
	#plt.savefig('/home/LZS/LTL/git/LTL_MAS_C-action/data/output_data/task'+task_number+'/gantt_s'+situation+'2_rebuild.eps', dpi=300)
