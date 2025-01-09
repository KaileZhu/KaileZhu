import numpy as np
from .poset_builder import Buchi_poset_builder
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class Poset_producter(object):
	def __init__(self,ltl_formula_list):
		'''
		这部分为偏序处理部分，用于优化偏序结构，处理一些参数细节等等。

		'''
		self.poset_list=ltl_formula_list
		self.gantt_data_dic={}

	def generate_poset(self):
		self.ltl2poset={}
		self.ltl2poset[1] = []
		for  formula_list in self.poset_list :
				buchi=Buchi_poset_builder(formula_list)
				buchi.main_fun_to_get_poset(20)
				self.ltl2poset[1].append(buchi)

	def prodocter(self):
		self.final_poset = {'||': set(),
							'<=': set(),
							'<': set(),
							'!=': set(),
							'=': set(),
							'action_map': []}
		self.final_task_data_list=[]
		final_round_poset = {}
		for round,ltl2poset in self.ltl2poset.items():

			for poset in ltl2poset:
				sub_poset=poset.poset_list[0]
				sub_task_data_list=poset.task_data_list[0]
				#judge the task list
				if len(self.final_poset['action_map'])==0:
					self.final_poset['action_map']=poset.task_data_list[0]
					self.final_poset['<=']=sub_poset['<=']
					self.final_poset['!=']=sub_poset['!=']
					self.final_task_data_list.extend(sub_task_data_list)
				else:
					n=len(self.final_poset['action_map'])
					new_sub_task_data_list=[]
					if len(np.shape(poset.task_data_list))==3:
						for task in poset.task_data_list[0] :
							new_sub_task_data_list.append((task[0]+n,task[1],task[2],task[3],task[4]))
					else:
						for task in poset.task_data_list :
							new_sub_task_data_list.append((task[0]+n,task[1],task[2],task[3],task[4]))
					self.final_poset['action_map'].extend(new_sub_task_data_list)
					self.final_task_data_list.extend(new_sub_task_data_list)
					for i,j in sub_poset['<=']:
						self.final_poset['<='].add((i+n,j+n))
					for i,j in sub_poset['!=']:
						self.final_poset['!='].add((i+n,j+n))
			if len(final_round_poset)==0:
				final_round_poset=range(len(self.final_task_data_list))
			else:
				for i in range(len(self.final_task_data_list)):
					for j in final_round_poset:
						if not i in final_round_poset:
							self.final_poset['<='].add((j,i))
							self.final_poset['!='].add((j,i))
				final_round_poset=range(len(self.final_task_data_list))

			#实际上只需要在添加一组考虑波次的order的即可

	def gantt_plotter(self,  poset, estimate_time_table,task_time_table ):
		# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
		plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
		plt.figure(figsize=(10, 5), dpi=80)
		# fig, ax=plt.subplots()
		max_time=max([time_list[2] for time_list in task_time_table])
		color_dic = {}
		color = ['b', 'c', 'r', 'y', 'm', 'g', 'gold', 'indigo', 'violet', 'lime', 'peru', 'pink', 'brown', 'b',
				 'c', 'r', 'y', 'm', 'brown', 'b', 'c', 'r', 'y', 'm', 'g', 'gold', 'indigo', 'violet', 'b', 'c', 'r',
				 'y', 'm', 'g', 'gold', 'indigo', 'violet', ]
		# self.poset=poset
		for i in range(35):
			color_dic[i] = color[i]
		i = 1
		color_dic['error'] = color[i]
		for agent in  estimate_time_table:
			for task in agent:
				task_id=task[0][0]
				plt.barh(i, task_time_table[task_id][2]-task_time_table[task_id][1],
						 left=task_time_table[task_id][1], color=color_dic[task_id], linewidth=5,
						  alpha=0.8)
			# print('left=',time[task[0][0]][2])
			i = i + 1

		plt.xlim(0, max_time//10*10+30)
		plt.ylim(0, 20)
		x_tick = np.linspace(0, max_time//10*10+30, 11)
		y_tick = np.linspace(1, len(estimate_time_table), 5)
		plt.yticks(y_tick, fontsize=20)
		plt.xticks(x_tick, fontsize=20)
		text_list=[]
		patches = [mpatches.Patch(color=color[i[0]], label=i[1]+i[2]) for i in poset['action_map']]
		# self.ax.legend()
		# plt.legend(loc='lower right', handles=patches, fontsize='25')
		plt.legend(loc=2, bbox_to_anchor=(-0.03, 1.2), handles=patches, fontsize='15', ncol=5)

		plt.title("Task assigment Gantt graph", fontsize=30)
		# XY轴标签
		plt.xlabel("time/s", fontsize=30)
		plt.ylabel("agent", fontsize=30)
		plt.show()


	def Turn_Matrix(self, M):
		r = [[] for i in M[0]]
		for i in M:
			for j in range(len(i)):
				r[j].append(i[j])
		return r