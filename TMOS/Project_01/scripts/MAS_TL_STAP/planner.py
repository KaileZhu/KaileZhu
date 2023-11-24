#！/usr/bin/env python3
'''
@Date   : 2023/11 -->  
@Authors: Junjie Wang, Yunyi Zhang
@Contact: pkuwjj1998@163.com
@Version: 1.0
@Descrip: the module contains the class for planner, which is used to solve
        the simultaneous task allocation and planning for multi-agent system.
@Log:
        - 2023/11/20: the first version that contains the class for BnB Search
'''

import time
import copy
import random
import numpy as np

import cvxpy as cp
import networkx as nx

# from B_A_B2 import Branch_And_Bound

class BnBSearch:
    """
    Class for the branch and bound algorithm.
    """
    def __init__(self, poset, task_data, input_data, ):
        """
        ----------
        Args:
            poset: the poset list
            task: 
            input: 
        """
        self.poset = poset
        self.position = input_data.position
        self.agent_data = input_data.agent_data
        self.task_data = task_data
        self.task_type = input_data.task_type
        self.sub_task_type = input_data.sub_task_type
        self.agent_type = input_data.agent_type
        self.Astar_table = []
        self.get_horizon()
        self.explored_node_dic = {}
        self.generate_poset_graph()

    def begin_bnb_search(self, time_budget, upper_bound, lower_bound, search_method):
        time_start = time.time()
        self.generate_time_budget()
        node_root = list([] for _ in self.agent_data)
        tasks_assigned = list()
        self.round = 1
        up_bound,solution = self.get_upper_bound(node_root, tasks_assigned, 'slowly')
        tasks_assigned = list()
        self.get_lower_bound(lower_bound)

    def generate_time_budget(self):
        self.time_budget = 0
        for i in self.task_data:
            self.time_budget = self.time_budget + self.task_type[self.task_data[i[0]][1]][0]

    def get_upper_bound(self, node, tasks_assigned, upper_bound):
        if upper_bound=='greedy':#should be finished
            up_bound,solution=self.found_solution_greedy(node,tasks_assigned)
            return up_bound,solution
        elif upper_bound=='slowly':
            up_bound, solution = self.found_solution_greedy2(node, tasks_assigned)
            return up_bound, solution
        else:
            raise Exception('Undefined upper_bound method!')

    def found_solution_greedy(self, init_node, tasks_assigned, repeat_number=1):
        """
        
        """
        #print('assigned_task',tasks_assigned)
        if len(tasks_assigned) == 0:
            t = 0
        else:
            t,_ = self.optimal_partial_assign(init_node, tasks_assigned)
        t2 = 0
        for i in tasks_assigned:
            num_execute = 0
            for n in self.task_type[self.task_data[i][1]][1].values():
                num_execute = num_execute + n
            t2 = t2 + self.task_type[self.task_data[i][1]][0] * num_execute
            #t2=t2+self.task_type[self.task_data[i][1]][0]
        if t<=0.01:
            tstar=1
        else:
            tstar=t2/t/len(self.agent_data)
        sequence = [(set(tasks_assigned), init_node)]
        sequence_t=[tstar]
        un_found=1
        sample_list=[]
        while sequence!=[] and un_found>0:
            popi=sequence_t.index(max(sequence_t))
            root_node=sequence.pop(popi)
            t_label=sequence_t.pop(popi)
            assiged_task=root_node[0]
            init_node=root_node[1]
            #================get to assign task
            assiged_task_set=set()
            to_assig_task=set(self.poset_graph.succ['root'])
            for task in assiged_task:
                assiged_task_set.add(task)
                to_assig_task=to_assig_task|set(self.poset_graph.succ[task])
            un_assig_task1=to_assig_task-assiged_task_set
            un_assig_task=copy.deepcopy(un_assig_task1)
            for i in un_assig_task1:
                if not len(set(self.poset_graph.pred[i])-assiged_task_set-{'root'})==0:
                    un_assig_task.remove(i)
            #-===== already get feasible assgin task
            sequence=[]
            sequence_t=[]
            for to_assig_task in un_assig_task:
                new_assiged_task=assiged_task |{to_assig_task}
                #print(self.task_type.keys())
                sub_task_list=self.task_type[self.task_data[to_assig_task][1]][1]
                feasible={}
                for sub_task,num in sub_task_list.items():
                    list=[]
                    for agent_i in self.agent_data:
                        if sub_task in self.agent_type[agent_i[2]]['serve']:
                            list.append(agent_i[0])
                    feasible[sub_task]=(list,num)
            #== get task distribution
                for z in range(repeat_number):
                    new_node=copy.deepcopy(init_node)
                    while 1:
                        assign_list={}
                        check_list=[]
                        for sub_task,(list,num) in feasible.items():
                            assign_list[sub_task] = random.sample(list,num)
                            check_list.extend(assign_list[sub_task])
                            repet_Num = Counter(check_list)
                        if len(repet_Num)==len(check_list):
                            break
                    #print('assign_list',assign_list)
                    for sub_task,samb in assign_list.items():
                        for agent_i in samb:
                            new_node[agent_i].append((self.task_data[to_assig_task], sub_task))
                    #calculate t *
                    if len(new_assiged_task)==len(self.task_data):
                        sample_list.append(new_node)
                        un_found=un_found-1
                    else:
                        sequence.append((new_assiged_task,new_node))
                        t,_=self.opt_for_partial_assigment(new_node,new_assiged_task)
                        #print('time',t)
                        t2=0
                        for i in new_assiged_task:
                            num_execute=0
                            for n in self.task_type[self.task_data[i][1]][1].values():
                                num_execute=num_execute+n
                            t2=t2+self.task_type[self.task_data[i][1]][0]*num_execute
                            #t2=t2+self.task_type[self.task_data[i][1]][0]
                        if t<=0.1:
                            tstar=1
                        else:
                            tstar=t2/t/len(self.agent_data)
                        sequence_t.append(tstar)
        time_list=[]
        if sample_list==[]:
            return self.horizon,[]
        else:
            for node in sample_list:
                a,b=self.opt_for_partial_assigment(node,range(len(self.task_data)))
                time_list.append(a)
            solution=sample_list[time_list.index(min(time_list))]
            return min(time_list),solution

    def found_solution_greedy2(self, node_par, tasks_assigned):
        if not len(tasks_assigned)==0:
            t,_=self.optimal_partial_assign(init_node,tasks_assigned)
        else:
            t = 0
        t2 = 0
        for i in tasks_assigned:
            t2 = t2+self.task_type[self.task_data[i][1]][0]

        tstar = t2/t/len(self.agent_data) if t > 0.1 else 1

        node = {
            'plan': node_par,
            'tasks': set(tasks_assigned),
            'time': tstar
        }
        nodes_seq = [node, ]  # a sequence of nodes that contains the tasks assignment
        time_seq = [tstar, ]  # a time sequence of the 
        un_found = 1
        sample_list = []
        while nodes_seq != None and un_found > 0:
            popi = time_seq.index(min(sequence_t))
            root_node = nodes_seq.pop(popi)
            t_label = time_seq.pop(popi)
            tasks_assigned = root_node['tasks']
            init_node = root_node['plan']

            assiged_task_set = set()
            to_assig_task = set(self.poset_graph.succ['root'])
            # 
            for task in tasks_assigned:
                assiged_task_set.add(task)
                to_assig_task = to_assig_task|set(self.poset_graph.succ[task])
            un_assig_task1 = to_assig_task - assiged_task_set
            un_assig_task = copy.deepcopy(un_assig_task1)
            # 
            for i in un_assig_task1:
                if len(set(self.poset_graph.pred[i])-assiged_task_set-{'root'}) != 0:
                    un_assig_task.remove(i)
            #print('unassig:',un_assig_task)
            #print('assig:',assiged_task)
            sequence = []
            sequence_t = []
            for to_assig_task in un_assig_task:
                new_assiged_task = tasks_assigned.add(to_assig_task)
                subtasks = self.task_type[self.task_data[to_assig_task][1]][1]
                feasible = {}
                for subtask, num in subtasks.items():
                    agents = []
                    for agent_i in self.agent_data:
                        if subtask in self.agent_type[agent_i[2]]['actions']:
                            agents.append(agent_i[0])
                    feasible[subtask] = (agents, num)
            #== get task distribution
                new_node = copy.deepcopy(init_node)
                while 1:
                    assign_list={}
                    check_list=[]
                    for sub_task,(list,num) in feasible.items():
                        assign_list[sub_task] = random.sample(list,num)
                        check_list.extend(assign_list[sub_task])
                        repet_Num = Counter(check_list)
                    if len(repet_Num)==len(check_list):
                        break
                #print('assign_list',assign_list)
                for sub_task, samb in assign_list.items():
                    for agent_i in samb:
                        new_node[agent_i].append((self.task_data[to_assig_task],sub_task))
                if len(new_assiged_task)==len(self.task_data):
                    sample_list.append(new_node)
                    un_found=un_found-1
                else:
                    sequence.append((new_assiged_task,new_node))
                    t,_=self.opt_for_partial_assigment(new_node,new_assiged_task)
                    #print('time',t)
                    t2=0
                    for i in new_assiged_task:
                        t2=t2+self.task_type[self.task_data[i][1]][0]
                    if t<=0.1:
                        tstar = 1
                    else:
                        tstar = t2/t/len(self.agent_data)
                    sequence_t.append(tstar)
        time_list=[]
        if sample_list==[]:
            return self.horizon,[]
        else:
            for node in sample_list:
                a,b=self.opt_for_partial_assigment(node,range(len(self.task_data)))
                time_list.append(a)
            solution=sample_list[time_list.index(min(time_list))]
            return min(time_list),solution


    def get_lower_bound(self, lower_bound):
        if lower_bound == 'i_j':
            self.get_lower_bound_method = self.get_lower_bound_with_i_j
        if lower_bound == 'i+j':
            self.get_lower_bound_method = self.get_lower_bound_with_i_add_j
        #low_bound1=self.get_lower_bound_with_i_j_k_faster(node,tasks_assigned)
        #return  low_bound1

    def get_lower_bound_with_i_j(self, node, tasks_assigned):
        """
        
        ----------
        Parameters:

        ----------
        Returns:

        """
        t, time_list = self.opt_for_partial_assigment(node, tasks_assigned)
        tasks_unassigned=list(range(len(self.task_data)))
        assigned_task_dic={}
        z=0
        for i in tasks_assigned:
            tasks_unassigned.remove(i)
            assigned_task_dic[i]=z
            z=z+1
        task_number=len(tasks_unassigned)
        if task_number==0:
            print('return max time list')
            return max(time_list)
        begin_time_list=[]
        for agent in node:
            if len(agent)==0:
                begin_time_list.append(0)
            else:
                assigned_task_dic[agent[-1][0][0]]
                if np.shape(time_list[assigned_task_dic[agent[-1][0][0]]])==(1,):
                    begin_time_list.append(time_list[assigned_task_dic[agent[-1][0][0]]][0])
                else:
                    begin_time_list.append(time_list[assigned_task_dic[agent[-1][0][0]]])
        #print('begin time list',begin_time_list)
        z=0
        task_dic={}
        for i in tasks_unassigned:
            task_dic[i]=z
            z=z+1
        agent_pose=[]
        for agent in self.agent_data:
            if len(node[agent[0]])==0:
                agent_pose.append(self.position[agent[1]])
            else:
                pos=self.position[node[agent[0]][-1][0][2]]
                agent_pose.append(pos)
        agent_number=len(self.agent_data)
        x_i_j=cp.Variable(shape=(agent_number*task_number,1),boolean=True)
        t_i=cp.Variable(shape=(agent_number,1),nonneg=True)
        total_constrain=[]
        new_task_time_set={}
        # begin time constrain
        for task_j in tasks_unassigned:
            time_table=[]
            for agent in self.agent_data:
                x1=self.position[self.task_data[task_j][2]]
                x2=agent_pose[agent[0]]
                time=((x1[0]-x2[0])**2+(x1[1]-x2[1])**2)**0.5/self.agent_type[agent[2]]['velocity']
                time_table.append(time)
            for task in tasks_unassigned:
                if not task==task_j:
                    time=self.get_distance(self.task_data[task_j][2],self.task_data[task][2])/10
                    time_table.append(time)
            new_task_time=self.task_type[self.task_data[task_j][1]][0]+min(time_table)
            new_task_time_set[task_j]=new_task_time
        self.new_task_time_set=new_task_time_set
        M_time=[[0  for i in range(agent_number)] for j in range(agent_number * task_number)]
        b_time=[[begin_time_list[j] for j in range(agent_number)]]
        T_time=[]
        for agent_i in range(agent_number):
            t=[0 for o in range(agent_number)]
            t[agent_i]=1
            for task_j in task_dic.keys():
                num=agent_i*task_number+task_dic[task_j]
                M_time[num][agent_i]=-new_task_time_set[task_j]
            T_time.append(t)
        if len(np.shape(b_time))>2:
            s=1
        constrain_begin_time=[M_time @ x_i_j + T_time @ t_i >= b_time]
        total_constrain.append(*constrain_begin_time)

        # ================provide enough serves for the task j (2)
        M1=[[0 for i in range(task_number)] for j in range(agent_number*task_number)]
        b1=[[0 for i in range(task_number)]]
        #z=0
        for task_j in task_dic.keys():
            task_subs=self.task_type[self.task_data[task_j][1]][1]
            b=0
            for sub,num in task_subs.items():
                b=b+num
            b1[0][task_dic[task_j]]=b
            for agent_i in range(agent_number):
                num=agent_i*task_number+task_dic[task_j]
                M1[num][task_dic[task_j]]=1
        enough_constrain=[M1 @ x_i_j == b1]
        total_constrain.append(*enough_constrain)

        obj_1 = cp.Minimize(cp.max(t_i))

        prob_1=cp.Problem(obj_1,total_constrain)
        #solver: GLPK_MI CBC SCIP
        prob_1.solve(solver='GLPK_MI')
        self.x_i_j=x_i_j.value
        self.t_i=t_i.value
        self.constrain=total_constrain
        print('low bound:',prob_1.value)
        if prob_1.status=='optimal':
            return prob_1.value
        else:
            return 0

    def optimal_partial_assign(self, node_par, tasks_assigned, i=None):
        """
        The optimal assignment according to the partial relation
        ----------
        Parameters:
            node:
            tasks_assigned: 
        ----------
        Returns:
        """
        node_plan = [list(plan) for plan in node_par]
        tuple_node = tuple(node_plan)
        tasks_assigned_tuple = tuple(tasks_assigned)
        if tasks_assigned_tuple in self.explored_node_dic.keys():
            if tuple_node in self.explored_node_dic[tasks_assigned_tuple].keys():
                max_end_time_value,end_time_value=self.explored_node_dic[tasks_assigned_tuple][tuple_node]
                return max_end_time_value, end_time_value
        tasks_assigned_dic={}
        t=0
        #print('tasks_assigned:',tasks_assigned)
        #print(node)
        for i in tasks_assigned:
            tasks_assigned_dic[i]=t
            t=t+1
        end_time=cp.Variable(shape=(len(tasks_assigned),1),name='endtime',nonneg=True)
        total_constrain=[]
        M1=[]
        B1=[[]]
        #for i,j in self.poset['<']:
        for i,j in self.poset['<=']:
            if self.task_data[i][0] in tasks_assigned and self.task_data[j][0] in tasks_assigned:
                if not ((i,j) in self.poset['!='] or (j,i) in self.poset['!=']):
                    #<=
                    m=[0 for l in range(len(tasks_assigned))]
                    m[tasks_assigned_dic[self.task_data[i][0]]]=1
                    m[tasks_assigned_dic[self.task_data[j][0]]]=-1
                    M1.append(m)
                    B1[0].append(-self.task_type[self.task_data[j][1]][0]+self.task_type[self.task_data[i][1]][0])
                else:
                    #<
                    m=[0 for l in range(len(tasks_assigned))]
                    m[tasks_assigned_dic[self.task_data[i][0]]]=1
                    m[tasks_assigned_dic[self.task_data[j][0]]]=-1
                    M1.append(m)
                    B1[0].append(-self.task_type[self.task_data[j][1]][0])
        for i,j in self.poset['=']:
            if self.task_data[i][0] in tasks_assigned and self.task_data[j][0] in tasks_assigned:
                if self.task_type[self.task_data[i][1]][0]>= self.task_type[self.task_data[j][1]][0]:
                    changelabel=-1
                else:
                    changelabel=1
                m=[0 for l in range(len(tasks_assigned))]
                m[tasks_assigned_dic[self.task_data[i][0]]]=changelabel
                m[tasks_assigned_dic[self.task_data[j][0]]]=-changelabel
                M1.append(m)
                B1[0].append(0)
                #might remaining to do!!!!!!!!!!!!!!!
                m=[0 for l in range(len(tasks_assigned))]
                m[tasks_assigned_dic[self.task_data[i][0]]]=-changelabel
                m[tasks_assigned_dic[self.task_data[j][0]]]=changelabel
                M1.append(m)
                B1[0].append( -self.task_type[self.task_data[i][1]][0]*changelabel+self.task_type[self.task_data[j][1]][0]*changelabel)
        for i,j in self.poset['!=']:
            if i in tasks_assigned and j in tasks_assigned:
                if not (i,j) in self.poset['<='] and not (j,i) in self.poset['<=']:
                    m=[[0] for l in range(len(tasks_assigned))]
                    m[tasks_assigned_dic[self.task_data[i][0]]][0]=1
                    m[tasks_assigned_dic[self.task_data[j][0]]][0]=-1
                    bool_for_x=cp.Variable(1,boolean=True)
                    #ei-di - ej  >=0   ti >= ej
                    constrain0=[m @ end_time -self.task_type[self.task_data[i][1]][0] -bool_for_x * self.time_budget+self.time_budget>=0]
                    #ei - ej+ dj  <=0  ei<= tj
                    constrain1=[m @ end_time + self.task_type[self.task_data[j][1]][0] - bool_for_x * self.time_budget <=0]
                    print(1)
                    total_constrain.append(*constrain0)
                    total_constrain.append(*constrain1)
                    #m=[[0] for l in range(len(tasks_assigned))]
                    #m[tasks_assigned_dic[self.task_data[i][0]]][0]=1
                    #m[tasks_assigned_dic[self.task_data[j][0]]][0]=-1
                    #total_constrain.append(cp.abs(
                    #    m @ end_time + (-self.task_type[self.task_data[i][1]][0] + self.task_type[self.task_data[j][1]][0])) \
                    #                       >= (self.task_type[self.task_data[i][1]][0] + self.task_type[self.task_data[j][1]][
                    #    0]) / 2)
            #total_constrain.append(cp.abs(m @ end_time) >=max(self.task_type[self.task_data[i][1]][0],self.task_type[self.task_data[j][1]][0]))
        if not M1==[]:
            M11=self.Turn_Matrix(M1)
            constraint1=[M11 @ end_time <= B1]
            total_constrain.append(*constraint1)
            #print(B1)
        M2=[]
        B2=[[]]
        for agent_i in range(len(self.agent_data)):
            if len(node[agent_i])>0:
                m=[0 for i in range(len(tasks_assigned))]
                #print(tasks_assigned_dic)
                #print(node[agent_i][0][0])
                #print(tasks_assigned_dic)
                c=tasks_assigned_dic[node[agent_i][0][0][0]]
                m[c]=1
                #b=self.get_distance(self.agent_data[agent_i][1],node[agent_i][0][0][2])/self.agent_type[self.agent_data[agent_i][2]]['velocity']+\
                #    self.task_type[node[agent_i][0][0][1]][0]
                M2.append(m)
                B2[0].append(self.get_distance(self.agent_data[agent_i][1],node[agent_i][0][0][2])/self.agent_type[self.agent_data[agent_i][2]]['velocity']+\
                    self.task_type[node[agent_i][0][0][1]][0])
            if len(node[agent_i])>1:
                for task in range(len(node[agent_i])-1):
                    m=[0 for i in range(len(tasks_assigned))]
                    c=tasks_assigned_dic[node[agent_i][task][0][0]]
                    m[c]=-1
                    c=tasks_assigned_dic[node[agent_i][task+1][0][0]]
                    m[c]=1
                    b=self.get_distance(node[agent_i][task][0][2],node[agent_i][task+1][0][2])/self.agent_type[self.agent_data[agent_i][2]]['velocity']+\
                        self.task_type[node[agent_i][task+1][0][1]][0]
                    M2.append(m)
                    B2[0].append((b))
        M21=self.Turn_Matrix(M2)
        constraint2=[M21 @ end_time >= B2]# constraint of poset
        total_constrain.append(*constraint2)
        list1=[[1] for task in tasks_assigned]
        obj = cp.Minimize(list1 @ end_time)
        prob=cp.Problem(obj,total_constrain)
        #prob.solve(solver=cp.SCS)
        prob.solve(solver='GLPK_MI')
        if prob.status=='optimal':
            if tasks_assigned_tuple in self.explored_node_dic.keys():
                if tuple_node in self.explored_node_dic[tasks_assigned_tuple].keys():
                    max_end_time_value, end_time_value = self.explored_node_dic[tasks_assigned_tuple][tuple_node]
            else:
                self.explored_node_dic[tasks_assigned_tuple]={}
                self.explored_node_dic[tasks_assigned_tuple][tuple_node]=(max(end_time.value),end_time.value)
            return max(end_time.value),end_time.value
        else:
            return self.horizon,[]






class MILP:
    """
    Class for the mixed integer linear programming algorithm.
    """
    pass