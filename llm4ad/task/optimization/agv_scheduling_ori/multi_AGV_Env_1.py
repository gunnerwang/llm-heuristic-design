'''
# 多AGV场景强化学习环境
# 经过二次简化后的场景，一个上料口，一个下料口，一个暂存区
# 简单网络结构
'''

import numpy as np
import time
import random
import torch

class multiAGV_Env:
    def __init__(self):
        # 常量参数
        self.part_num = 50
        self.tray_num = 10
        self.AGV_num = 6
        self.part_max_num = 50
        self.AGV_max_num = 6
        self.tray_max_num = 10
        self.B_location_num = 5 # B加工区的数量
        self.C_location_num = 5 # C缓存区数量
        self.t_A_part_interval = 50 # 来料间隔时间
        self.t_B_processing = 600
        self.t_AB = 311
        self.t_AC = 201
        self.t_BC = 141
        self.action_list = [0,1,2,3]
        self.distance_fac = 0.01
        self.time_fac = 0.01

        # # 常量参数
        # self.part_num = 10
        # self.tray_num = 6
        # self.AGV_num = 1
        # self.part_max_num = 10
        # self.AGV_max_num = 1
        # self.tray_max_num = 6
        # self.B_location_num = 5 # B加工区的数量
        # self.C_location_num = 5 # C缓存区数量
        # self.t_A_part_interval = 50 # 来料间隔时间
        # self.t_B_processing = 600
        # self.t_AB = 311
        # self.t_AC = 201
        # self.t_BC = 141
        # self.action_list = [0,1,2,3]
        # self.distance_fac = 0.01
        # self.time_fac = 0.01
        

    def reset(self):
        # self.part_num = random.randint(15,self.part_max_num)
        # self.AGV_num = random.randint(2,self.AGV_max_num)
        # self.tray_num = 2*self.AGV_num

        # 其他变量
        self.agv_action_buffer = {} # agv_action_buffer[time] = {'AGV_index':第几辆AGV, 'node': 前往的地点, 'action':'要输出的内容'}

        # 上料口A信息
        self.A_part_info = np.zeros((self.part_max_num, 2), dtype=np.int32)
        for i in range(self.part_num):
            self.A_part_info[i,0] = 1
            self.A_part_info[i,1] = i*self.t_A_part_interval
        self.A_tray = np.zeros(self.tray_max_num, dtype=np.int32)
        for i in range(self.tray_num):
            self.A_tray[i] = 1

        # 加工区B信息
        self.B_info = np.zeros((self.B_location_num, 3),dtype=np.int32)

        # 缓存区C信息
        self.C_info = np.zeros((self.C_location_num, 3), dtype=np.int32)

        # AGV状态信息
        self.AGV_info = np.zeros((self.AGV_max_num, 4), dtype=np.int32)
        self.AGV_timer = np.zeros(self.AGV_num, dtype=np.int32) # 用于计算每辆AGV的全局时间
        self.time_queue = []
        # print(self.A_part_info)
        # print(self.A_tray)
        # print(self.B_info)
        # print(self.C_info)
        # print(self.AGV_info)
        # print(self.AGV_timer)
    
    def reset_test(self):
        # 其他变量
        self.agv_action_buffer = {} # agv_action_buffer[time] = {'AGV_index':第几辆AGV, 'node': 前往的地点, 'action':'要输出的内容'}

        # 上料口A信息
        self.A_part_info = np.zeros((self.part_max_num, 2), dtype=np.int32)
        for i in range(self.part_num):
            self.A_part_info[i,0] = 1
            self.A_part_info[i,1] = i*self.t_A_part_interval
        self.A_tray = np.zeros(self.tray_max_num, dtype=np.int32)
        for i in range(self.tray_num):
            self.A_tray[i] = 1

        # 加工区B信息
        self.B_info = np.zeros((self.B_location_num, 3),dtype=np.int32)

        # 缓存区C信息
        self.C_info = np.zeros((self.C_location_num, 3), dtype=np.int32)

        # AGV状态信息
        self.AGV_info = np.zeros((self.AGV_max_num, 4), dtype=np.int32)
        self.AGV_timer = np.zeros(self.AGV_num, dtype=np.int32) # 用于计算每辆AGV的全局时间
        self.time_queue = []


    # 修正time_queue中的时间信息
    def time_correction(self, list, time):
        if list.count(time) != 0:
            time_1 = time + np.random.random()
            while list.count(time_1) != 0:
                time_1 = time + np.random.random()
            list.append(time_1)
            return time_1, list
        else:
            list.append(time)
            return time, list
        
    # 输出场景的信息
    def return_state_info(self):
        # A信息聚合
        A_vec = np.concatenate((self.A_part_info.flatten(order='F'), self.A_tray))
        # B信息
        B_vec = self.B_info.flatten(order='F')
        # C信息
        C_vec = self.C_info.flatten(order='F')
        # AGV信息
        AGV_vec = self.AGV_info.flatten(order='F')
        return [A_vec, B_vec, C_vec, AGV_vec]
        # return [torch.from_numpy(A_vec), torch.from_numpy(B_vec), torch.from_numpy(C_vec), torch.from_numpy(AGV_vec)]
    
    # 返回上料口A最近时间出来的一个物料，返回最近的一个托盘
    def return_nearest_A_part_tray_index(self):
        part_index = 0
        tray_index = 0
        for i in range(self.part_num):
            if self.A_part_info[i,0] != 0:
                part_index = i + 1
                break
        for i in range(self.tray_num):
            if self.A_tray[i] != 0:
                tray_index = i + 1
                break
        return part_index, tray_index

    # 返回B/C中最近的一个空库位
    def return_nearest_available_pos(self, position):
        if position == 'B':
            for i in range(self.B_info.shape[0]):
                if self.B_info[i,0] == 0:
                    return i 
        else:
            for i in range(self.C_info.shape[0]):
                if self.C_info[i,0] == 0 and self.C_info[i,2] == 0:
                    return i
                
    def return_C_position_reserved(self):
        for i in range(self.C_info.shape[0]):
            if self.C_info[i,0] == 0 and self.C_info[i,2] == 1:
                return i
                
    def return_available_part_in_C(self):
        for i in range(self.C_info.shape[0]):
            if self.C_info[i,0] != 0:
                return i 


    # 返回B中加工完成时间最短的物料index
    def return_near_finished_part_pos_in_B(self):
        _list = self.B_info[:,2]
        for i in range(len(_list)):
            if _list[i] > 0:
                result = _list[i]
                index = i
        for i in range(len(_list)):
            if _list[i] > 0 and _list[i] <= result:
                index = i
                result = _list[i]
        return index
    
    # action: 0/2 从A拿物料和托盘前往B/C
    def take_part_from_A(self, AGV_index, start_location, destination):
        reward = 0
        if start_location == 'A': extra_t = 0
        elif start_location == 'B': 
            extra_t = self.t_AB
        else: 
            extra_t = self.t_AC
        if destination == 'B': travel = self.t_AB
        else: travel = self.t_AC
        A_part_index, A_tray_index = self.return_nearest_A_part_tray_index()
        # 更新AGV信息
        if destination == 'B':
            self.AGV_info[AGV_index, 0] = 2
        else:
            self.AGV_info[AGV_index, 0] = 3
        self.AGV_info[AGV_index, 1] = A_part_index
        self.AGV_info[AGV_index, 2] = A_tray_index
        self.AGV_info[AGV_index, 3] = extra_t + max(self.A_part_info[A_part_index-1,1] - extra_t - self.AGV_timer[AGV_index], 0) + travel
        self.AGV_timer[AGV_index] += self.AGV_info[AGV_index, 3]
        # 更新上料口A的信息
        self.A_part_info[A_part_index-1,:] = 0
        self.A_tray[A_tray_index-1] = 0
        # 更新B/C的信息
        part_index = self.AGV_info[AGV_index, 1]
        tray_index = self.AGV_info[AGV_index, 2]
        if destination == 'B':
            # 更新加工区B信息（物料和工件会在AGV到达时送达，因此加工需要额外加上AGV的运动时间）
            B_index = self.return_nearest_available_pos('B')
            self.B_info[B_index, 0] = part_index
            self.B_info[B_index, 1] = tray_index
            self.B_info[B_index, 2] = self.t_B_processing + self.AGV_timer[AGV_index]
        else:
            # 更新C信息，显示被预约
            C_index = self.return_nearest_available_pos('C')
            self.C_info[C_index, 2] = 1
        # 更新reward
        reward -= self.distance_fac*extra_t + self.time_fac*max(self.A_part_info[A_part_index-1,1] - extra_t - self.AGV_timer[AGV_index], 0)
        # 更新buffer
        corrected_time, self.time_queue = self.time_correction(self.time_queue, self.AGV_timer[AGV_index])
        # self.time_queue.append(self.AGV_timer[AGV_index])
        if destination == 'B':
            self.agv_action_buffer[corrected_time] = {'AGV_index': AGV_index, 'node': destination, \
                'action':'AGV {} is going from {} to {} with part {} tray {}'.format(AGV_index, start_location, destination, part_index, tray_index), 'action_num': 0}
        else:
            self.agv_action_buffer[corrected_time] = {'AGV_index': AGV_index, 'node': destination, \
                'action':'AGV {} is going from {} to {} with part {} tray {}'.format(AGV_index, start_location, destination, part_index, tray_index), 'action_num': 2}
            # print(self.agv_action_buffer[corrected_time]['action'])
        # print(self.B_info)
        return reward
    
    # AGV把物料和托盘放在加工区B
    def put_part_in_B(self, AGV_index):
        # part_index = self.AGV_info[AGV_index, 1]
        # tray_index = self.AGV_info[AGV_index, 2]
        # B_index = self.return_nearest_available_pos('B')
        # # 更新加工区B的信息
        # self.B_info[B_index, 0] = part_index
        # self.B_info[B_index, 1] = tray_index
        # self.B_info[B_index, 2] = self.t_B_processing
        # 更新AGV信息（只保留点位信息）
        self.AGV_info[AGV_index, :] = 0
        self.AGV_info[AGV_index, 0] = 2
    
    # AGV把物料和托盘放在缓存区C
    def put_part_in_C(self, AGV_index):
        # print('AGV INFO')
        # print(self.AGV_info)
        # print('C INFO')
        # print(self.C_info)
        part_index = self.AGV_info[AGV_index, 1]
        tray_index = self.AGV_info[AGV_index, 2]
        C_index = self.return_C_position_reserved()
        # print(C_index)
        # # 更新加工区C的信息
        self.C_info[C_index, 0] = part_index
        self.C_info[C_index, 1] = tray_index
        # 更新AGV信息
        self.AGV_info[AGV_index, :] = 0
        self.AGV_info[AGV_index, 0] = 3
        # print(self.C_info)

    # action: 1 从B点回收托盘前往A
    def recycle_tray_from_B_to_A(self, AGV_index, start_location):
        reward = 0
        if start_location == 'B': extra_t = 0
        elif start_location == 'C': extra_t = self.t_BC
        else: extra_t = self.t_AB
        tray_index = self.return_near_finished_part_pos_in_B()
        # 更新AGV信息
        self.AGV_info[AGV_index, 0] = 1
        self.AGV_info[AGV_index, 1] = 0
        self.AGV_info[AGV_index, 2] = self.B_info[tray_index, 1]
        self.AGV_info[AGV_index, 3] = extra_t + max(0, self.B_info[tray_index, 2]-extra_t-self.AGV_timer[AGV_index]) + self.t_AB
        # 更新reward
        reward -= self.distance_fac*extra_t + self.time_fac*max(0, self.B_info[tray_index, 2]-extra_t-self.AGV_timer[AGV_index])
        self.AGV_timer[AGV_index] += self.AGV_info[AGV_index, 3]
        # 更新buffer
        # self.time_queue.append(self.AGV_timer[AGV_index])
        corrected_time, self.time_queue = self.time_correction(self.time_queue, self.AGV_timer[AGV_index])
        self.agv_action_buffer[corrected_time] = {'AGV_index': AGV_index, 'node': 'A', \
            'action':'AGV {} is going from B to A with part none tray {}'.format(AGV_index, self.B_info[tray_index, 1]), 'action_num': 1}
        # 更新B信息
        self.B_info[tray_index, :] = 0
        
        return reward
    
    # 在A点放置托盘
    def put_tray_in_A(self, AGV_index):
        # print(self.AGV_info)
        tray_index = self.AGV_info[AGV_index, 2]
        # 更新A信息
        self.A_tray[tray_index-1] = 1
        # 更新AGV信息
        self.AGV_info[AGV_index, :] = 0

    #action: 3 AGV把物料和托盘从C运往B
    def carry_partandtray_from_C_to_B(self, AGV_index, start_location):
        reward = 0
        if start_location == 'C': extra_t = 0
        elif start_location == 'B': extra_t = self.t_BC
        else: extra_t = self.t_AC
        C_index = self.return_available_part_in_C()
        part = self.C_info[C_index, 0]
        tray = self.C_info[C_index, 1]
        # 更新AGV信息
        self.AGV_info[AGV_index, 0] = 2
        self.AGV_info[AGV_index, 1] = part
        self.AGV_info[AGV_index, 2] = tray
        self.AGV_info[AGV_index, 3] = extra_t + self.t_BC
        self.AGV_timer[AGV_index] += self.AGV_info[AGV_index, 3]
        # 更新C信息
        self.C_info[C_index, :] = 0
        # 更新B信息
        B_index = self.return_nearest_available_pos('B')
        self.B_info[B_index, 0] = part
        self.B_info[B_index, 1] = tray
        self.B_info[B_index, 2] = self.t_B_processing + self.AGV_timer[AGV_index]
        # 更新buffer
        # self.time_queue.append(self.AGV_timer[AGV_index])
        corrected_time, self.time_queue = self.time_correction(self.time_queue, self.AGV_timer[AGV_index])
        self.agv_action_buffer[corrected_time] = {'AGV_index': AGV_index, 'node': 'B', \
            'action':'AGV {} is going from C to B with part {} tray {}'.format(AGV_index, part, tray), 'action_num': 3}
        # 更新reward
        reward -= extra_t*self.distance_fac
        return reward

    # 检查Action是否valid
    # 检查action0，A有物料，A有托盘，B有空位置
    def check_action0_valide(self):
        if max(self.A_part_info[:,0]) == 1 and max(self.A_tray[:]) == 1 and min(self.B_info[:,0]) == 0:
            return True
        else:
            return False

    # 检查action1
    def check_action1_valide(self):
        if max(self.B_info[:,0]) >= 1:
            return True
        else:
            return False

    # 检查action2
    def check_action2_valide(self):
        cnt = 0
        if max(self.A_part_info[:,0]) == 1 and max(self.A_tray[:]) == 1:
            for i in range(self.C_location_num):
                if self.C_info[i,0] == 0 and self.C_info[i,2] == 0:
                    cnt += 1
        if cnt == 0:
            return False
        else:
            return True


    # 检查action3
    def check_action3_valide(self):
        if max(self.C_info[:,0]) >= 1 and min(self.B_info[:,0]) == 0:
            return True
        else:
            return False

    # 初始状态调度
    def simulation_initialize(self):
        self.reset()
        for i in range(self.AGV_num):
            self.take_part_from_A(i, 'A', 'B')

    def select_action_greedy(self, AGV_index, node):
        reward_list = []
        if self.check_action0_valide():
            if node == 'A':
                extra_t = 0
            elif node == 'B':
                extra_t = self.t_AB
            elif node == 'C':
                extra_t = self.t_AC
            A_part_index, A_tray_index = self.return_nearest_A_part_tray_index()
            reward = self.distance_fac*extra_t + self.time_fac*max(self.A_part_info[A_part_index-1,1] - extra_t - self.AGV_timer[AGV_index], 0)
            reward_list.append(reward)
        else:
            reward_list.append(10e5)
        if self.check_action1_valide():
            if node == 'A':
                extra_t = self.t_AB
            elif node == 'B':
                extra_t = 0
            elif node == 'C':
                extra_t = self.t_BC
            tray_index = self.return_near_finished_part_pos_in_B()
            reward = self.distance_fac*extra_t + self.time_fac*max(0, self.B_info[tray_index, 2]-extra_t-self.AGV_timer[AGV_index])
            reward_list.append(reward)
        else:
            reward_list.append(10e5)
        if self.check_action2_valide():
            if node == 'A':
                extra_t = 0
            elif node == 'B':
                extra_t = self.t_AB
            elif node == 'C':
                extra_t = self.t_AC
            A_part_index, A_tray_index = self.return_nearest_A_part_tray_index()
            reward = self.distance_fac*extra_t + self.time_fac*max(self.A_part_info[A_part_index-1,1] - extra_t - self.AGV_timer[AGV_index], 0)
            reward_list.append(reward)
        else:
            reward_list.append(10e5)
        if self.check_action3_valide():
            if node == 'C': extra_t = 0
            elif node == 'B': extra_t = self.t_BC
            else: extra_t = self.t_AC
            reward = extra_t*self.distance_fac
            reward_list.append(reward)
        else:
            reward_list.append(10e5)
            # print(reward_list)
        if min(reward_list) == 10e5:
            return -1
        else:
            return np.argmin(reward_list)
        
    def select_action_shortest_distance(self, AGV_index, node):
        reward_list = []
        if self.check_action0_valide():
            if node == 'A':
                extra_t = 0
            elif node == 'B':
                extra_t = self.t_AB
            elif node == 'C':
                extra_t = self.t_AC
            A_part_index, A_tray_index = self.return_nearest_A_part_tray_index()
            reward = extra_t
            reward_list.append(reward)
        else:
            reward_list.append(10e5)
        if self.check_action1_valide():
            if node == 'A':
                extra_t = self.t_AB
            elif node == 'B':
                extra_t = 0
            elif node == 'C':
                extra_t = self.t_BC
            tray_index = self.return_near_finished_part_pos_in_B()
            reward = extra_t
            reward_list.append(reward)
        else:
            reward_list.append(10e5)
        if self.check_action2_valide():
            if node == 'A':
                extra_t = 0
            elif node == 'B':
                extra_t = self.t_AB
            elif node == 'C':
                extra_t = self.t_AC
            A_part_index, A_tray_index = self.return_nearest_A_part_tray_index()
            reward = extra_t
            reward_list.append(reward)
        else:
            reward_list.append(10e5)
        if self.check_action3_valide():
            if node == 'C': extra_t = 0
            elif node == 'B': extra_t = self.t_BC
            else: extra_t = self.t_AC
            reward = extra_t
            reward_list.append(reward)
        else:
            reward_list.append(10e5)
            # print(reward_list)
        if min(reward_list) == 10e5:
            return -1
        else:
            return np.argmin(reward_list)





if __name__ == "__main__": 
    solver = multiAGV_Env()
    solver.reset()
    # solver.simulation_initialize()
    # solver.recycle_tray_from_B_to_A(0, 'B')
    # solver.take_part_from_A(1, 'B', 'B')
    # solver.take_part_from_A(2, 'B', 'B')
    solver.take_part_from_A(0, 'A', 'C')
    solver.put_part_in_C(0)
    solver.carry_partandtray_from_C_to_B(1, 'C')
    # solver.recycle_tray_from_B_to_A(0, 'C')
    # solver.recycle_tray_from_B_to_A(1, 'B')
    # solver.recycle_tray_from_B_to_A(2, 'B')
    # solver.recycle_tray_from_B_to_A(1, 'A')
    # solver.recycle_tray_from_B_to_A(2, 'A')
    # solver.recycle_tray_from_B_to_A(1, 'A')
    # solver.recycle_tray_from_B_to_A(2, 'A')
    # solver.recycle_tray_from_B_to_A(0, 'A')
    # solver.recycle_tray_from_B_to_A(1, 'A')
    print(solver.A_part_info)
    print(solver.A_tray)
    print(solver.B_info)
    print(solver.C_info)
    print(solver.AGV_info)
    print(solver.AGV_timer)
