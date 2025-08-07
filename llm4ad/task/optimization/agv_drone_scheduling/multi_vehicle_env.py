'''
# 多AGV/无人机场景强化学习环境
# 在原有基础上扩展，加入无人机、交通拥堵、加工空转和充电需求
# 简单网络结构
'''

import numpy as np
import time
import random
import torch

class multiVehicleEnv:
    def __init__(self, mode="challenging"):
        self.set_mode_parameters(mode)
        self.action_list = [0, 1, 2, 3, 4]  # 添加充电动作 (4)
        self.distance_fac = 0.01
        self.time_fac = 0.01

    def set_mode_parameters(self, mode):
        if mode == "challenging":
            # 更具挑战性的参数
            self.part_num = 80  # 更多物料
            self.tray_num = 6   # 更少托盘
            self.AGV_num = 3    # 更少AGV
            self.drone_num = 1  # 更少无人机
            self.vehicle_num = self.AGV_num + self.drone_num
            self.part_max_num = 80
            self.vehicle_max_num = 4
            self.tray_max_num = 6
            self.B_location_num = 4 # 更少加工区
            self.C_location_num = 3 # 更少缓存区
            self.t_A_part_interval = 30 # 来料更快
            self.t_B_processing = 900  # 加工更久
            self.t_AB_base = 400  # 距离更远
            self.t_AC_base = 300
            self.t_BC_base = 200
            self.drone_speed_factor = 0.7  # 无人机略慢
            self.traffic_jam_prob = 0.4    # 拥堵概率更高
            self.traffic_jam_factor_agv = 3.0
            self.traffic_jam_factor_drone = 1.5
            self.processing_failure_prob = 0.2  # 空转概率更高
            self.battery_capacity = 60           # 电池更小
            self.battery_consumption_per_distance = 0.08  # 更耗电
            self.battery_critical_level = 25     # 临界电量略高
            self.charging_time_base = 300        # 充电更慢
            self.charging_time_per_unit = 3      # 单位电量充电更慢
            self.charging_station = 'A'
        else:
            # 标准参数
            self.part_num = 50
            self.tray_num = 10
            self.AGV_num = 4
            self.drone_num = 2
            self.vehicle_num = self.AGV_num + self.drone_num
            self.part_max_num = 50
            self.vehicle_max_num = 6
            self.tray_max_num = 10
            self.B_location_num = 5 # B加工区的数量
            self.C_location_num = 5 # C缓存区数量
            self.t_A_part_interval = 50 # 来料间隔时间
            self.t_B_processing = 600
            self.t_AB_base = 311
            self.t_AC_base = 201
            self.t_BC_base = 141
            self.drone_speed_factor = 0.6
            self.traffic_jam_prob = 0.2
            self.traffic_jam_factor_agv = 2.0
            self.traffic_jam_factor_drone = 1.2
            self.processing_failure_prob = 0.1
            self.battery_capacity = 100
            self.battery_consumption_per_distance = 0.05
            self.battery_critical_level = 20
            self.charging_time_base = 200
            self.charging_time_per_unit = 2
            self.charging_station = 'A'  # 充电站在A位置

    def reset(self):
        # 其他变量
        self.vehicle_action_buffer = {} # vehicle_action_buffer[time] = {'vehicle_index':第几辆车, 'node': 前往的地点, 'action':'要输出的内容'}

        # 上料口A信息
        self.A_part_info = np.zeros((self.part_max_num, 2), dtype=np.int32)
        for i in range(self.part_num):
            self.A_part_info[i,0] = 1
            self.A_part_info[i,1] = i*self.t_A_part_interval
        self.A_tray = np.zeros(self.tray_max_num, dtype=np.int32)
        for i in range(self.tray_num):
            self.A_tray[i] = 1

        # 加工区B信息
        self.B_info = np.zeros((self.B_location_num, 4), dtype=np.int32)
        # 第四列新增加工状态: 0=未加工, 1=正常加工中, 2=空转

        # 缓存区C信息
        self.C_info = np.zeros((self.C_location_num, 3), dtype=np.int32)

        # 车辆状态信息
        self.vehicle_info = np.zeros((self.vehicle_max_num, 5), dtype=np.int32)
        # 第五列为车辆类型: 0=AGV, 1=无人机
        for i in range(self.AGV_num):
            self.vehicle_info[i, 4] = 0  # AGV
        for i in range(self.AGV_num, self.vehicle_num):
            self.vehicle_info[i, 4] = 1  # 无人机
        
        # 车辆电池状态 (0-100)
        self.vehicle_battery = np.ones(self.vehicle_num) * self.battery_capacity
        
        self.vehicle_timer = np.zeros(self.vehicle_num, dtype=np.int32) # 用于计算每辆车的全局时间
        self.time_queue = []

    def reset_test(self):
        # 其他变量
        self.vehicle_action_buffer = {} # vehicle_action_buffer[time] = {'vehicle_index':第几辆车, 'node': 前往的地点, 'action':'要输出的内容'}

        # 上料口A信息
        self.A_part_info = np.zeros((self.part_max_num, 2), dtype=np.int32)
        for i in range(self.part_num):
            self.A_part_info[i,0] = 1
            self.A_part_info[i,1] = i*self.t_A_part_interval
        self.A_tray = np.zeros(self.tray_max_num, dtype=np.int32)
        for i in range(self.tray_num):
            self.A_tray[i] = 1

        # 加工区B信息
        self.B_info = np.zeros((self.B_location_num, 4), dtype=np.int32)
        # 第四列新增加工状态: 0=未加工, 1=正常加工中, 2=空转

        # 缓存区C信息
        self.C_info = np.zeros((self.C_location_num, 3), dtype=np.int32)

        # 车辆状态信息
        self.vehicle_info = np.zeros((self.vehicle_max_num, 5), dtype=np.int32)
        # 第五列为车辆类型: 0=AGV, 1=无人机
        for i in range(self.AGV_num):
            self.vehicle_info[i, 4] = 0  # AGV
        for i in range(self.AGV_num, self.vehicle_num):
            self.vehicle_info[i, 4] = 1  # 无人机
        
        # 车辆电池状态 (0-100)
        self.vehicle_battery = np.ones(self.vehicle_num) * self.battery_capacity
        
        self.vehicle_timer = np.zeros(self.vehicle_num, dtype=np.int32) # 用于计算每辆车的全局时间
        self.time_queue = []

    # 计算交通时间 (考虑车辆类型和拥堵情况)
    def calculate_travel_time(self, vehicle_index, start, end):
        # 基础时间
        if (start == 'A' and end == 'B') or (start == 'B' and end == 'A'):
            base_time = self.t_AB_base
        elif (start == 'A' and end == 'C') or (start == 'C' and end == 'A'):
            base_time = self.t_AC_base
        elif (start == 'B' and end == 'C') or (start == 'C' and end == 'B'):
            base_time = self.t_BC_base
        else:
            base_time = 0
        
        # 判断是否拥堵
        is_traffic_jam = np.random.random() < self.traffic_jam_prob
        
        # 计算实际时间
        if self.vehicle_info[vehicle_index, 4] == 1:  # 无人机
            # 无人机速度更快
            actual_time = base_time * self.drone_speed_factor
            # 无人机受拥堵影响小
            if is_traffic_jam:
                actual_time *= self.traffic_jam_factor_drone
        else:  # AGV
            actual_time = base_time
            # AGV受拥堵影响大
            if is_traffic_jam:
                actual_time *= self.traffic_jam_factor_agv
        
        # 消耗电池
        distance = base_time  # 用基础时间作为距离的代理
        battery_consumption = distance * self.battery_consumption_per_distance
        self.vehicle_battery[vehicle_index] -= battery_consumption
        
        # 确保电池电量不会低于0
        self.vehicle_battery[vehicle_index] = max(0, self.vehicle_battery[vehicle_index])
        
        return int(actual_time)

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
        # 车辆信息
        vehicle_vec = np.concatenate((self.vehicle_info.flatten(order='F'), self.vehicle_battery))
        return [A_vec, B_vec, C_vec, vehicle_vec]
    
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
    
    # 检查电池电量是否足够完成任务
    def check_battery_sufficient(self, vehicle_index, start, end):
        # 计算任务需要的电量
        if (start == 'A' and end == 'B') or (start == 'B' and end == 'A'):
            distance = self.t_AB_base
        elif (start == 'A' and end == 'C') or (start == 'C' and end == 'A'):
            distance = self.t_AC_base
        elif (start == 'B' and end == 'C') or (start == 'C' and end == 'B'):
            distance = self.t_BC_base
        else:
            distance = 0
        
        required_battery = distance * self.battery_consumption_per_distance
        return self.vehicle_battery[vehicle_index] >= required_battery
    
    # 检查是否需要充电
    def need_charging(self, vehicle_index):
        return self.vehicle_battery[vehicle_index] <= self.battery_critical_level
    
    # 充电过程
    def charge_vehicle(self, vehicle_index, start_location):
        reward = 0
        # 如果不在充电站，需要先移动到充电站
        if start_location != self.charging_station:
            # 计算到充电站的时间
            travel_time = self.calculate_travel_time(vehicle_index, start_location, self.charging_station)
            reward -= travel_time * self.time_fac
        else:
            travel_time = 0
        
        # 计算需要充电的量
        charge_needed = self.battery_capacity - self.vehicle_battery[vehicle_index]
        # 计算充电时间
        charging_time = self.charging_time_base + charge_needed * self.charging_time_per_unit
        
        # 更新车辆信息
        self.vehicle_info[vehicle_index, 0] = 1  # 位置设为充电站 (A)
        self.vehicle_info[vehicle_index, 3] = travel_time + charging_time
        self.vehicle_timer[vehicle_index] += self.vehicle_info[vehicle_index, 3]
        
        # 充满电
        self.vehicle_battery[vehicle_index] = self.battery_capacity
        
        # 更新reward
        reward -= charging_time * self.time_fac
        
        # 更新buffer
        corrected_time, self.time_queue = self.time_correction(self.time_queue, self.vehicle_timer[vehicle_index])
        vehicle_type = "Drone" if self.vehicle_info[vehicle_index, 4] == 1 else "AGV"
        self.vehicle_action_buffer[corrected_time] = {
            'vehicle_index': vehicle_index, 
            'node': self.charging_station, 
            'action': f'{vehicle_type} {vehicle_index} charged at {self.charging_station}',
            'action_num': 4
        }
        
        return reward
    
    # action: 0/2 从A拿物料和托盘前往B/C
    def take_part_from_A(self, vehicle_index, start_location, destination):
        reward = 0
        if start_location == 'A': extra_t = 0
        elif start_location == 'B': 
            extra_t = self.calculate_travel_time(vehicle_index, 'B', 'A')
        else: 
            extra_t = self.calculate_travel_time(vehicle_index, 'C', 'A')
        
        if destination == 'B': 
            travel = self.calculate_travel_time(vehicle_index, 'A', 'B')
        else: 
            travel = self.calculate_travel_time(vehicle_index, 'A', 'C')
        
        A_part_index, A_tray_index = self.return_nearest_A_part_tray_index()
        
        # 验证托盘索引是否在有效范围内
        if A_tray_index <= 0 or A_tray_index > self.tray_max_num:
            print(f"警告: 托盘索引 {A_tray_index} 超出有效范围，将使用第一个可用托盘")
            # 重新查找有效托盘
            for i in range(self.tray_max_num):
                if self.A_tray[i] == 1:
                    A_tray_index = i + 1
                    break
            # 如果所有托盘都不可用，则不能执行任务
            if A_tray_index <= 0 or A_tray_index > self.tray_max_num:
                print("错误: 没有可用托盘")
                return 0
        
        # 更新车辆信息
        if destination == 'B':
            self.vehicle_info[vehicle_index, 0] = 2
        else:
            self.vehicle_info[vehicle_index, 0] = 3
        self.vehicle_info[vehicle_index, 1] = A_part_index
        self.vehicle_info[vehicle_index, 2] = A_tray_index
        self.vehicle_info[vehicle_index, 3] = extra_t + max(self.A_part_info[A_part_index-1,1] - extra_t - self.vehicle_timer[vehicle_index], 0) + travel
        self.vehicle_timer[vehicle_index] += self.vehicle_info[vehicle_index, 3]
        
        # 更新上料口A的信息
        self.A_part_info[A_part_index-1,:] = 0
        self.A_tray[A_tray_index-1] = 0
        
        # 更新B/C的信息
        part_index = self.vehicle_info[vehicle_index, 1]
        tray_index = self.vehicle_info[vehicle_index, 2]
        
        if destination == 'B':
            # 更新加工区B信息
            B_index = self.return_nearest_available_pos('B')
            self.B_info[B_index, 0] = part_index
            self.B_info[B_index, 1] = tray_index
            
            # 确定是否加工空转
            if np.random.random() < self.processing_failure_prob:
                # 加工空转
                self.B_info[B_index, 3] = 2  # 标记为空转状态
                # 加工时间
                self.B_info[B_index, 2] = self.t_B_processing + self.vehicle_timer[vehicle_index]
            else:
                # 正常加工
                self.B_info[B_index, 3] = 1  # 标记为正常加工状态
                # 加工时间
                self.B_info[B_index, 2] = self.t_B_processing + self.vehicle_timer[vehicle_index]
        else:
            # 更新C信息，显示被预约
            C_index = self.return_nearest_available_pos('C')
            self.C_info[C_index, 2] = 1
        
        # 更新reward
        reward -= self.distance_fac*extra_t + self.time_fac*max(self.A_part_info[A_part_index-1,1] - extra_t - self.vehicle_timer[vehicle_index], 0)
        
        # 更新buffer
        corrected_time, self.time_queue = self.time_correction(self.time_queue, self.vehicle_timer[vehicle_index])
        vehicle_type = "Drone" if self.vehicle_info[vehicle_index, 4] == 1 else "AGV"
        
        if destination == 'B':
            self.vehicle_action_buffer[corrected_time] = {
                'vehicle_index': vehicle_index, 
                'node': destination, 
                'action': f'{vehicle_type} {vehicle_index} is going from {start_location} to {destination} with part {part_index} tray {tray_index}',
                'action_num': 0
            }
        else:
            self.vehicle_action_buffer[corrected_time] = {
                'vehicle_index': vehicle_index, 
                'node': destination, 
                'action': f'{vehicle_type} {vehicle_index} is going from {start_location} to {destination} with part {part_index} tray {tray_index}',
                'action_num': 2
            }
        
        return reward
    
    # 车辆把物料和托盘放在加工区B
    def put_part_in_B(self, vehicle_index):
        # 更新车辆信息（只保留点位信息）
        self.vehicle_info[vehicle_index, :4] = 0
        self.vehicle_info[vehicle_index, 0] = 2
    
    # 车辆把物料和托盘放在缓存区C
    def put_part_in_C(self, vehicle_index):
        part_index = self.vehicle_info[vehicle_index, 1]
        tray_index = self.vehicle_info[vehicle_index, 2]
        C_index = self.return_C_position_reserved()
        
        # 更新缓存区C的信息
        self.C_info[C_index, 0] = part_index
        self.C_info[C_index, 1] = tray_index
        self.C_info[C_index, 2] = 0  # 取消预约状态
        
        # 更新车辆信息
        self.vehicle_info[vehicle_index, :4] = 0
        self.vehicle_info[vehicle_index, 0] = 3

    # action: 1 从B点回收托盘前往A
    def recycle_tray_from_B_to_A(self, vehicle_index, start_location):
        reward = 0
        if start_location == 'B': 
            extra_t = 0
        elif start_location == 'C': 
            extra_t = self.calculate_travel_time(vehicle_index, 'C', 'B')
        else: 
            extra_t = self.calculate_travel_time(vehicle_index, 'A', 'B')
        
        tray_index = self.return_near_finished_part_pos_in_B()
        
        # 如果是空转的工件，不产生成品，只回收托盘
        is_idle_processing = (self.B_info[tray_index, 3] == 2)
        
        # 获取托盘编号并验证
        tray_id = self.B_info[tray_index, 1]
        if tray_id <= 0 or tray_id > self.tray_max_num:
            print(f"警告: B区域托盘索引 {tray_id} 超出有效范围，将被重置为1")
            tray_id = 1
            self.B_info[tray_index, 1] = tray_id
        
        # 更新车辆信息
        self.vehicle_info[vehicle_index, 0] = 1
        self.vehicle_info[vehicle_index, 1] = 0 if is_idle_processing else self.B_info[tray_index, 0]  # 空转时不携带零件
        self.vehicle_info[vehicle_index, 2] = tray_id  # 使用验证后的托盘编号
        
        # 计算到A的时间
        travel_to_A = self.calculate_travel_time(vehicle_index, 'B', 'A')
        self.vehicle_info[vehicle_index, 3] = extra_t + max(0, self.B_info[tray_index, 2]-extra_t-self.vehicle_timer[vehicle_index]) + travel_to_A
        
        # 更新reward
        reward -= self.distance_fac*extra_t + self.time_fac*max(0, self.B_info[tray_index, 2]-extra_t-self.vehicle_timer[vehicle_index])
        if is_idle_processing:
            reward -= 10  # 空转加工的惩罚
        
        self.vehicle_timer[vehicle_index] += self.vehicle_info[vehicle_index, 3]
        
        # 更新buffer
        corrected_time, self.time_queue = self.time_correction(self.time_queue, self.vehicle_timer[vehicle_index])
        vehicle_type = "Drone" if self.vehicle_info[vehicle_index, 4] == 1 else "AGV"
        
        part_info = "none" if is_idle_processing else self.B_info[tray_index, 0]
        self.vehicle_action_buffer[corrected_time] = {
            'vehicle_index': vehicle_index, 
            'node': 'A', 
            'action': f'{vehicle_type} {vehicle_index} is going from B to A with part {part_info} tray {self.B_info[tray_index, 1]}' + (" (idle processing)" if is_idle_processing else ""),
            'action_num': 1
        }
        
        # 更新B信息
        self.B_info[tray_index, :] = 0
        
        return reward
    
    # 在A点放置托盘
    def put_tray_in_A(self, vehicle_index):
        tray_index = self.vehicle_info[vehicle_index, 2]
        # 检查托盘索引是否有效
        if tray_index <= 0 or tray_index > self.tray_max_num:
            print(f"警告: 托盘索引 {tray_index} 超出有效范围 (1-{self.tray_max_num})，将被重置为第一个可用托盘")
            # 寻找第一个可用的托盘索引
            for i in range(self.tray_max_num):
                if self.A_tray[i] == 0:
                    tray_index = i + 1
                    break
            # 如果没有可用托盘，使用第一个索引
            if tray_index > self.tray_max_num:
                tray_index = 1
                
            # 更新车辆信息中的托盘索引
            self.vehicle_info[vehicle_index, 2] = tray_index
        
        # 更新A信息
        self.A_tray[tray_index-1] = 1
        # 更新车辆信息
        self.vehicle_info[vehicle_index, :4] = 0

    # action: 3 车辆把物料和托盘从C运往B
    def carry_partandtray_from_C_to_B(self, vehicle_index, start_location):
        reward = 0
        if start_location == 'C': 
            extra_t = 0
        elif start_location == 'B': 
            extra_t = self.calculate_travel_time(vehicle_index, 'B', 'C')
        else: 
            extra_t = self.calculate_travel_time(vehicle_index, 'A', 'C')
        
        C_index = self.return_available_part_in_C()
        part = self.C_info[C_index, 0]
        tray = self.C_info[C_index, 1]
        
        # 验证托盘索引是否有效
        if tray <= 0 or tray > self.tray_max_num:
            print(f"警告: C区域托盘索引 {tray} 超出有效范围，将被重置为默认值")
            # 使用一个有效的托盘索引
            tray = max(1, min(self.tray_max_num, tray))
        
        # 更新车辆信息
        self.vehicle_info[vehicle_index, 0] = 2
        self.vehicle_info[vehicle_index, 1] = part
        self.vehicle_info[vehicle_index, 2] = tray
        
        # 计算到B的时间
        travel_to_B = self.calculate_travel_time(vehicle_index, 'C', 'B')
        self.vehicle_info[vehicle_index, 3] = extra_t + travel_to_B
        self.vehicle_timer[vehicle_index] += self.vehicle_info[vehicle_index, 3]
        
        # 更新C信息
        self.C_info[C_index, :] = 0
        
        # 更新B信息
        B_index = self.return_nearest_available_pos('B')
        self.B_info[B_index, 0] = part
        self.B_info[B_index, 1] = tray
        
        # 确定是否加工空转
        if np.random.random() < self.processing_failure_prob:
            # 加工空转
            self.B_info[B_index, 3] = 2  # 标记为空转状态
            # 加工时间
            self.B_info[B_index, 2] = self.t_B_processing + self.vehicle_timer[vehicle_index]
        else:
            # 正常加工
            self.B_info[B_index, 3] = 1  # 标记为正常加工状态
            # 加工时间
            self.B_info[B_index, 2] = self.t_B_processing + self.vehicle_timer[vehicle_index]
        
        # 更新buffer
        corrected_time, self.time_queue = self.time_correction(self.time_queue, self.vehicle_timer[vehicle_index])
        vehicle_type = "Drone" if self.vehicle_info[vehicle_index, 4] == 1 else "AGV"
        
        self.vehicle_action_buffer[corrected_time] = {
            'vehicle_index': vehicle_index, 
            'node': 'B', 
            'action': f'{vehicle_type} {vehicle_index} is going from C to B with part {part} tray {tray}',
            'action_num': 3
        }
        
        # 更新reward
        reward -= extra_t*self.distance_fac
        return reward

    # 检查Action是否valid
    # 检查action0，A有物料，A有托盘，B有空位置
    def check_action0_valide(self, vehicle_index=None):
        # 基本条件验证
        if max(self.A_part_info[:,0]) == 1 and max(self.A_tray[:]) == 1 and min(self.B_info[:,0]) == 0:
            # 如果指定了车辆index，还需要验证电池是否足够
            if vehicle_index is not None:
                # 检查车辆当前位置，如果已经在B，则不应该执行A到B的动作
                if self.vehicle_info[vehicle_index, 0] == 2:  # 2 表示位置B
                    return False
                return self.check_battery_sufficient(vehicle_index, 'A', 'B')
            return True
        return False

    # 检查action1
    def check_action1_valide(self, vehicle_index=None):
        # 基本条件验证
        if max(self.B_info[:,0]) >= 1:
            # 如果指定了车辆index，还需要验证电池是否足够
            if vehicle_index is not None:
                # 检查车辆当前位置，如果已经在A，则不应该执行B到A的动作
                if self.vehicle_info[vehicle_index, 0] == 1:  # 1 表示位置A
                    return False
                return self.check_battery_sufficient(vehicle_index, 'B', 'A')
            return True
        return False

    # 检查action2
    def check_action2_valide(self, vehicle_index=None):
        cnt = 0
        if max(self.A_part_info[:,0]) == 1 and max(self.A_tray[:]) == 1:
            for i in range(self.C_location_num):
                if self.C_info[i,0] == 0 and self.C_info[i,2] == 0:
                    cnt += 1
        if cnt == 0:
            return False
        
        # 如果指定了车辆index，还需要验证电池是否足够
        if vehicle_index is not None:
            # 检查车辆当前位置，如果已经在C，则不应该执行A到C的动作
            if self.vehicle_info[vehicle_index, 0] == 3:  # 3 表示位置C
                return False
            return self.check_battery_sufficient(vehicle_index, 'A', 'C')
        return True

    # 检查action3
    def check_action3_valide(self, vehicle_index=None):
        # 基本条件验证
        if max(self.C_info[:,0]) >= 1 and min(self.B_info[:,0]) == 0:
            # 如果指定了车辆index，还需要验证电池是否足够
            if vehicle_index is not None:
                # 检查车辆当前位置，如果已经在B，则不应该执行C到B的动作
                if self.vehicle_info[vehicle_index, 0] == 2:  # 2 表示位置B
                    return False
                return self.check_battery_sufficient(vehicle_index, 'C', 'B')
            return True
        return False
    
    # 检查action4 (充电)
    def check_action4_valide(self, vehicle_index):
        return self.need_charging(vehicle_index)

    # 初始状态调度
    def simulation_initialize(self):
        self.reset()
        for i in range(self.vehicle_num):
            self.take_part_from_A(i, 'A', 'B')


if __name__ == "__main__": 
    solver = multiVehicleEnv()
    solver.reset()
    
    # 测试场景
    solver.take_part_from_A(0, 'A', 'B')  # AGV 0 从A拿货到B
    solver.take_part_from_A(1, 'A', 'C')  # AGV 1 从A拿货到C
    solver.take_part_from_A(4, 'A', 'B')  # Drone 0 从A拿货到B
    
    # 充电测试
    solver.charge_vehicle(2, 'A')  # AGV 2 在A充电
    
    # 从C到B测试
    solver.put_part_in_C(1)  # AGV 1 将货物放在C
    solver.carry_partandtray_from_C_to_B(5, 'C')  # Drone 1 从C拿货到B
    
    # 打印状态
    print("A_part_info:")
    print(solver.A_part_info[:10])
    print("A_tray:")
    print(solver.A_tray[:10])
    print("B_info:")
    print(solver.B_info)
    print("C_info:")
    print(solver.C_info)
    print("vehicle_info:")
    print(solver.vehicle_info[:6])
    print("vehicle_battery:")
    print(solver.vehicle_battery[:6])
    print("vehicle_timer:")
    print(solver.vehicle_timer[:6]) 