task_description = """
# AGV与无人机混合调度问题

## 目标
设计一个高效的算法来调度AGV和无人机，处理所有零件，同时最小化总完成时间。分数 = 成功实例的平均完成时间的负值（越高越好）。

## 环境设置
- 3个位置: A (上料区), B (加工区), C (缓存区)
- AGV与无人机负责运输零件和托盘
- 零件从A到达，需要在B进行加工，托盘返回A
- 所有车辆需要管理电池电量

## 关键参数
- part_num = 50, tray_num = 10, AGV_num = 4, drone_num = 2
- B_location_num = 5, C_location_num = 5
- t_B_processing = 600 (加工时间)
- 基础行驶时间: t_AB_base = 311, t_AC_base = 201, t_BC_base = 141
- t_A_part_interval = 50 (零件到达间隔)
- 无人机速度因子: drone_speed_factor = 0.6 (无人机速度为AGV的1.67倍)
- 交通拥堵发生概率: traffic_jam_prob = 0.2
- AGV拥堵影响系数: traffic_jam_factor_agv = 2.0
- 无人机拥堵影响系数: traffic_jam_factor_drone = 1.2
- 加工空转概率: processing_failure_prob = 0.1
- 电池容量: battery_capacity = 100
- 每单位距离消耗电量: battery_consumption_per_distance = 0.05
- 临界电量: battery_critical_level = 20
- 充电站位于A点

## 状态信息
- A_part_info[i,0]: 1表示零件存在，0表示不存在
- A_part_info[i,1]: 零件到达时间
- A_tray[i]: 1表示托盘可用，0表示不可用
- B_info[i,0]: 零件ID (0表示无)
- B_info[i,1]: 托盘ID (0表示无)
- B_info[i,2]: 加工结束时间
- B_info[i,3]: 加工状态 (0=未加工, 1=正常加工, 2=空转)
- C_info[i,0]: 零件ID (0表示无)
- C_info[i,1]: 托盘ID (0表示无)
- C_info[i,2]: 1表示位置已预约，0表示未预约
- vehicle_info[i,0]: 位置 (1=A, 2=B, 3=C)
- vehicle_info[i,1]: 携带的零件ID (0表示无)
- vehicle_info[i,2]: 携带的托盘ID (0表示无)
- vehicle_info[i,3]: 完成当前动作的时间
- vehicle_info[i,4]: 车辆类型 (0=AGV, 1=无人机)
- vehicle_battery[i]: 车辆电池电量 (0-100)
- vehicle_timer[i]: 每个车辆的当前时间

## 可用动作
- 0: 从A取零件和托盘运往B
- 1: 从B回收托盘运往A
- 2: 从A取零件和托盘运往C
- 3: 从C取零件和托盘运往B
- 4: 前往A充电
- 返回-1表示无可用动作

## 验证函数
- check_action0_valide(vehicle_index): A有零件，A有托盘，B有空间，电池足够
- check_action1_valide(vehicle_index): B有完成的零件/空托盘，电池足够
- check_action2_valide(vehicle_index): A有零件，A有托盘，C有未预约空间，电池足够
- check_action3_valide(vehicle_index): C有零件，B有空间，电池足够
- check_action4_valide(vehicle_index): 是否需要充电

## 辅助函数
- return_nearest_A_part_tray_index(): 获取A处最近的零件/托盘索引
- return_nearest_available_pos(position): 获取B/C处最近的可用位置
- return_near_finished_part_pos_in_B(): 获取B中结束时间最早的零件索引
- check_battery_sufficient(vehicle_index, start, end): 检查电池是否足够完成任务
- need_charging(vehicle_index): 检查是否需要充电
- calculate_travel_time(vehicle_index, start, end): 计算考虑交通拥堵后的实际行驶时间

## 关键考虑因素
- 防止死锁: 确保托盘回收到A
- 平衡缓存区C的使用
- 所有零件必须被处理才能成功
- 处理加工空转: 空转的零件不会产生成品，只需回收托盘
- 电池管理: 需要适时充电，防止电量耗尽
- 合理分配AGV和无人机: 无人机速度快且拥堵影响小，适合远距离和拥堵路段
- 完成时间 = 所有车辆的最大时间
"""

template_program = '''
import numpy as np

def select_next_action(env, vehicle_index, current_node):
    """
    设计一个创新的算法来为给定的车辆选择下一个动作。
    
    参数:
        env: 包含当前状态的车辆环境
        vehicle_index: 要调度的车辆索引
        current_node: 车辆当前位置 ('A', 'B', 或 'C')
        
    返回:
        动作索引 (0, 1, 2, 3, 4) 或 -1 表示无可用动作
        
        动作 0: 从A取零件和托盘运往B
        动作 1: 从B回收托盘运往A
        动作 2: 从A取零件和托盘运往C
        动作 3: 从C取零件和托盘运往B
        动作 4: 前往A充电
    """
    # 首先检查是否需要充电
    if env.check_action4_valide(vehicle_index):
        return 4
        
    # 判断车辆类型
    is_drone = (env.vehicle_info[vehicle_index, 4] == 1)
    
    # 基础实现
    if env.check_action0_valide(vehicle_index):
        return 0
    elif env.check_action1_valide(vehicle_index):
        return 1
    elif env.check_action2_valide(vehicle_index):
        return 2
    elif env.check_action3_valide(vehicle_index):
        return 3
    else:
        return -1
''' 