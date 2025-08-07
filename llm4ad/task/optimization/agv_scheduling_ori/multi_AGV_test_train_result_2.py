import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from multi_AGV_Env_1 import multiAGV_Env
from multi_AGV_network_2 import DQN
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Multi-AGV Testing with different selection methods')
parser.add_argument('--selection', type=int, default=1, choices=[0, 1, 2],
                    help='Selection method: 0=DQN, 1=Greedy, 2=Shortest Distance')
args = parser.parse_args()

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

device = torch.device("mps")

env = multiAGV_Env()
# Get number of actions from gym action space
n_actions = len(env.action_list)
# Get the number of state observations
env.reset_test()
state_list = env.return_state_info()

choice = args.selection
print(f"Using selection method: {['DQN', 'Greedy', 'Shortest Distance'][choice]}")

# Only load policy network if using DQN
policy_net = None
if choice == 0:
    policy_net = DQN(state_list, n_actions).to(device)
    policy_net.load_state_dict(torch.load('DQN_params_epoch_300_part_100_AGV_6_tray_10_withNormalization_new_2.pth', map_location=device))

steps_done = 0
not_valid_time = 0

def select_action(A_vec, B_vec, C_vec):
    global steps_done
    global not_valid_time
    sample = random.random()
    sample = 1000
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # 返回一个valid的且Q值最大的action
            valid_action_list = []
            if env.check_action0_valide():
                valid_action_list.append(policy_net(A_vec, B_vec, C_vec)[0][0].view(1, 1).item())
            else:
                valid_action_list.append(-5e10)
            if env.check_action1_valide():
                valid_action_list.append(policy_net(A_vec, B_vec, C_vec)[0][1].view(1, 1).item())
            else:
                valid_action_list.append(-5e10)
            if env.check_action2_valide():
                valid_action_list.append(policy_net(A_vec, B_vec, C_vec)[0][2].view(1, 1).item())
            else:
                valid_action_list.append(-5e10)
            if env.check_action3_valide():
                valid_action_list.append(policy_net(A_vec, B_vec, C_vec)[0][3].view(1, 1).item())
            else:
                valid_action_list.append(-5e10)
            if policy_net(A_vec, B_vec, C_vec).max(1)[0].view(1, 1).item() not in valid_action_list:
                not_valid_time += 1
            if max(valid_action_list) != -5e10:
                return valid_action_list.index(max(valid_action_list))
            else:
                return -1
    else:
        # 从valid的action里面随机采样一个action输出
        valid_action_list = []
        if env.check_action0_valide():
            valid_action_list.append(0)
        if env.check_action1_valide():
            valid_action_list.append(1)
        if env.check_action2_valide():
            valid_action_list.append(2)
        if env.check_action3_valide():
            valid_action_list.append(3)
        if len(valid_action_list) != 0:
            index = random.randint(0,len(valid_action_list)-1)
            return valid_action_list[index]
        else:
            return -1



final_reward = 0

env.reset_test()
state_list = env.return_state_info()
# print(state_list)
A_vec = torch.tensor(state_list[0], dtype=torch.float32, device=device).unsqueeze(0)
B_vec = torch.tensor(state_list[1], dtype=torch.float32, device=device).unsqueeze(0)
C_vec = torch.tensor(state_list[2], dtype=torch.float32, device=device).unsqueeze(0)
# AGV_vec = torch.tensor(state_list[3], dtype=torch.float32, device=device).unsqueeze(0)
terminated = False
for i in range(env.AGV_num):
    if choice == 0:
        action = select_action(A_vec, B_vec, C_vec)
    elif choice == 1:
        action = env.select_action_greedy(i, 'A')
    else:
        action = env.select_action_shortest_distance(i, 'A')
    # print(AGV_index, node, action)
    if action == 0: reward = env.take_part_from_A(i, 'A', 'B')
    elif action == 1: reward = env.recycle_tray_from_B_to_A(i, 'A')
    elif action == 2: reward = env.take_part_from_A(i, 'A', 'C')
    elif action == 3: reward = env.carry_partandtray_from_C_to_B(i, 'A')
final_reward += reward
state_list = env.return_state_info()
A_vec = torch.tensor(state_list[0], dtype=torch.float32, device=device).unsqueeze(0)
B_vec = torch.tensor(state_list[1], dtype=torch.float32, device=device).unsqueeze(0)
C_vec = torch.tensor(state_list[2], dtype=torch.float32, device=device).unsqueeze(0)
# AGV_vec = torch.tensor(state_list[3], dtype=torch.float32, device=device).unsqueeze(0)

start = time.time() 
for t in range(1000):
    # pop出时间序列中最终结束动作最短的
    env.time_queue.sort()
    times = env.time_queue.pop(0)
    # print(time)
    # print(env.agv_action_buffer)
    # print(env.B_info)
    message = env.agv_action_buffer[times]['action']
    print(message)
    node = env.agv_action_buffer[times]['node']
    AGV_index = env.agv_action_buffer[times]['AGV_index']
    action_num = env.agv_action_buffer[times]['action_num']
    if action_num == 0 or action_num == 3:
        env.put_part_in_B(AGV_index)
    elif action_num == 1:
        env.put_tray_in_A(AGV_index)
    elif action_num == 2:
        env.put_part_in_C(AGV_index)
    del env.agv_action_buffer[times]
    # 输出动作的AGV继续决策，放入buffer
    if choice == 0:
        action = select_action(A_vec, B_vec, C_vec)
    elif choice == 1:
        action = env.select_action_greedy(AGV_index, node)
    else:
        action = env.select_action_shortest_distance(AGV_index, node)
    # print(AGV_index, node, action)
    if action == 0: reward = env.take_part_from_A(AGV_index, node, 'B')
    elif action == 1: reward = env.recycle_tray_from_B_to_A(AGV_index, node)
    elif action == 2: reward = env.take_part_from_A(AGV_index, node, 'C')
    elif action == 3: reward = env.carry_partandtray_from_C_to_B(AGV_index, node)
    else: 
        terminated = True
    final_reward += reward
    reward = torch.tensor([reward], device=device)
    action = torch.tensor([[action]], device=device, dtype=torch.long)
    if terminated:
        # print("end")
        next_A_vec = None
        next_B_vec = None
        next_C_vec = None
        # next_AGV_vec = None
    else:
        next_state_list = env.return_state_info()
        next_A_vec = torch.tensor(next_state_list[0], dtype=torch.float32, device=device).unsqueeze(0)
        next_B_vec = torch.tensor(next_state_list[1], dtype=torch.float32, device=device).unsqueeze(0)
        next_C_vec = torch.tensor(next_state_list[2], dtype=torch.float32, device=device).unsqueeze(0)
        # next_AGV_vec = torch.tensor(next_state_list[3], dtype=torch.float32, device=device).unsqueeze(0)


    # Move to the next state
    A_vec = next_A_vec 
    B_vec = next_B_vec
    C_vec = next_C_vec
    # AGV_vec = next_AGV_vec

    if terminated:
        print(env.A_part_info)
        print(env.A_tray)
        print(env.B_info)
        print(env.C_info)
        print(env.AGV_info)
        print(env.AGV_timer)
        env.time_queue.sort()
        for item in env.time_queue:
            times = env.time_queue.pop(0)
            message = env.agv_action_buffer[times]['action']
            print(message)
        print(final_reward/env.part_num)
        print(env.AGV_timer)
        print(not_valid_time)
        break 
      
end = time.time()
elapsed_time = end - start
print("Elapsed time: {:.4f} seconds".format(elapsed_time))