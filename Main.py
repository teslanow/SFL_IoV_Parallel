import os

# MergeSFL
# python_path = "/root/anaconda3/envs/fedscale/bin/python"
# datatype = "CIFAR10"
# data_path = "/data/zhongxiangwei/data/CIFAR10"
# data_pattern = 0
# expname = "MergeSFL-Test"
# server_python_file = f"MergeSFL/server.py"
# client_python_file = f"MergeSFL/client.py"

# SplitFed
python_path = "/root/anaconda3/envs/fedscale/bin/python"
datatype = "CIFAR10"
data_path = "/data/zhongxiangwei/data/CIFAR10"
data_pattern = 0
expname = "SplitFed"
server_python_file = f"SplitFed/server.py"
client_python_file = f"SplitFed/client.py"
local_epoch = 42
worker_num = 40
active_process_num = 30
# worker_num = 2
# active_process_num = 2

command_str = f'mpiexec -n 1 python {server_python_file} --local_epoch {local_epoch} --data_pattern {data_pattern} --dataset_type {datatype} --epoch 150 --worker_num {worker_num} --data_path {data_path} --expname {expname} : -n {active_process_num} python {client_python_file} --expname {expname}'
print(command_str)


os.system(command_str)