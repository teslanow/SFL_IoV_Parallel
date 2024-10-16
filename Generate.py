device_id = 0
for data_pattern in [0, 6, 8]:
    for ratio in [0.2, 0.4, 0.6, 0.8, 1.0]:
        print(f"nohup python Test.py --data_pattern {data_pattern} --dataset_type CIFAR10 --epoch 140 --worker_num 60 --data_path /data/zhongxiangwei/data/CIFAR10 --expname MergeSFL_SEQ --active_num 30 --finish_ratio {ratio} --device cuda:{device_id % 8} 2>&1 | tee output.log &")
        device_id += 1