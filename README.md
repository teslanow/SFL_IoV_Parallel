# MergeSFL

## Run
```bash
mpiexec -n 1 python server.py : -n 10 python client.py
```

## Argments

`--data_pattern` (int) 

options: 
+ 0: IID
+ 1-4: non-IID
+ 5~7: Dirichlet($\alpha$, $n$). $\alpha \in (0, +\infty) $ is the distribution parameter, n is the number of virtual workers. The lower the value of $\alpha$, the greater the degree of skewedness. The file containing data partitions (.npy) is saved in ./data_partition/
    + 5: Dir(1.0, n)
    + 6: Dir(0.5, n)
    + 7: Dir(0.1, n)


`--dataset_type` (str) 

options: image100, CIFAR10, UCIHAR, SPEECH


`--epoch` (int) 

80 for UCIHAR, 500 for IMAGE100, 250 for CIFAR10 and SPEECH

`--worker_num` (int) 

the number of virtual workers, default 10


## Example
Run on UCIHAR with Dirichlet distribution parameterized by 0.1, across 10 clients for 80 epoch, with 10 clients being active each epoch
```bash
mpiexec -n 1 python server.py --data_pattern 7 --dataset_type UCIHAR --epoch 80 --worker_num 10 : -n 10 python client.py
```

