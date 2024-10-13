import argparse
import asyncio
import torch
import torch.optim as optim
from config import ClientConfig, CommonConfig
from comm_utils import *
import datasets, models
from mpi4py import MPI
import logging
from utils import *

parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--visible_cuda', type=str, default='-1')
parser.add_argument('--use_cuda', action="store_false", default=True)
parser.add_argument('--expname', type=str, default='MergeSFL')

args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
csize = comm.Get_size()

device = torch.device("cuda:%d" % (rank % 7 + 1) if args.use_cuda and torch.cuda.is_available() else "cpu")


logger = get_client_logger(args, rank)
# end logger

MASTER_RANK=0
async def get_init_config(comm, MASTER_RANK, config):
    logger.info("before init")
    config_received = await get_data(comm, MASTER_RANK, 1)
    logger.info("after init")
    for k, v in config_received.__dict__.items():
        setattr(config, k, v)

def main():
    logger.info("client_rank:{}".format(rank))
    client_config = ClientConfig(
        common_config=CommonConfig()
    )

    logger.info("start")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []
    task = asyncio.ensure_future(get_init_config(comm, MASTER_RANK, client_config))
    tasks.append(task)
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()

    common_config = CommonConfig()
    common_config.model_type = client_config.common_config.model_type
    common_config.dataset_type = client_config.common_config.dataset_type
    common_config.batch_size = client_config.common_config.batch_size
    common_config.data_pattern=client_config.common_config.data_pattern
    common_config.lr = client_config.common_config.lr
    common_config.decay_rate = client_config.common_config.decay_rate
    common_config.min_lr=client_config.common_config.min_lr
    common_config.epoch = client_config.common_config.epoch
    common_config.momentum = client_config.common_config.momentum
    common_config.weight_decay = client_config.common_config.weight_decay
    common_config.data_path = client_config.common_config.data_path
    common_config.para = client_config.para
    common_config.labels = client_config.common_config.labels
    
    common_config.tag = 1
    # init config
    logger.info(str(common_config.__dict__))

    # logger.info(str(len(client_config.train_data_idxes)))
    
    train_dataset, test_dataset = datasets.load_datasets(common_config.dataset_type, common_config.data_path)
    
    # train_loader = datasets.create_dataloaders(train_dataset, batch_size=common_config.batch_size, selected_idxs=client_config.train_data_idxes)
    # test_loader = datasets.create_dataloaders(test_dataset, batch_size=16, shuffle=False)
    test_loader = None
    while True:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = []
        tasks.append(
            asyncio.ensure_future(
                local_training(comm, common_config, train_dataset, test_loader, common_config.labels)
            )
        )
        loop.run_until_complete(asyncio.wait(tasks))

        loop.close()
        if common_config.tag == common_config.epoch+1:
            break

async def local_training(comm, common_config, train_dataset, test_loader, labels):
    # lr
    epoch_lr = common_config.lr
    if common_config.tag > 1:
        epoch_lr = max((common_config.decay_rate * epoch_lr, common_config.min_lr))
        common_config.lr = epoch_lr

    # get configuration
    vid, bsz = await get_data(comm, MASTER_RANK, common_config.tag)

    if vid < 0:
        common_config.tag = common_config.tag+1
        logger.info("get end")
        return

    # get data_idxes
    train_data_idxes = await get_data(comm, MASTER_RANK, common_config.tag)
    
    if labels:
        train_loader = datasets.create_dataloaders(train_dataset, batch_size=int(bsz), selected_idxs=train_data_idxes, pin_memory=False, drop_last=True, collate_fn=lambda x: datasets.collate_fn(x, labels))
    else:
        train_loader = datasets.create_dataloaders(train_dataset, batch_size=int(bsz), selected_idxs=train_data_idxes, pin_memory=False, drop_last=True)

    logger.info("vid({}) epoch-{} lr: {} bsz: {} ".format(vid, common_config.tag, epoch_lr, bsz))

    local_model, _ = models.create_model_instance_SL(common_config.dataset_type, common_config.model_type)

    # download model
    local_para = await get_data(comm, MASTER_RANK, common_config.tag)
    # torch.nn.utils.vector_to_parameters(local_para, local_model.parameters())
    local_model.load_state_dict(local_para)
    local_model.to(device)

    if common_config.momentum < 0:
        optimizer = optim.SGD(local_model.parameters(), lr=epoch_lr, weight_decay=common_config.weight_decay)
    else:
        optimizer = optim.SGD(local_model.parameters(), momentum=common_config.momentum, nesterov=True, lr=epoch_lr, weight_decay=common_config.weight_decay)

    # train
    local_steps = 42
    local_model.train()
    for iter_idx in range(local_steps):
        inputs, targets = next(train_loader)

        inputs = inputs.to(device)
        smashed_data = local_model(inputs)

        send_smash = smashed_data.detach()
        data = (send_smash, targets)
        await send_data(comm, data, MASTER_RANK, common_config.tag)
        
        grad_in = await get_data(comm, MASTER_RANK, common_config.tag) 

        optimizer.zero_grad()
        smashed_data.backward(grad_in.to(device))
        optimizer.step()

    # upload model
    # local_para = torch.nn.utils.parameters_to_vector(local_model.parameters()).detach() 
    local_para = local_model.state_dict()
    await send_data(comm, local_para, MASTER_RANK, common_config.tag)

    # del local_model
    # del local_para
    # del optimizer
    # del smashed_data
    # del grad_in
    # del inputs

    # with torch.cuda.device("cuda:%d" % (gpu_id % 7 + 1)):
        # torch.cuda.empty_cache()
    #     logger.info("gpu allocated: {}".format(torch.cuda.memory_allocated()))

    # common_config.para = local_para
    common_config.tag = common_config.tag+1
    logger.info("get end")


if __name__ == '__main__':
    main()
