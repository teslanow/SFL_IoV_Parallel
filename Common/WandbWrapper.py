import wandb

def wandbInit(args):
    wandb.init(
        # set the wandb project where this run will be logged
        project=args.expname,

        # track hyperparameters and run metadata
        config={
            "lr": args.lr,
            "architecture": args.model_type,
            "batch_size": args.batch_size,
            "local_epoch" : args.local_epoch,
            "algorithm": args.expname,
            "worker_num":args.worker_num,
            "active_num":args.active_num,
            "dataset": args.dataset_type,
            "global_rounds": args.round,
            "data_pattern": args.data_pattern,
            "pretrained": args.pretrained
        }
    )

def wandbLogWrap(content):
    if wandb.run is not None:
        wandb.log(content)

def wandbFinishWrap():
    if wandb.run is not None:
        wandb.finish()
