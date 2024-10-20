def conventional_FL():
    """
    经典FL方式，client先完成本地的训练，然后上传参数/梯度，server聚合梯度，然后broadcast
    Returns:

    """
    # server发送parameters
    # client进行locally training
    # 结束所有client的训练后，发送gradient/parameter到server