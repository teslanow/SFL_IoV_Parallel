class ServerBase:
    def __init__(self):
        super().__init__()
        # 单位B/s
        self.comm_bandwidth = None
        # 单位flop/s
        self.compute_density = None
        self.optimizer = None
    def aggregate(self):
        """

        Returns: aggregate使用的时间

        """
        raise NotImplementedError

    def set_compute_density(self, compute_density):
        """
        暂时先用计算速度Flops代替，后续应该使用更细的粒度
        Args:
            compute_density: unit in Flops
        """
        self.compute_density = compute_density

    def set_comm_bandwidth(self, comm_bandwidth):
        """
        Args:
            comm_bandwidth: unit in byte/s
        """
        self.comm_bandwidth = comm_bandwidth
