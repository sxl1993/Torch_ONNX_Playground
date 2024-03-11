import uuid
import torch
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

def worker_fn(t1, t2):
    return torch.add(t1, t2)

def main():
    t1 = torch.rand((3, 3), requires_grad=False)
    t2 = torch.rand((3, 3), requires_grad=False)

    config = LaunchConfig(
        min_nodes=1,
        max_nodes=4,
        nproc_per_node=2,
        run_id=str(uuid.uuid4()),
        role="trainer",
        rdzv_endpoint="localhost:29400",
        rdzv_backend="c10d",
        max_restarts=1,
        monitor_interval=1,
        start_method="spawn",
    )

    # breakpoint()
    outputs = elastic_launch(config, worker_fn)(t1, t2)
    print(outputs)

if __name__ == "__main__":
    main()