import os
import sys

import torch

sys.path.insert(0, "/data2/hezm/klss/code/teacher")

import nr_cuda


def main():
    assert torch.cuda.is_available(), "CUDA is not available"
    assert torch.cuda.device_count() == 1, "expected exactly one visible GPU"
    print("gpu:", torch.cuda.get_device_name(0))
    print("arch:", os.environ.get("NR_CUDA_ARCH", "sm_86"))
    nr_cuda.lib()
    print("built:", nr_cuda._SO)


if __name__ == "__main__":
    main()
