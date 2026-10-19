import torch
print(torch.cuda.is_available())  # 如果返回 True，则说明 CUDA 可用
print(torch.cuda.get_device_name(0))  # 显示第一个 GPU 的名称
