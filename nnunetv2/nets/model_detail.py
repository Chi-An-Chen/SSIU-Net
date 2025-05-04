import time
import torch
from thop import profile
from torchtools.utils import print_summary
from ptflops import get_model_complexity_info

from nnunetv2.nets.UMambaBot import UMambaBot
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.UMambaBot import get_umamba_bot_from_plans


from nnunetv2.nets.LightMUNet_replace import LightMUNet
# from nnunetv2.nets.LightMUNet_replace_mambaOut import LightMUNet
# from nnunetv2.nets.LightMUNet_replace_mambaRotate import LightMUNet
model = LightMUNet(
    spatial_dims=2,
    init_filters=32,
    in_channels=3,  # Fixed to match input tensor channels
    out_channels=2,
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    num_heads=[4, 8, 16, 32],
    patches_resolution=[320, 384]
).cuda()



def calculate_throughput(model, input_size=(4, 3, 320, 384), device='cuda', warmup=20, repeat=100):
    model = model.to(device)
    model.eval()
    dummy_input = torch.randn(*input_size).to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(repeat):
            _ = model(dummy_input)

    torch.cuda.synchronize()
    total_time = time.time() - start_time

    total_images = input_size[0] * repeat
    throughput = total_images / total_time
    print(f"Throughput: {throughput:.2f} images/sec")

    return throughput

# x = torch.randn(1, 3, 320, 384).cuda().cuda()
# print(model)
# print_summary(model, x)
# flops, params = profile(model, inputs=(x,))
# print(f"FLOPs: {flops / 1e9:.4f} GFLOPs")
# print(f"Params: {params / 1e6:.4f} M")
# macs, _ = get_model_complexity_info(model, (3, 320, 384), as_strings=True, print_per_layer_stat=False)
# print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
calculate_throughput(model)
