import torch
import torch.nn as nn
import thop
from torchsummary import summary
# from rs_mamba_ss import RSM_SS
# from utils.path_hyperparameter import ph
from geoseg.models.UVNet import UVNet
# from geoseg.models.SBHE import SBHE
# -------------------------------
# 初始化设备
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 初始化模型
# -------------------------------
# model = RSM_SS(
#     dims=ph.dims, 
#     depths=ph.depths, 
#     ssm_d_state=ph.ssm_d_state, 
#     ssm_dt_rank=ph.ssm_dt_rank,
#     ssm_ratio=ph.ssm_ratio, 
#     mlp_ratio=ph.mlp_ratio, 
#     downsample_version=ph.downsample_raito, 
#     patchembed_version=ph.patchembed_version
# ).to(device).eval()  # 评估模式
model = UVNet(67,2,True)  # <-- 先实例化

# model = SBHE(in_channels2=3, out_channels2=2, in_channels1=64, out_channels1=2)

model = model.to(device).eval()
# -------------------------------
# 创建输入张量
# -------------------------------
# 1. 原图输入
input_img = torch.randn(1, 3, 256, 256).to(device)
# 2. embed 输入
input_embed = torch.randn(1, 64, 256, 256).to(device)

# -------------------------------
# 计算模型参数
# -------------------------------
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

num_params = count_parameters(model)
print(f"Number of parameters: {num_params} ({num_params/1e6:.2f} M)")

# -------------------------------
# 计算 FLOPs
# -------------------------------
flops, _ = thop.profile(model, inputs=(input_img, input_embed))
print(f"FLOPs: {flops} ({flops/1e9:.2f} GFLOPs)")

# -------------------------------
# 用 torchsummary 打印网络信息
# -------------------------------
# summary 支持多输入，用列表传入 shape
summary(model, [(3, 256, 256), (64, 256, 256)])
