import torch
import thop
from torchsummary import summary
from STTNet import STTNet

model_infos = {
    # vgg16_bn, resnet50, resnet18
    'backbone': 'resnet50',
    'pretrained': False,
    'out_keys': ['block1','block2','block3','block4'],
    'in_channel': 3,
    'n_classes': 2,
    'top_k_s': 64,
    'top_k_c': 16,
    'encoder_pos': True,
    'decoder_pos': True,
    'model_pattern': ['X', 'A', 'S', 'C'],

    'BATCH_SIZE': 3,
    'IS_SHUFFLE': True,
    'NUM_WORKERS': 0,
    'DATASET': 'generate_dep_info/train_data.csv',
    'model_path': 'Checkpoints',
    'log_path': 'Results',
    # if you need the validation process.
    'IS_VAL': True,
    'VAL_BATCH_SIZE': 3,
    'VAL_DATASET': 'generate_dep_info/val_data.csv',
    # if you need the test process.
    'IS_TEST': True,
    'TEST_DATASET': 'generate_dep_info/test_data.csv',
    'IMG_SIZE': [512, 512],
    'PHASE': 'seg',

    # China Dataset
    'PRIOR_MEAN': [0.37618707362175813, 0.38140236772409364, 0.3907937124297842],
    'PRIOR_STD': [0.0332261068138814, 0.030094983534503284, 0.028014925652876384],
    # # # WHU Dataset
    # 'PRIOR_MEAN': [0.4352682576428411, 0.44523221318154493, 0.41307610541534784],
    # 'PRIOR_STD': [0.026973196780331585, 0.026424642808887323, 0.02791246590291434],

    # if you want to load state dict
    'load_checkpoint_path': '',
    # if you want to resume a checkpoint
    'resume_checkpoint_path': '',
}

def count_parameters_and_display(model):
    total_params = 0
    print("Parameters in each layer/block:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_params = param.numel()
            print(f"{name}: {layer_params} parameters")
            total_params += layer_params
    return total_params

def calculate_flops(model, input_data):
    flops, _ = thop.profile(model, inputs=(input_data,))
    return flops

# Example usage

model = STTNet(**model_infos)

model.eval()  # Ensure the model is in evaluation mode
input_data = torch.randn(1, 3, 512, 512)  # Create a random input tensor
print("Number of parameters:", count_parameters_and_display(model))
print("FLOPs:", calculate_flops(model, input_data))
