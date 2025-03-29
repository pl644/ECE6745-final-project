import torch

Device = "cuda" if torch.cuda.is_available() else "cpu"
Train_dir = "./data"  # Directory where MNIST will be downloaded
Test_dir = "./data"
Learning_rate = 1e-3
Batch_size = 64
Num_workers = 1
image_size = 28  # MNIST images are 28x28
Num_epochs = 10
Num_classes = 10  # 10 digits in MNIST
Load_model = False
Save_model = True