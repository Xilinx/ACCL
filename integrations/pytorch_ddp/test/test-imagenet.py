import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import torch.distributed as dist
import accl_process_group as accl

from mpi4py.MPI import COMM_WORLD as mpi
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import os
import sys
import logging
from PIL import Image
from tempfile import TemporaryDirectory

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

logger = logging.getLogger(__name__)

if "ACCL_DEBUG" in os.environ and os.environ["ACCL_DEBUG"]=="1":
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.WARNING)

# Run via ACCL

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualization

def train(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model


    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-n", type=int, default=1)
    parser.add_argument("-d", type=bool, default=None)


    parser.add_argument('-s', '--simulator', action='store_true',
                        default=False, help='Use simulation instead of '
                                            'hardware')
    parser.add_argument('-c', '--comms', choices=['udp', 'tcp', 'cyt_rdma'], default='tcp',
                        help='Run tests over specified communication backend')
    parser.add_argument('-i', '--host-file', type=str, help='Specify the file, where the host IPs are listed')
    parser.add_argument('-f', '--fpga-file', type=str, help='Specify the file, where the FPGA IPs are listed')
    parser.add_argument('-a','--master-address', type=str)
    parser.add_argument('-p','--master-port', type=str)


    args = parser.parse_args()

    if args.n == 1 and args.d == None :
        print("only one machine specified. Assuming Non distributed setup")
        args.d = False
    elif args.n > 1 and args.d == None:
        print("Assung DDP setup")
        args.d = True

    
    global rank, size
    if args.master_address==None:
        args.master_address = "localhost"
    if args.master_port==None:
        args.master_port = "30505"
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port
    rank = mpi.Get_rank()
    size = mpi.Get_size()

    host_file = args.host_file
    fpga_file = args.fpga_file
    comms = args.comms
    start_port = 5005

    rxbufsize = 4096 * 1024

    if args.d:
        if not args.simulator:
            #default from test.cpp
            rxbufsize = 4096 * 1024
            if host_file==None or fpga_file==None: sys.exit('Host and FPGA file need to be specified in hardware mode')
        
            with open(host_file, 'r') as hf:
                host_ips = hf.read().splitlines()
            
            with open(fpga_file, 'r') as ff:
                fpga_ips = ff.read().splitlines()

            if comms == "cyt_rdma":
                ranks = [accl.Rank(a, start_port, i, rxbufsize) for i, a in enumerate(fpga_ips)]
            else:
                ranks = [accl.Rank(a, start_port + i, 0, rxbufsize) for i, a in enumerate(fpga_ips)]
        else:
            rxbufsize = 4096 * 1024
            ranks = [accl.Rank("127.0.0.1", 5500 + i, i, rxbufsize) for i in range(size)]

        logger.debug(f'Ranks: {ranks}')

        if args.comms == 'udp':
            design = accl.ACCLDesign.udp
        elif args.comms == 'tcp':
            design = accl.ACCLDesign.tcp
        elif args.comms == 'cyt_rdma': # and not simulator:
            design = accl.ACCLDesign.cyt_rdma
    

        mpi.Barrier()            
    
        accl.create_process_group(ranks, design, bufsize=rxbufsize, initialize=True, simulation=args.simulator)
        dist.init_process_group("ACCL", rank=rank, world_size=size)
        
    device = 'cpu'


    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'imagenet-data/hymenoptera_data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}

    if args.d : sampler = DistributedSampler
    else : sampler = lambda x : None

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=False, num_workers=4, sampler=sampler(image_datasets[x]))
              for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    model_ft = models.resnet50(weights='IMAGENET1K_V1')
    
    num_ftrs = model_ft.fc.in_features

    model_ft.fc = nn.Linear(num_ftrs, 2)

    if args.d : model_ft = DDP(model_ft, bucket_cap_mb=4)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    loss_func = nn.CrossEntropyLoss()

    model_ft = train(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)


