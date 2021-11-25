
import dataloader
import argparse
import os
import time
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import MyLSTM
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser(description='Fire detection')

parser.add_argument('--epochs', default=30, type=int, help='epoch (default: 50)')
parser.add_argument('--batch_size', default=128, type=int, help='batch size (default: 512)')
parser.add_argument('--model', default='cnn', type=str, help='model type')
parser.add_argument('--save_root', default='./container/lstm/', type=str, help='save root')
parser.add_argument('--hidden', default=256, type=int, help='hidden size (default: 128)')
parser.add_argument('--print_freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seq_len', default=150, type=int, help='input size (default: 150)')
parser.add_argument('--pred_len', default=1, type=int, help='output size (default: 1)')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--col', default='TEMP', type=str, help='name of variable')
parser.add_argument('--scaler', default='percentile', type=str, help='type of scaler')
parser.add_argument('--device', default=4, type=int, help='type of devices')
parser.add_argument('--traindir', default='/nas/datahub/iot_device/container', type=str, help='train data path')
parser.add_argument('--testdir', default='/nas/datahub/iot_device/container', type=str, help='test data path')
parser.add_argument('--gpu_id', default='2', type=str, help='gpu id')
parser.add_argument('--multimodal', default=True, type=bool, help='multimodal or unimodal')

args = parser.parse_args()

def train(loader, net, criterion, optimizer, epoch, args):
    loss_list = []
    max_norm = 1
    net.train()
    for i, (input, target) in enumerate(loader):
        # measure data loading time
        # data_time.update(time.time() - end)
        args.batch = input.shape[0]
        input = input.float().cuda()
        target = target.float().cuda()
        output = net(input)
        loss = criterion(output, target)
        loss_list.append(loss.item())
        wandb.log({'train_batch_loss':loss.item()})
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm)
        optimizer.step()
        # measure elapsed time
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss ({3})'.format(
                    epoch, i, len(loader), np.mean(loss_list)))
    return epoch, np.mean(loss_list)
    
def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    file_list = sorted(os.listdir(args.traindir))
    if args.device == 4:
        col_list = ['TEMP', 'HUMIDITY', 'SMOKE_DEN']
        args.num_var = len(col_list)
    elif args.device == 5:
        col_list = ['TEMP', 'HUMIDITY', 'SMELL_DEN', 'DEN']
        args.num_var = len(col_list)
    
    if not args.multimodal:
        args.num_var = 1
        args.input_c = 1
    else :
        col_list = [['TEMP','HUMIDITY']]
        args.num_var = 2
        args.input_c = 2
    
    for i in col_list:
        sweep_defaults = {
            "save_root" : args.save_root,
            "device" : args.device,
            "dataset" : col_list,
            "learning_rate": args.lr,
            "batch_size" : args.batch_size,
            "epochs" : args.epochs,
            "pred_len" : args.pred_len,
            "seq_len" : args.seq_len,
            "hidden_dim" : args.hidden
        }
        train_dataset = dataloader.loader(args.traindir, args.device, i, args.seq_len, args.pred_len, args=args)
        print(i)
        if args.multimodal:
            save_path = os.path.join(args.save_root, 'multi')
            run_name = str(sweep_defaults['device']) + str(sweep_defaults['dataset'])
            project_name = 'Time_Series_multimodal'
        else:
            save_path = os.path.join(args.save_root, 'uni/' + str(i))
            run_name = str(sweep_defaults['device']) + str(i)
            project_name = 'Time_Series'

        args.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        train_loader = DataLoader(train_dataset,
                                    shuffle=False,
                                    batch_size=args.batch_size,
                                    pin_memory=False)
        if args.model == 'cnn':
            net = MyCNN(args).cuda()
        elif args.model == 'lstm':
            net = MyLSTM(args).cuda()
        elif args.mode == 'lstm_em' : 
            net = MyLSTM_em(args).cuda()
        criterion = nn.MSELoss().cuda()
        optimizer = optim.Adam(net.parameters(),lr=args.lr)
        # Start Train
        wandb.init(config=sweep_defaults, project='Time_Series', entity='join', name=run_name, reinit=True)
        for epoch in range(1, args.epochs + 1):
            epoch, loss = train(train_loader,
                                net,
                                criterion,
                                optimizer,
                                epoch,
                                args)
        torch.save(net.state_dict(),
            os.path.join(save_path, f'model_e{int(args.epochs)}_s{int(args.seq_len)}_p{int(args.pred_len)}_d{int(args.device)}.pth'))
if __name__ == "__main__":
    main()