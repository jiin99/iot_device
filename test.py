import time
from model import MyLSTM, MyCNN
import csv
import argparse
import dataloader
import matplotlib.pyplot as plt
from matplotlib import dates
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser(description='Fire detection')

parser.add_argument('--model', default='cnn', type=str, help='epoch (default: 50)')
parser.add_argument('--batch_size', default=2048, type=int, help='batch size (default: 128)')
parser.add_argument('--save_root', default='./test', type=str, help='save root')
parser.add_argument('--load_path', default='/nas/home/jiin9/iot_device/hanok/1011-lstm-pe-norm/multi', type=str, help='load path')
parser.add_argument('--model_name', default='model_e50_s60_p1_d4.pth', type=str, help='model name')
parser.add_argument('--seq_len', default=60, type=int, help='input size (default: 150)')
parser.add_argument('--pred_len', default=1, type=int, help='output size (default: 1)')
parser.add_argument('--col', default='TEMP', type=str, help='name of variable')
parser.add_argument('--scaler', default='percentile', type=str, help='type of scaler')
parser.add_argument('--hidden',default=64, type=int, help='hidden size (default: 128)')
parser.add_argument('--device', default=4, type=int, help='type of device')
parser.add_argument('--input_c', default=2, type=int, help='type of device')
parser.add_argument('--num_var', default=2, type=int, help='type of device')
parser.add_argument('--print_freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--testdir',default='/nas/datahub/iot_device/container', type=str, help='test data path')
parser.add_argument('--gpu_id', default='1', type=str, help='gpu number')
parser.add_argument('--multimodal', default=True, type=bool, help='multimodal or unimodal')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
args.load_path = args.load_path + '/' + args.model_name
criterion = nn.MSELoss().cuda()

def inverse(x, mini, maxi):
    output = mini + x * (maxi - mini)
    return output
def evaluate(loader, model, criterion, order=-1):
    model.eval()
    loss_list = []
    output_list = torch.tensor([]).cuda()
    with torch.no_grad():
        for i, (inputs, target) in enumerate(loader):
            inputs = inputs.float().cuda()
            target = target.float().cuda()
            if args.model == 'cnn':
                output = model(inputs)[:, order, :]
            elif args.model == 'lstm':
                if order == -1:
                    output = model(inputs).squeeze()
                else:
                    output = model(inputs)[:, order, :]
            target = target[:, order, :]
            output_list = torch.cat([output_list, output])
            loss = criterion(output, target)
            loss_list = np.append(loss_list, loss.squeeze().cpu().data.numpy())
            if i % args.print_freq == 0:
                print('{2} : [{0}/{1}]\t'.format(i, len(loader), loss.item()))
        output_dict = dict(zip(range(len(output_list)+int(args.pred_len)), [[] for i in range(len(output_list)+int(args.pred_len))]))
        count = 0
        for i in output_list:
            ind = 0
            if args.pred_len == 1:
                i = [i]
            for j in i:
                output_dict[count+ind].append(j.item())
                ind += 1
            count += 1
        total = [np.mean(i) for i in output_dict.values()]
    return total, loss_list

if args.device == 4:
    col_list = ['TEMP', 'HUMIDITY']#, 'SMOKE_DEN']
elif args.device == 5:
    col_list = ['TEMP', 'HUMIDITY']#, 'SMELL_DEN', 'DEN']

for i in range(len(col_list)):
    print(col_list[i])
    # if col_list[i] == args.col:
    if args.model == 'cnn':
        net = MyCNN(args).cuda()
    elif args.model == 'lstm':
        net = MyLSTM(args).cuda()
    test_dataset = dataloader.loader(args.testdir, args.device, col_list[i], args.seq_len, args.pred_len, loader_type='test', args=args)
    
    test_loader = DataLoader(test_dataset,
                                shuffle=False,
                                batch_size=args.batch_size,
                                pin_memory=False)
    state_dict = torch.load(args.load_path)
    net.load_state_dict(state_dict)
    if args.multimodal:
        test_output, test_loss = evaluate(test_loader, net, criterion, i)
    else:
        test_output, test_loss = evaluate(test_loader, net, criterion)
    test_input = test_dataset.data[args.seq_len+1:]
    test_input = pd.DataFrame(test_input)
    test_input['output'] = test_output[:-1]
    test_input.index = pd.to_datetime(test_input.index)
    test_input['output'] = inverse(test_input['output'], test_dataset.min_value[i], test_dataset.max_value[i])
    test_input[col_list[i]] = inverse(test_input[col_list[i]], test_dataset.min_value[i], test_dataset.max_value[i])
    
    sum_error = np.sum((test_input[col_list[i]] - test_input.output) ** 2)
    max_error_point = test_input.index[np.argmax((test_input[col_list[i]] - test_input.output) ** 2)]
    prediction_label = torch.nn.Sigmoid()(torch.from_numpy(((test_input[col_list[i]] - test_input.output) ** 2).values))
    true_label = torch.from_numpy(test_dataset.label[args.seq_len+1:].values)
    prediction_label[prediction_label < 0.92] = 0
    prediction_label[prediction_label >= 0.92] = 1
    right = torch.sum(true_label * prediction_label == 1)
    precision = right / torch.sum(prediction_label)
    recall = right / torch.sum(true_label)
    f1 = 2 * precision * recall/(precision + recall)
    
    print(f'f1_score: {f1}, precision: {precision}, recall: {recall}')
    plt.figure(figsize = (20,8))
    plt.plot_date(test_input.index.values, test_input[[col_list[i], 'output']])
    plt.title(f'==Device{args.device} : {col_list[i]}==', fontsize=10)
    plt.gca().xaxis.set_major_locator(dates.DayLocator())
    plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%m-%d-%H-%M'))
    plt.setp(plt.gca().xaxis.get_majorticklabels(), 'rotation', 90)
    plt.axvline(pd.Timestamp('2021-04-12 15:00:00'), color='black', linewidth=3)
    plt.axvline(pd.Timestamp('2021-04-12 15:30:00'), color='black', linewidth=3)
    plt.axvline(pd.Timestamp('2021-04-17 17:50:00'), color='black', linewidth=3)
    plt.axvline(pd.Timestamp('2021-04-24 15:00:00'), color='black', linewidth=3)
    plt.axvline(pd.Timestamp('2021-04-24 17:00:00'), color='black', linewidth=3)
    plt.axvline(max_error_point, color='red', linewidth=1)
    plt.grid()
    plt.legend(['True', 'Pred'])
    
    if not os.path.exists(args.save_root + '/multi' + '/%d' % args.device + '/' + col_list[i]):
        os.makedirs(args.save_root + '/multi' + '/%d' % args.device + '/' + col_list[i])
    if not os.path.exists(args.save_root + '/uni' + '/%d' % args.device + '/' + col_list[i]):
        os.makedirs(args.save_root + '/uni' + '/%d' % args.device + '/' + col_list[i])
        
    if args.multimodal:
        plt.savefig(args.save_root + '/multi' + '/%d' % args.device + '/' + col_list[i] + '/' + col_list[i] + str(args.seq_len) + str(args.pred_len) + 'e%.3f' % sum_error +'.png')
    else: plt.savefig(args.save_root + '/uni' + '/%d' % args.device + '/' + col_list[i] + '/' + col_list[i] + str(args.seq_len) + str(args.pred_len) + 'e%.3f' % sum_error +'.png')
    plt.close()
    
    max_error_point = test_input['2021-04-24'].index[np.argmax((test_input['2021-04-24'][col_list[i]] - test_input['2021-04-24'].output) ** 2)]
    prediction_label = torch.nn.Sigmoid()(torch.from_numpy(((test_input['2021-04-24'][col_list[i]] - test_input['2021-04-24'].output) ** 2).values))
    true_label = torch.from_numpy(test_dataset.label['2021-04-24'].values)
    prediction_label[prediction_label < 0.92] = 0
    prediction_label[prediction_label >= 0.92] = 1
    right = torch.sum(true_label * prediction_label == 1)
    precision = right / torch.sum(prediction_label)
    recall = right / torch.sum(true_label)
    f1 = 2 * precision * recall/(precision + recall)
    plt.figure(figsize = (20,8))
    plt.plot_date(test_input['2021-04-24'].index.values, test_input['2021-04-24'][[col_list[i], 'output']])
    plt.title(f'==Device{args.device} : {col_list[i]}==', fontsize=10)
    plt.gca().xaxis.set_major_locator(dates.HourLocator())
    plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%d-%H-%M'))
    plt.setp(plt.gca().xaxis.get_majorticklabels(), 'rotation', 90)
    plt.axvline(pd.Timestamp('2021-04-24 15:02:00'), color='black', linewidth=2)
    plt.axvline(pd.Timestamp('2021-04-24 15:20:00'), color='black', linewidth=2)
    plt.axvline(pd.Timestamp('2021-04-24 15:50:00'), color='black', linewidth=2)
    plt.axvline(pd.Timestamp('2021-04-24 16:30:00'), color='black', linewidth=2)
    plt.axvline(pd.Timestamp('2021-04-24 17:00:00'), color='black', linewidth=2)
    plt.axvline(max_error_point, color='red', linewidth=1)
    plt.grid()

    if args.multimodal:
        plt.savefig(args.save_root + '/multi' + '/%d' % args.device + '/' + col_list[i] + '/' + col_list[i] + str(args.seq_len) + str(args.pred_len) +'_0424.png')
    else: plt.savefig(args.save_root + '/uni' + '/%d' % args.device + '/' + col_list[i] + '/' + col_list[i] + str(args.seq_len) + str(args.pred_len) +'_0424.png')