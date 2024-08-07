import argparse
import os
import torch.nn as nn
import torch
import datetime
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def plot_trajectory_3d(trajectory_1, trajectory_2):
        x = trajectory_1[:, 0]
        y = trajectory_1[:, 1]
        z = trajectory_1[:, 2]
        
        x_2 = trajectory_2[:, 0]
        y_2 = trajectory_2[:, 1]
        z_2 = trajectory_2[:, 2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, 'b-')
        ax.plot(x_2, y_2, z_2, 'r-')
        ax.scatter(x, y, z, c='b', marker='o')
        ax.scatter(x_2, y_2, z_2, c='r', marker='*')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Trajectory')
        plt.show()    
        #plt.savefig('test.png')


if __name__ == '__main__':
    torch.manual_seed(888)
    torch.cuda.manual_seed_all(888)
    np.random.seed(888)
    current_datetime = datetime.datetime.now()
    datetime_string = current_datetime.strftime('%Y-%m-%d_%H-%M-%S')
    parser = argparse.ArgumentParser()
    ### Model Classification
    parser.add_argument('--task', type=str, default='tracking', help='The main task') # classification | tracking | osr
    parser.add_argument('--weight', default=None, help='The inital weigth path')
    parser.add_argument('--store_dir', default=None, help='The inital weigth path')
    parser.add_argument('--cfg_model', type=str, help='The model configuration', default=None)
    parser.add_argument('--model_name', type=str, help='The model name', default='resnet_18')  # resnet_18 | resnet_18_ABN | classifier_32_ABN
    parser.add_argument('--in_channel', type=int, default=3, help="The number of input channel") # mnist: 1, cifar10: 3, stft: 3
    parser.add_argument('--hidden_channel', type=int, default=128, help="The number of hidden channel") # 64 128
    parser.add_argument('--emb_channel', type=int, default=64, help="The number of embedding channel")

    ## Model Tracking
    parser.add_argument('--model_architecture', type=str, default="ED", help="The model architecture") # ED  ED_con
    parser.add_argument('--central_block', type=str, default="gru", help="The central block")  #rnn  gru  lstm
    parser.add_argument('--num_layers', type=int, default=3, help="The number of layer")
    parser.add_argument('--activate', type=str, default="leaky", help="The number of layer")
    parser.add_argument('--time_sampling', type=str, default="regular", help="The time sampling assumption")
    parser.add_argument('--in_seq_len', type=int, default=10, help="The number of input data")
    parser.add_argument('--out_seq_len', type=int, default=20, help="The number of out data") # 10 20
    parser.add_argument('--out_seq_total_len', type=int, default=20, help="The number of out data") # 10 20
    parser.add_argument('--window_size', type=int, default=5, help="The number of out data") # None 5
    ### Data 
    parser.add_argument('--data_name', type=str, help='The data configuration', default="trajectory") # trajectory_3d | trajectory_2d | stft | trajair | mnist | cifar10
    parser.add_argument('--num_classes', type=int, default=4, help="The number of classes") ###
    parser.add_argument('--image_size', type=int, default=224, help='The image size') # stft: 224 | mnist: 28 | cifar: 32
    parser.add_argument('--train_percent', type=float, default=0.75, help='The percentage of the training set.')
    parser.add_argument('--offset', type=int, default=8, help="Offset")
    parser.add_argument('--known_class', type=list, default=[0, 1, 2, 3, 4, 5, 6])

    parser.add_argument('--train_dir', type=str, help='The train dir', default="data/p70_v5_a0.5_n10_t50.npy") # d2_p80_v2_a0.1_n10000_t50.npy
    parser.add_argument('--val_dir', type=str, help='The val dir', default="data/p70_v5_a0.5_n10_t50.npy")
    parser.add_argument('--test_dir', type=str, help='The test dir', default="data/p70_v5_a0.5_n10_t50.npy")

    ### Learning phases
    parser.add_argument('--mode', type=str, default="train", help="train/test process") # test | train
    parser.add_argument('--epochs', type=int, default=200, help='The number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='The batch size')
    parser.add_argument('--device', default='cpu', help='cpu/gpu')
    parser.add_argument('--optimizer', type=str, default="adam", help="Optimizer")
    parser.add_argument('--update_lr', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--criterion', type=str, default="RMSE", help="loss_function") # MSE_MAE | CrossEntropy | ARPL | RMSE | RMSE_Direction
    ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = parser.parse_args()
    if opt.store_dir is None:
        if opt.task == "tracking":
            store_dir = f"experiments/{opt.task}/{opt.data_name}/{opt.model_architecture}_{opt.central_block}_l{opt.num_layers}_w{opt.window_size}_h{opt.hidden_channel}/{datetime_string}"
        os.makedirs(store_dir, exist_ok=True)
    else:
        store_dir = opt.store_dir
    
    ####################################################################### 1) Model #######################################################################
    if opt.task == "classification":
        from models.model_classification import ModelClassification
        # (self, model_name=None, in_channel=1, hidden_channel=128, num_classes=11, weight=None, img_size=112)
        model = ModelClassification(model_name=opt.model_name, in_channel=opt.in_channel, hidden_channel=opt.hidden_channel, emb_channel=opt.emb_channel,
         num_classes=opt.num_classes, weight=opt.weight)
        params_to_update = model.parameters() ############# model params to update ##################
        num_params = sum(p.numel() for p in model.parameters())
        model = model.to(device)
    
    elif opt.task == "tracking":
        ## Time sampling
        if opt.time_sampling == "regular":
            in_time_steps = np.arange(0, opt.in_seq_len)
            out_time_steps = np.arange(opt.in_seq_len, opt.in_seq_len + opt.out_seq_len)
        ## Model tracking
        if opt.model_architecture == "ED":
            from models.model_tracking_ED import ED, ED_conf, Encoder, Decoder
            E_nets, D_nets = ED_conf(input_channel=opt.in_channel, hidden_channel=opt.hidden_channel, num_layer=opt.num_layers, block=opt.central_block, activate=opt.activate)
            encoder = Encoder(E_nets[0], E_nets[1], in_time_steps)
            decoder = Decoder(D_nets[0], D_nets[1], out_time_steps)
            model = ED(encoder, decoder, in_time_steps, out_time_steps)

            
        params_to_update = model.parameters() ############# model params to update ##################
        num_params = sum(p.numel() for p in model.parameters())
        print("-----------------", model)
        model = model.to(device)  

    ## Save model params
    with open(f"{store_dir}/model.txt", 'w') as file:
        # Write the model configuration to the file
        file.write("Model Configuration:\n")
        file.write(str(model))
        file.write("\n\n")
        file.write("Number of Parameters: {}\n".format(num_params))
    with open(f"{store_dir}/opt.txt", 'w') as file:
        file.write(str(opt))

            
    if opt.task == "tracking":
        if "trajectory" in opt.data_name:
            from data.trajectory_loader import TrajectoryLoader
            ## Dataset
            train_dataset = TrajectoryLoader(file_path=opt.train_dir, input_len=opt.in_seq_len, output_len=opt.out_seq_total_len, irregular=False, offset=opt.offset)
            val_dataset = TrajectoryLoader(file_path=opt.val_dir, input_len=opt.in_seq_len, output_len=opt.out_seq_total_len, irregular=False, offset=opt.offset)
            test_dataset = TrajectoryLoader(file_path=opt.test_dir, input_len=opt.in_seq_len, output_len=opt.out_seq_total_len, irregular=False, offset=opt.offset)
            print(f"Data: {opt.data_name}, train set: {len(train_dataset)}, val set: {len(val_dataset)}, test set: {len(test_dataset)}")
            ## Dataloader
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)
            val_dataloader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False)


    
    ####################################################################### 3) Learning Phases #######################################################################
    ## Optimizer
    if opt.optimizer == "adam":
        optimizer = torch.optim.Adam(params_to_update, lr=opt.update_lr)
    ## Criterion
    if opt.criterion == "CrossEntropy":
        criterion = nn.CrossEntropyLoss()
    elif opt.criterion == "MSE_MAE":
        from tools.loss_funcs import MyMseMae
        criterion = MyMseMae(w_mae=1.0, w_mse=1.0)
    elif opt.criterion == "RMSE":
        from tools.loss_funcs import RMSE
        criterion = RMSE()
    elif opt.criterion == "RMSE_Direction":
        from tools.loss_funcs import RMSE_direction
        criterion = RMSE_direction()
        

    if opt.task == "tracking":
        if opt.mode == "train":
            from tools.train_tracking import train
            train(model, train_dataloader, val_dataloader, optimizer, criterion, device="gpu", epoch=opt.epochs, weight_dir=opt.weight, store_dir=store_dir, pretrain=False, opt=opt)

        else:
            print("-------------- Test tracking -----------")
            #from tools.test_tracking import test
            #test(model, train_dataloader, criterion, opt)
    

    
    


