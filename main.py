import argparse
import logging
import os
import torch.nn as nn
import yaml
import torch
import datetime
import shutil
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
    # /home/qle/Project/DRDC_PassiveRadar_Detection_Tracking/experiments/tracking/trajair/ED_gru_l3_wNone_h128/i10_o10/checkpoints/
    parser.add_argument('--weight', default="/home/qle/Project/DRDC_PassiveRadar_Detection_Tracking/experiments/tracking/trajair/ED_con_gru_l3_w5_h128/testing_3_ok/checkpoints/best_new.pt", help='The inital weigth path')
    #parser.add_argument('--weight', default="/home/qle/Project/DRDC_PassiveRadar_Detection_Tracking/experiments/tracking/trajair/ED_gru_l3_wNone_h128/i10_o10/checkpoints/best_new.pt", help='The inital weigth path')
    parser.add_argument('--store_dir', default="/home/qle/Project/DRDC_PassiveRadar_Detection_Tracking/experiments/tracking/trajair/ED_con_gru_l3_w5_h128/testing_3_ok/", help='The inital weigth path')
    parser.add_argument('--cfg_model', type=str, help='The model configuration', default=None)
    parser.add_argument('--model_name', type=str, help='The model name', default='resnet_18')  # resnet_18 | resnet_18_ABN | classifier_32_ABN
    parser.add_argument('--in_channel', type=int, default=3, help="The number of input channel") # mnist: 1, cifar10: 3, stft: 3
    parser.add_argument('--hidden_channel', type=int, default=128, help="The number of hidden channel") # 64 128
    parser.add_argument('--emb_channel', type=int, default=64, help="The number of embedding channel")
    ## OOD / OSR
    parser.add_argument('--ood_osr', type=str, default='arpl', help='The ood_osr method') # None sigmoid softmax openmax arpl
    parser.add_argument('--ood_threshold', type=float, default=0.95, help='The fixed ood threshold value')
    parser.add_argument('--beta_gan', type=float, default=0.1, help='The fixed ood threshold value')
    ## Model Tracking
    parser.add_argument('--model_architecture', type=str, default="ED_con", help="The model architecture") # ED  ED_con
    parser.add_argument('--central_block', type=str, default="gru", help="The central block")  #rnn  gru  lstm
    parser.add_argument('--num_layers', type=int, default=3, help="The number of layer")
    parser.add_argument('--activate', type=str, default="leaky", help="The number of layer")
    parser.add_argument('--time_sampling', type=str, default="regular", help="The time sampling assumption")
    parser.add_argument('--in_seq_len', type=int, default=10, help="The number of input data")
    parser.add_argument('--out_seq_len', type=int, default=20, help="The number of out data") # 10 20
    parser.add_argument('--out_seq_total_len', type=int, default=20, help="The number of out data") # 10 20
    parser.add_argument('--window_size', type=int, default=5, help="The number of out data") # None 5
    ### Data 
    parser.add_argument('--data_name', type=str, help='The data configuration', default="trajair") # trajectory_3d | trajectory_2d | stft | trajair | mnist | cifar10
    parser.add_argument('--data_root', type=str, help='The data configuration', default="/home/qle/Project/DRDC_PassiveRadar_Detection_Tracking/data/data/mnist/") # mnist | cifar10
    parser.add_argument('--num_classes', type=int, default=4, help="The number of classes") ###
    parser.add_argument('--image_size', type=int, default=224, help='The image size') # stft: 224 | mnist: 28 | cifar: 32
    parser.add_argument('--train_percent', type=float, default=0.75, help='The percentage of the training set.')
    parser.add_argument('--offset', type=int, default=8, help="Offset")
    parser.add_argument('--known_class', type=list, default=[0, 1, 2, 3, 4, 5, 6])
    ## Classification
    #parser.add_argument('--train_dir', type=str, help='The train dir', default="data/STFT_data/STFT_3/2024-01-11_02-13-56/train.txt") #p80_v5_a0.5_n10000_t50.npy  d2_p80_v2_a0.1_n10000_t50.npy
    #parser.add_argument('--val_dir', type=str, help='The val dir', default="data/STFT_data/STFT_3/2024-01-11_02-13-56/test_1.txt")
    #parser.add_argument('--test_dir', type=str, help='The test dir', default="data/STFT_data/STFT_3/2024-01-11_02-13-56/test_1.txt")
    ## Tracking
    # data/trajectory_data/d2_p80_v2_a0.1_n10000_t50.npy
    # data/trajectory_data/d2_p80_v2_a0.1_n1000_t50.npy
    # data/trajectory_data/d2_p80_v2_a0.1_n2000_t50.npy
    # TrajAir
    # /home/qle/Project/DRDC_PassiveRadar_Detection_Tracking/data/trajectory_data/TrajAir/7days1/processed_data/train | test
    #parser.add_argument('--train_dir', type=str, help='The train dir', default="/home/qle/Project/DRDC_PassiveRadar_Detection_Tracking/data/STFT_data/STFT_4/2024-02-14_03-25-17_od_class/train.txt") #p80_v5_a0.5_n10000_t50.npy 
    parser.add_argument('--train_dir', type=str, help='The train dir', default="/home/qle/Project/DRDC_PassiveRadar_Detection_Tracking/data/trajectory_data/TrajAir/7days1/processed_data/train") # d2_p80_v2_a0.1_n10000_t50.npy
    parser.add_argument('--val_dir', type=str, help='The val dir', default="/home/qle/Project/DRDC_PassiveRadar_Detection_Tracking/data/trajectory_data/TrajAir/7days1/processed_data/test")
    parser.add_argument('--test_dir', type=str, help='The test dir', default="/home/qle/Project/DRDC_PassiveRadar_Detection_Tracking/data/trajectory_data/TrajAir/7days1/processed_data/test")
    parser.add_argument('--test_fusion_dir', type=list, help='The test dir', default=["/home/qle/Project/DRDC_PassiveRadar_Detection_Tracking/data/STFT_data/STFT_4/2024-02-14_03-25-17_od_class/test_id_sp.txt",
                                                                                      "/home/qle/Project/DRDC_PassiveRadar_Detection_Tracking/data/STFT_data/STFT_4/2024-02-14_03-25-17_od_class/test_od_sp.txt"])
    #parser.add_argument('--test_fusion_dir', type=list, help='The test dir', default=["/home/qle/Project/DRDC_PassiveRadar_Detection_Tracking/data/STFT_data/STFT_5/2024-03-15_08-00-50/test_2.txt",
    #                                                                                  "/home/qle/Project/DRDC_PassiveRadar_Detection_Tracking/data/STFT_data/STFT_5/2024-03-15_08-00-50/test_1.txt"])
    parser.add_argument('--ood_snr_dir', type=str, help='The test dir', default="/home/qle/Project/DRDC_PassiveRadar_Detection_Tracking/data/STFT_data/STFT_4/2024-02-14_03-25-17_od_class/test_od.txt")
    #parser.add_argument('--ood_snr_dir', type=str, help='The test dir', default="/home/qle/Project/DRDC_PassiveRadar_Detection_Tracking/data/STFT_data/STFT_5/2024-03-15_08-00-50/test_1.txt")
    parser.add_argument('--ood_class_dir', type=str, help='The test dir', default="/home/qle/Project/DRDC_PassiveRadar_Detection_Tracking/data/STFT_data/STFT_4/2024-02-14_03-25-17_od_class/test_od_class.txt")
    #parser.add_argument('--ood_class_dir', type=str, help='The test dir', default="/home/qle/Project/DRDC_PassiveRadar_Detection_Tracking/data/STFT_data/STFT_5/2024-03-15_08-00-50/test_1.txt")
    ### Learning phases
    parser.add_argument('--mode', type=str, default="test", help="train/test process") # test | train
    parser.add_argument('--data_fusion', type=str, default="avg", help=" data fusion method") # avg | None
    parser.add_argument('--epochs', type=int, default=200, help='The number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='The batch size')
    parser.add_argument('--device', default='cpu', help='cpu/gpu')
    parser.add_argument('--optimizer', type=str, default="adam", help="Optimizer")
    parser.add_argument('--update_lr', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--gan_lr', type=float, default=0.0002, help="Learning rate")
    parser.add_argument('--criterion', type=str, default="RMSE", help="loss_function") # MSE_MAE | CrossEntropy | ARPL | RMSE | RMSE_Direction
    ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = parser.parse_args()
    if opt.store_dir is None:
        if opt.task == "classification" or  opt.task == "osr":
            if opt.data_fusion:
                store_dir = f"experiments/{opt.task}_{opt.ood_osr}_{opt.data_fusion}/{opt.data_name}/{opt.model_name}/{datetime_string}" ##############################
            else:
                store_dir = f"experiments/{opt.task}_{opt.ood_osr}_{opt.data_fusion}/{opt.data_name}/{opt.model_name}/{datetime_string}"
        
        elif opt.task == "tracking":
            store_dir = f"experiments/{opt.task}/{opt.data_name}/{opt.model_architecture}_{opt.central_block}_l{opt.num_layers}_w{opt.window_size}_h{opt.hidden_channel}/{datetime_string}"
            #store_dir = f"experiments/{opt.task}/{opt.data_name}/{opt.model_architecture}_{opt.central_block}_l{opt.num_layers}_w{opt.window_size}_h{opt.hidden_channel}/i{opt.in_seq_len}_o{opt.out_seq_len}_{opt.criterion}"
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
        
        elif opt.model_architecture == "ED_con":
            from models.model_tracking_ED_con import ED_conf, Encoder, Decoder, ED_con
            E_nets, D_nets = ED_conf(input_channel=opt.in_channel, hidden_channel=opt.hidden_channel, num_layer=opt.num_layers, block=opt.central_block, activate=opt.activate)
            encoder = Encoder(E_nets[0], E_nets[1])
            decoder = Decoder(D_nets[0], D_nets[1])
            model = ED_con(encoder, decoder, opt=opt)
            
        params_to_update = model.parameters() ############# model params to update ##################
        num_params = sum(p.numel() for p in model.parameters())
        print("-----------------", model)
        model = model.to(device)  
    
    elif opt.task == "osr":
        from models.model_osr import ModelOSR  
        model = ModelOSR(model_name=opt.model_name, in_channel=opt.in_channel, num_classes=opt.num_classes, img_size=opt.image_size)
        params_to_update = model.parameters() ############# model params to update ##################
        num_params = sum(p.numel() for p in model.parameters())
        if opt.ood_osr == "arpl":
            from models.components import gan
            from tools.loss.ARPLoss import ARPLoss
            net_G = gan.Generator32(1, opt.emb_channel, opt.hidden_channel, opt.in_channel)
            net_D = gan.Discriminator32(1, opt.in_channel, opt.hidden_channel) ###
            net_G = net_G.to(device)
            net_D = net_D.to(device)
            criterion = ARPLoss(opt)
            criterion = criterion.to(device)
            criterion_D = nn.BCELoss()
            params_to_update = [{'params': model.parameters()},
                   {'params': criterion.parameters()}]
        model = model.to(device)
        print(model)
    
    ## Save model params
    with open(f"{store_dir}/model.txt", 'w') as file:
        # Write the model configuration to the file
        file.write("Model Configuration:\n")
        file.write(str(model))
        file.write("\n\n")
        file.write("Number of Parameters: {}\n".format(num_params))
    with open(f"{store_dir}/opt.txt", 'w') as file:
        file.write(str(opt))
    
    ####################################################################### 2) Data #######################################################################
    if opt.task == "classification":
        if opt.data_name == "stft":
            if opt.data_fusion is None:
                from data.stft_dataloader import STFTLoader, transform
                ## Dataset
                train_dataset = STFTLoader(file_path=opt.train_dir, ood_path=None, transform=transform)
                val_dataset = STFTLoader(file_path=opt.val_dir, ood_path=opt.ood_dir, transform=transform)
                test_dataset = STFTLoader(file_path=opt.test_dir, ood_path=opt.ood_dir, transform=transform)
                print(f"Data: {opt.data_name}, train set: {len(train_dataset)}, val set: {len(val_dataset)}, test set: {len(test_dataset)}")
                ## Dataloader
                train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)
                val_dataloader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False)
                test_dataloader = DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False)
                if opt.ood_dir is not None:
                    ood_dataset = STFTLoader(file_path=opt.ood_dir, ood_path=None, transform=transform)
                    ood_dataloader = DataLoader(dataset=ood_dataset, batch_size=opt.batch_size, shuffle=False)
            else:
                print("Implement the data fusion ...")
                from data.stft_dataloader import STFTLoaderFusion, transform
                ## Dataset
                train_dataset = STFTLoaderFusion(file_path=opt.train_dir, ood_path=None, transform=transform)
                #val_dataset = STFTLoader(file_path=opt.val_dir, ood_path=opt.ood_dir, transform=transform)
                test_dataset = STFTLoaderFusion(file_path=opt.test_dir, ood_path=None, transform=transform)
                test_dataset_ood_snr = STFTLoaderFusion(file_path=opt.ood_snr_dir, ood_path=None, transform=transform)
                test_dataset_ood_class = STFTLoaderFusion(file_path=opt.ood_class_dir, ood_path=None, transform=transform)
                test_dataset_fusion = []
                for fusion_dir in opt.test_fusion_dir: ################
                    test_dataset_fusion.append(STFTLoaderFusion(file_path=fusion_dir, ood_path=None, transform=transform))    
                print(f"Data: {opt.data_name}, train set: {len(train_dataset)}, test set: {len(test_dataset)}")
                ## Dataloader
                train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)
                #val_dataloader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False)
                test_dataloader = DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False)
                test_dataloader_ood_snr = DataLoader(dataset=test_dataset_ood_snr, batch_size=opt.batch_size, shuffle=False)
                test_dataloader_ood_class = DataLoader(dataset=test_dataset_ood_class, batch_size=opt.batch_size, shuffle=False)
                test_dataloader_fusion = []
                for fusion_dataset in test_dataset_fusion:
                    test_dataloader_fusion.append(DataLoader(dataset=fusion_dataset, batch_size=opt.batch_size, shuffle=False))
            
    elif opt.task == "tracking":
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
        
        elif opt.data_name == "trajair":
            from data.trajair_loader import TrajectoryDataset, seq_collate
            train_dataset = TrajectoryDataset(data_dir=opt.train_dir, obs_len=opt.in_seq_len, pred_len=opt.out_seq_total_len, step=1, delim=' ')
            val_dataset = TrajectoryDataset(data_dir=opt.val_dir, obs_len=opt.in_seq_len, pred_len=opt.out_seq_total_len, step=1, delim=' ')
            test_dataset = TrajectoryDataset(data_dir=opt.val_dir, obs_len=opt.in_seq_len, pred_len=opt.out_seq_total_len, step=1, delim=' ')
            print(f"Data: {opt.data_name}, train set: {len(train_dataset)}, val set: {len(val_dataset)}, test set: {len(test_dataset)}")
            ## Dataloader
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=False, collate_fn=seq_collate)
            val_dataloader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False, collate_fn=seq_collate)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False, collate_fn=seq_collate)
        
    elif opt.task == "osr":
        if opt.data_name == "mnist":
            from data.osr_loader import MNIST_OSR
            Data = MNIST_OSR(known=opt.known_class, dataroot=opt.data_root, batch_size=opt.batch_size, img_size=opt.image_size)
            train_dataloader, val_dataloader, osr_dataloader = Data.train_loader, Data.test_loader, Data.out_loader
        elif opt.data_name == "cifar10":
            from data.osr_loader import CIFAR10_OSR
            Data = CIFAR10_OSR(known=opt.known_class, dataroot=opt.data_root, batch_size=opt.batch_size, img_size=opt.image_size)
            train_dataloader, val_dataloader, osr_dataloader = Data.train_loader, Data.test_loader, Data.out_loader
        ###
        if opt.data_name == "stft":
            if opt.data_fusion is None:
                from data.stft_dataloader import STFTLoader, transform
                ## Dataset
                train_dataset = STFTLoader(file_path=opt.train_dir, ood_path=None, transform=transform)
                val_dataset = STFTLoader(file_path=opt.val_dir, ood_path=opt.ood_dir, transform=transform)
                test_dataset = STFTLoader(file_path=opt.test_dir, ood_path=opt.ood_dir, transform=transform)
                print(f"Data: {opt.data_name}, train set: {len(train_dataset)}, val set: {len(val_dataset)}, test set: {len(test_dataset)}")
                ## Dataloader
                train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)
                val_dataloader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False)
                test_dataloader = DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False)
                if opt.ood_dir is not None:
                    ood_dataset = STFTLoader(file_path=opt.ood_dir, ood_path=None, transform=transform)
                    ood_dataloader = DataLoader(dataset=ood_dataset, batch_size=opt.batch_size, shuffle=False)
            else:
                print("Implement the data fusion ...")
                from data.stft_dataloader import STFTLoaderFusion, transform
                ## Dataset
                train_dataset = STFTLoaderFusion(file_path=opt.train_dir, ood_path=None, transform=transform)
                #val_dataset = STFTLoader(file_path=opt.val_dir, ood_path=opt.ood_dir, transform=transform)
                test_dataset = STFTLoaderFusion(file_path=opt.test_dir, ood_path=None, transform=transform)
                test_dataset_ood_snr = STFTLoaderFusion(file_path=opt.ood_snr_dir, ood_path=None, transform=transform)
                test_dataset_ood_class = STFTLoaderFusion(file_path=opt.ood_class_dir, ood_path=None, transform=transform)
                test_dataset_fusion = []
                for fusion_dir in opt.test_fusion_dir: ################
                    test_dataset_fusion.append(STFTLoaderFusion(file_path=fusion_dir, ood_path=None, transform=transform))    
                print(f"Data: {opt.data_name}, train set: {len(train_dataset)}, test set: {len(test_dataset)}")
                ## Dataloader
                train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)
                #val_dataloader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False)
                val_dataloader = DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False)
                test_dataloader_ood_snr = DataLoader(dataset=test_dataset_ood_snr, batch_size=opt.batch_size, shuffle=False)
                osr_dataloader = DataLoader(dataset=test_dataset_ood_class, batch_size=opt.batch_size, shuffle=False)
                test_dataloader_fusion = []
                for fusion_dataset in test_dataset_fusion:
                    test_dataloader_fusion.append(DataLoader(dataset=fusion_dataset, batch_size=opt.batch_size, shuffle=False))
            
    
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
        
    ## Training phase
    if opt.task == "classification":
        if opt.mode == "train": 
            if opt.data_fusion is None:
                if opt.ood_osr is None:
                    from tools.train_classification import train
                    train(model, train_dataloader, val_dataloader, optimizer, criterion, device="gpu", epoch=opt.epochs, weight_dir=opt.weight, store_dir=store_dir, pretrain=False)
                elif opt.ood_osr == "sigmoid" or opt.ood_osr == "softmax":
                    from tools.train_ood_sigmoid import train
                    train(model, train_dataloader, val_dataloader, ood_dataloader, optimizer, criterion, device="gpu", epoch=opt.epochs, weight_dir=opt.weight,
                     store_dir=store_dir, pretrain=False, ood_threshold=opt.ood_threshold, activation=opt.ood_osr)
                elif opt.ood_osr == "openmax":
                    from tools.train_test_openmax import train
                    train(model, train_dataloader, val_dataloader, ood_dataloader, optimizer, criterion, device="gpu", epoch=opt.epochs, weight_dir=opt.weight,
                     store_dir=store_dir, pretrain=False, ood_threshold=opt.ood_threshold, activation=opt.ood_osr)
            else:
                from tools.train_test_data_fusion import train
                train(model, train_dataloader, test_dataloader, test_dataloader_fusion, test_dataloader_ood_snr, test_dataloader_ood_class, optimizer, criterion, device="gpu", epoch=opt.epochs, weight_dir=opt.weight,
                         store_dir=store_dir, pretrain=False, ood_threshold=opt.ood_threshold, activation=opt.ood_osr)
    elif opt.task == "tracking":
        if opt.mode == "train": 
            if opt.window_size is None:
                from tools.train_tracking import train
                train(model, train_dataloader, val_dataloader, optimizer, criterion, device="gpu", epoch=opt.epochs, weight_dir=opt.weight, store_dir=store_dir, pretrain=False, opt=opt)
            else:
                print("--------------Apply the window sliding-----------")
                from tools.train_tracking_con import train
                train(model, train_dataloader, val_dataloader, optimizer, criterion, device="gpu", epoch=opt.epochs, weight_dir=opt.weight, store_dir=store_dir, pretrain=False, opt=opt)
        else:
            print("-------------- Test tracking -----------")
            from tools.test_tracking import test
            test(model, train_dataloader, criterion, opt)
    
    elif opt.task == "osr":
        if opt.mode == "train": 
            if opt.ood_osr == "arpl":
                from tools.train_osr import train_cs
                optimizer_D = torch.optim.Adam(net_D.parameters(), lr=opt.gan_lr, betas=(0.5, 0.999))
                optimizer_G = torch.optim.Adam(net_G.parameters(), lr=opt.gan_lr, betas=(0.5, 0.999))
                train_cs(model=model, net_D=net_D, net_G=net_G, train_loader=train_dataloader, val_loader=val_dataloader, ood_loader=osr_dataloader,
                 optimizer=optimizer, criterion=criterion, criterion_D=criterion_D, optimizer_D=optimizer_D, optimizer_G=optimizer_G,
                 device=device, epoch=opt.epochs, weight_dir=opt.weight, store_dir=store_dir, opt=opt)
    
    


