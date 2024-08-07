import torch
import torch.nn as nn
import time
import os
import numpy as np
from tools.utils import update_learning_rate, predict, plot_learning_curves
from tools.loss_funcs import ade, fde
import matplotlib.pyplot as plt

def plot_sequences(input_seqs, output_seqs, predictions, out_file, num_subfigures=8):
    B, S, C = input_seqs.shape

    fig = plt.figure(figsize=(10, 6 * num_subfigures))

    for i in range(num_subfigures):
        ax = fig.add_subplot(num_subfigures, 1, i + 1, projection='3d')

        # Plot input sequences
        input_seq = input_seqs[i, :, :]
        ax.plot(input_seq[:, 0], input_seq[:, 1], input_seq[:, 2], alpha=0.3, label='Input', color='blue')

        # Plot output sequences
        output_seq = output_seqs[i, :, :]
        ax.plot(output_seq[:, 0], output_seq[:, 1], output_seq[:, 2], alpha=0.7, label='Output', color='red')

        # Plot predicted sequences
        pred_seq = predictions[i, :, :]
        ax.plot(pred_seq[:, 0], pred_seq[:, 1], pred_seq[:, 2], alpha=0.7, label='Prediction', color='green')

        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Set legend
        ax.legend()

    # Save figure
    print("save images!!!")
    plt.savefig(out_file)


def train(model, train_loader, val_loader, optimizer, criterion, device, epoch=100, weight_dir="weights/checkpoints", store_dir="weights/checkpoints", pretrain=False, opt=None):
    if weight_dir is not None:
        print('Load the last checkpoint...')
        try:
            model.load_state_dict(torch.load(weight_dir))
            val_loss_list = []
            val_acc_list = []
            train_loss_list = []
            best_percent = 0
        except:
            params = torch.load(weight_dir)
            model.load_state_dict(params["weight"])
            init_epoch = params["epoch"]
            lr = params["lr"]
            update_learning_rate(optimizer=optimizer, lr=lr)
            best_percent = min(params["val_losses"])
            val_acc_list = params["val_accs"]
            val_loss_list = params["val_losses"] 
            train_loss_list = params["train_losses"]
    else:
        init_epoch = 0
        val_loss_list = []
        val_acc_list = []
        train_loss_list = []
        print('Start training without reload.')
        
    #set the device
    if device == 'cpu':
        device = torch.device('cpu')
    elif device == 'gpu':
        device = torch.device('cuda')
    best_loss = 0
    
    criterion = criterion
    model = model.to(device)
    for epoch in range(init_epoch, init_epoch + epoch):
        print(f'Start epoch {epoch}')
        # initialize total loss
        total_loss = 0
        avg_train_loss = 0
        avg_val_loss = 0
        ade_loss = 0
        fde_loss = 0 
        # iterate over triplets of data
        start_time = time.time()
        for batch_idx, batch in enumerate(train_loader):
            #set to val mode - change the model emp to same to model triplet
            #if batch_idx == 1:
            #    break
            if "trajectory" in opt.data_name:
                input_seq, output_seq = batch
            elif opt.data_name == "trajair":
                input_seq, output_seq, _, _, _, _ = batch
                input_seq = input_seq.permute(1, 0, 2)
                output_seq = output_seq.permute(1, 0, 2)
                #print(input_seq.shape, output_seq.shape)
            input_seq = input_seq.to(device) # B x S x C
            output_seq = output_seq.to(device)
            # start training
            model.train()
            optimizer.zero_grad()
            # Output
            output = model(input_seq.float())
            #print(output.shape)
            #loss = bce_loss(output[:,0], labels[:, 0])
            #for i in range(1,num_classes):
            #    loss += bce_loss(output[:,i], labels[:, i])
            loss = criterion(output, output_seq.float())
            loss.backward()
            optimizer.step()
            # calculate the average loss
            total_loss += loss
            ### ade, fde
            ade_value = ade(output.detach().cpu().numpy(), output_seq.cpu().numpy()) 
            fde_value = fde(output.detach().cpu().numpy(), output_seq.cpu().numpy()) 
            ade_loss += ade_value
            fde_loss += fde_value
            # print the training progress after each epoch
            #print('Epoch: {} Batch: {} Loss: {:.4f}'.format(epoch, batch_idx ,loss))
        avg_train_loss = total_loss / (batch_idx + 1)
        ade_loss /= (batch_idx + 1)
        fde_loss /= (batch_idx + 1)
        train_loss_list.append(avg_train_loss.detach().cpu().numpy())
        
        ###
        if ((epoch+1) % 5 == 0):
            if not os.path.exists(os.path.join(store_dir, 'images_trained')):
                os.makedirs(os.path.join(store_dir, 'images_trained'))
            plot_sequences(input_seq.cpu().numpy(), output_seq.cpu().numpy(), output.detach().cpu().numpy(), out_file=f"{store_dir}/images_trained/images_e{epoch}.png", num_subfigures=8)
        ###        
        
        for batch_idx, batch in enumerate(val_loader):
            if "trajectory" in opt.data_name:
                input_seq, output_seq = batch
            elif opt.data_name == "trajair":
                input_seq, output_seq, _, _, _, _ = batch
                #print(input_seq.shape, output_seq.shape)
                input_seq = input_seq.permute(1, 0, 2)
                output_seq = output_seq.permute(1, 0, 2)
            input_seq = input_seq.to(device) # B x S x C
            output_seq = output_seq.to(device)
            with torch.no_grad():
                model.eval()
                val_prediction = model(input_seq.float())
                val_loss = criterion(val_prediction, output_seq)
                avg_val_loss += val_loss
        avg_val_loss /= (batch_idx + 1)
        avg_val_loss = avg_val_loss.detach().cpu().numpy()
        val_loss_list.append(avg_val_loss)
        #print(f"Validation results of Epoch {epoch}: {val_avg_loss}")
        ################# Save model ##############################
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        params = {
            'weight': model.state_dict(),
            'epoch': epoch,
            'lr': lr,
            'val_accs': val_acc_list,
            'val_losses': val_loss_list,
            'train_losses': train_loss_list,
            'ade': ade,
            'fde': fde  
        }
        if not os.path.exists(os.path.join(store_dir, 'checkpoints')):
            os.makedirs(os.path.join(store_dir, 'checkpoints'))
        torch.save(params, f'{store_dir}/checkpoints/last.pt')
        # deep copy the model
        if epoch == 0:
            best_percent = avg_val_loss
            #save best weight
            print('Save intitial weight')
            torch.save(params, f'{store_dir}/checkpoints/init.pt')
        elif best_percent >= avg_val_loss:
            best_percent = avg_val_loss
            #save best weight
            print(f'Save best weight: {best_percent}')
            torch.save(params, f'{store_dir}/checkpoints/best_new.pt')
        if (epoch+1) % 10 == 0:
            update_learning_rate(optimizer)
        ################# Save model ##############################
        torch.cuda.empty_cache()
        #print('Epoch: {} Avg Loss: {:.4f}, Val percent: {}'.format(epoch, avg_train_loss, val_percent))
        with open(f'{store_dir}/results.txt', 'a') as f:
            f.writelines('Epoch: {} avg_train_Loss: {:.8f}, avg_val_loss: {:.8f}, ade: {:.8f}, fde: {:.8f} \n'.format(epoch ,avg_train_loss, avg_val_loss, ade_loss, fde_loss))
            f.close()
        total_time = time.time() - start_time
        print('Epoch: {} avg_train_Loss: {:.8f}, avg_val_loss: {:.8f}, ade: {:.8f}, fde: {:.8f}, total_time: {:.4f} \n'.format(epoch ,avg_train_loss, avg_val_loss, ade_loss, fde_loss, total_time))
    plot_learning_curves(train_loss_list, val_loss_list, f"{store_dir}/loss_curves.png", "loss", "Loss Learning Curve")
    with open(f'{store_dir}/results.txt', 'a') as f:
        f.writelines(f"Val loss list: {val_loss_list} \n")
        f.writelines(f"Train loss list: {train_loss_list} \n")
        f.close()
        