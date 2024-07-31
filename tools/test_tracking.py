import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F
import pickle

def plot_sequences(input_seqs, output_seqs, predictions, predictions_2):
    num_agents = 1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot input sequences
    input_seq = input_seqs[:, :]
    ax.plot(input_seq[:, 0], input_seq[:, 1], input_seq[:, 2], alpha=0.3, label=f'(Input)', color='blue')

    # Plot output sequences
    output_seq = output_seqs[:, :]
    ax.plot(output_seq[:, 0], output_seq[:, 1], output_seq[:, 2], alpha=0.7, label=f'(Output)', color='red')

    # Plot predicted sequences
    pred_seq = predictions[:, :]
    ax.plot(pred_seq[:, 0], pred_seq[:, 1], pred_seq[:, 2], alpha=0.7, label=f'(Normal Prediction)', color='green')
    
    # Plot predicted sequences
    pred_seq = predictions_2[:, :]
    ax.plot(pred_seq[:, 0], pred_seq[:, 1], pred_seq[:, 2], alpha=0.7, label=f'(Prediction with continual inference)', color='purple')

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set legend
    ax.legend()
    plt.show()
    
def direction_loss(y1, y2):
    """
    y1, y2: tensors of shape (B, S, C)
    """

    y1_normalized = F.normalize(y1, dim=2)
    y2_normalized = F.normalize(y2, dim=2)

    cosine_similarity = torch.sum(y1_normalized * y2_normalized, dim=2)

    # Add a small epsilon to handle the case when y1 and y2 are exactly the same
    epsilon = 1e-8
    cosine_similarity = cosine_similarity.clamp(-1 + epsilon, 1 - epsilon)

    angle_difference = torch.acos(cosine_similarity)

    return angle_difference.mean()
    
def plot_trajectory_3d(trajectory_1, trajectory_2, trajectory_3=None):
        x = trajectory_1[:, 0]
        y = trajectory_1[:, 1]
        z = trajectory_1[:, 2]
        
        x_2 = trajectory_2[:, 0]
        y_2 = trajectory_2[:, 1]
        z_2 = trajectory_2[:, 2]
        
        if trajectory_3 is not None:
            x_3 = trajectory_3[:, 0]
            y_3 = trajectory_3[:, 1]
            z_3 = trajectory_3[:, 2]

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
        
def combine_tensors(s1, s2, w):
    #assert s1.shape == s2.shape, "Tensors must have the same shape"
    assert w <= s1.size(1), "w must be smaller than or equal to the sequence length"
    if w == 0:
        combined = torch.cat((s1, s2), dim=1)
    else:
        combined = torch.cat((s1[:, :-w, :], (s1[:, -w:, :] + s2[:, :w, :]) / 2, s2[:, w:, :]), dim=1)
    return combined

def test(model, test_dataloader, criterion, opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model.load_state_dict(torch.load(opt.weight))
    except:
        params = torch.load(opt.weight)
        model.load_state_dict(params["weight"])
    val_loss_best = 0.0
    val_loss_avg = 0.0
    val_loss_con_avg = 0.0
    prediction_best = None
    prediction_best_con = None
    input_best = None
    output_best = None
    for id, (batch) in enumerate(test_dataloader):
        #if id == 1:
        #    break
        if "trajectory" in opt.data_name:
            input_seq, output_seq = batch
        elif opt.data_name == "trajair":
            input_seq, output_seq, _, _, _, _ = batch
            input_seq = input_seq.permute(1, 0, 2)
            output_seq = output_seq.permute(1, 0, 2)
            #print(input_seq.shape, output_seq.shape)
        input_seq, output_seq = input_seq.to(device), output_seq.to(device)
        seq = torch.cat([input_seq, output_seq], dim=1)
        with torch.no_grad():
            model.eval()
            if opt.model_architecture == "ED_con":
                val_prediction, _, _ = model(input_seq)
                
                val_prediction = val_prediction[:,-20:,:]
                #print(val_prediction.shape)
            else:
                val_prediction = model(input_seq)
            val_loss = criterion(val_prediction, output_seq)
            pred_recent = val_prediction[:,:5,:]
            for i in range(1, 4):
                if opt.model_architecture == "ED_con":
                    pred, _, _ = model(seq[:, i*5:i*5+10,:])
                    pred = pred[:,-20:,:][:,:5,:]
                else: 
                    pred = model(seq[:, i*5:i*5+10,:])[:,:5,:]
                pred_recent = combine_tensors(pred_recent, pred, 0)
            val_loss_con = criterion(pred_recent, output_seq)
            ### continual inference
            print(id, val_loss, val_loss_con)
            val_loss_avg += val_loss
            val_loss_con_avg += val_loss_con
            if id == 0:
                val_loss_best = val_loss
                prediction_best = val_prediction
                input_best = input_seq
                output_best = output_seq
                prediction_best_con = pred_recent
                prediction_best = val_prediction
            else:
                if val_loss <= val_loss_best:
                    val_loss_best = val_loss
                    print(val_loss_best)
                    prediction_best = val_prediction
                    input_best = input_seq
                    output_best = output_seq
                    prediction_best_con = pred_recent
    
    val_loss_avg /= (id + 1)
    val_loss_con_avg /= (id + 1)               
    #trajectory_1 = np.array(torch.cat([input_best.cpu() , output_seq.cpu() ], dim=1)[9])
    trajectory_in = np.array(input_best.cpu()[0])
    trajectory_out = np.array(output_best.cpu()[0])
    ###
    #val_prediction = model(input_best)
    #pred_recent = val_prediction[:,:5,:]
    #for i in range(1, 4):
    #    pred = model(seq[:, i*5:i*5+10,:])[:,:5,:]
    #    pred_recent = combine_tensors(pred_recent, pred, 0)
    ###
    trajectory_2 = np.array(prediction_best.cpu()[0])
    trajectory_3 = np.array(prediction_best_con.cpu()[0])
    #trajectory_4 =  np.array(val_prediction.detach().cpu()[0])
    #plot_trajectory_3d(trajectory_in, trajectory_out)
    #plot_trajectory_3d(trajectory_in, trajectory_3)
    val_loss = criterion(prediction_best, output_best)
    val_loss_con = criterion(prediction_best_con, output_best)
    val_loss_test = criterion(output_best, output_best) 
    print("data shape", trajectory_2.shape, trajectory_3.shape)
    #print("---RMSE----", val_loss, val_loss_con, val_loss_test)
    print("---RMSE----", val_loss_avg, val_loss_con_avg)
    print("---Direction loss----", direction_loss(prediction_best, output_best), direction_loss(prediction_best_con, output_best))
    #plot_sequences(trajectory_in, trajectory_out, trajectory_2, trajectory_3)
    ### 
    data_dict = {
    "trajectory_in": trajectory_in,
    "trajectory_out": trajectory_out,
    "trajectory_gru_normal": trajectory_2,
    "trajectory_gru_con": trajectory_3
    }
    file_path = "data_con.pickle"
    with open(file_path, "wb") as file:
        pickle.dump(data_dict, file)