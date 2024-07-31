import pickle

###
def plot_sequences(input_seqs, output_seqs, predictions, predictions_1, predictions_2, predictions_3):
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
    ax.plot(pred_seq[:, 0], pred_seq[:, 1], pred_seq[:, 2], alpha=0.7, label=f'(GRU Normal Prediction)', color='green')
    
    # Plot predicted sequences
    pred_seq = predictions_1[:, :]
    ax.plot(pred_seq[:, 0], pred_seq[:, 1], pred_seq[:, 2], alpha=0.7, label=f'(GRU Normal Prediction)', color='green')
    
    # Plot predicted sequences
    pred_seq = predictions_2[:, :]
    ax.plot(pred_seq[:, 0], pred_seq[:, 1], pred_seq[:, 2], alpha=0.7, label=f'(GRU Normal Prediction)', color='green')
    
    # Plot predicted sequences
    pred_seq = predictions_3[:, :]
    ax.plot(pred_seq[:, 0], pred_seq[:, 1], pred_seq[:, 2], alpha=0.7, label=f'(GRU Normal Prediction)', color='green')

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set legend
    ax.legend()
    plt.show()
    
# Specify the file path for the pickle file
file_path = "data.pickle"

# Open the file in binary mode and load the data using pickle
with open(file_path, "rb") as file:
    loaded_data = pickle.load(file)

file_path_con = "data_con.pickle"

# Open the file in binary mode and load the data using pickle
with open(file_path_con, "rb") as file:
    loaded_data_con = pickle.load(file)
    
###
data_in = loaded_data_con["trajectory_in"]
data_out = loaded_data_con["trajectory_out"]
data_gru_2 = loaded_data_con["trajectory_gru_normal"]
data_gru_con_2 = loaded_data_con["trajectory_gru_con"]
data_gru_1 = loaded_data["trajectory_gru_normal"]
data_gru_con_1 = loaded_data["trajectory_gru_con"]