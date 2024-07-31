import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from tools.utils_ood import compare_tp_fp, calculate_ood_metrics

def update_learning_rate(optimizer, decay_rate=0.9, lowest=0.0000001, lr=None):
    if lr is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            print(f"The pretrain initial lr: {lr}")
    else:
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            print(f"Recent lr: {lr}")
            lr = max(lr * decay_rate, lowest)
            param_group['lr'] = lr
        for param_group in optimizer.param_groups:
            print(f"Updated lr: {param_group['lr']}")


def predict_logictics(outputs, labels, threshold=0.5):
    predictions = []
    _, predicted_classes = torch.max(outputs, dim=1)
    predicted_probabilities, _ = torch.max(outputs, dim=1)
    predicted_classes[predicted_probabilities < threshold] = 33
    predictions.append(predicted_classes.cpu().numpy())
    predictions = np.concatenate(predictions)
    ###
    label_list = []
    label_np = labels.cpu().numpy()
    for i in range(label_np.shape[0]):
        if max(label_np[i,:]) == 0:
            index = 33
            label_list.append(index)
        else:
            index = np.where(label_np[i,:] == 1)
            label_list.append(index[0][0])
    label_index = np.array(label_list)
    #####
    result = (label_index  == predictions).astype(int)
    result = result.sum()
    return result


def predict(outputs, labels, threshold=None):
    predictions = []
    predicted_probabilities, predicted_classes = torch.max(outputs, dim=1)
    #print(predicted_probabilities)
    if threshold is not None:
        predicted_classes[predicted_probabilities < threshold] = 4 
        
    predictions.append(predicted_classes.cpu().numpy())
    predictions = np.concatenate(predictions)
    ###
    label_list = []
    label_np = labels.cpu().numpy()
    if threshold is not None:
        result_total = (label_np == predictions).astype(int)
        result_total = result_total.sum()
        tpr, fpr, tnr, fnr = calculate_ood_metrics(predictions, label_np, 4)
        result_ood = (tpr, fpr, tnr, fnr)
        return result_total, result_ood
    else:
        result = (label_np == predictions).astype(int)
        result = result.sum()
        return result
    
        
def plot_images(original_image, reconstructed_images, save_path):
    num_images = len(reconstructed_images)
    
    fig, axes = plt.subplots(nrows=1, ncols=num_images+1, figsize=(4*(num_images+1), 4))
    if len(original_image.shape) == 3:
        original_image = np.transpose(original_image, (1, 2, 0))
    # Plot original image
    axes[0].imshow(original_image)
    axes[0].axis('off')
    axes[0].set_title('Original')
    
    # Plot reconstructed images
    for i, img_dict in enumerate(reconstructed_images):
        if len(img_dict["recon_batch"].shape) == 3:
            img_dict["recon_batch"] = np.transpose(img_dict["recon_batch"], (1, 2, 0))

        axes[i+1].imshow(img_dict["recon_batch"])
        axes[i+1].axis('off')
        axes[i+1].set_title(f'Reconstructed - Epoch {img_dict["epoch"]}')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    
def calculate_metrics(TP, TN, FP, FN):
    accuracy = (TP + TN) / (TP + TN + FP + FN + 0.0000000001)
    precision = TP / (TP + FP + 0.0000000001)
    recall = TP / (TP + FN + 0.0000000001)
    specificity = TN / (TN + FP + 0.0000000001)
    f1_score = 2 * (precision * recall) / (precision + recall + 0.0000000001)
    return accuracy, precision, recall, specificity, f1_score
    
    
def plot_learning_curves(train_loss, val_loss, save_path, prefix, title):
    
    
    if train_loss is not None:
        epochs = range(1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, label=f'Train {prefix}')
    if val_loss is not None:
        epochs = range(1, len(val_loss) + 1)
        plt.plot(epochs, val_loss, label=f'Validation {prefix}')

    plt.title(f'Learning Curves of {title}')
    plt.xlabel('Epoch')
    plt.ylabel(f'{prefix}')
    plt.legend()

    plt.savefig(save_path)
    plt.show()
    
    
class BasicBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_layers,
                 batchnorm=False, feature_layer_type="linear"):
        super(BasicBlock, self).__init__()
        self.backbone = nn.Sequential()
        self.feature_layer_type = feature_layer_type
        if num_layers == 1:
            hidden_dim = out_dim
        if feature_layer_type == "linear":
            self.backbone.append(nn.Linear(input_dim, hidden_dim))
            if batchnorm:
                self.backbone.append(nn.BatchNorm1d(hidden_dim))
            self.backbone.append(nn.LeakyReLU())
            c1 = hidden_dim
            c2 = int(hidden_dim // 2)
            for l in range(num_layers - 1):
                self.backbone.append(nn.Linear(c1, c2))
                if batchnorm:
                    self.backbone.append(nn.BatchNorm1d(c2))
                self.backbone.append(nn.LeakyReLU())
                c1 = c2
                c2 = int(c2 / 2)
        else:
            self.backbone.append(nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1))
            if batchnorm:
                self.backbone.append(nn.BatchNorm2d(hidden_dim))
            self.backbone.append(nn.LeakyReLU())
            c1 = hidden_dim
            c2 = int(hidden_dim // 2)
            for l in range(num_layers - 1):
                self.backbone.append(nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1))
                if batchnorm:
                    self.backbone.append(nn.BatchNorm2d(c2))
                self.backbone.append(nn.LeakyReLU())
                c1 = c2
                c2 = int(c2 / 2)

    def forward(self, x):
        out = self.backbone(x)
        return out