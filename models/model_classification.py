import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class PermuteLayer(nn.Module):
    def __init__(self, *order):
        super(PermuteLayer, self).__init__()
        self.order = order

    def forward(self, x):
        return x.permute(*self.order)


class SqueezeModule(nn.Module):
    def __init__(self, dim):
        super(SqueezeModule, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)


class ClassifierHeadModule(nn.Module):
    def __init__(self, c1, c2, num_classes):
        super(ClassifierHeadModule, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(c1, 1),
            SqueezeModule(dim=2),
            nn.ReLU(inplace=True),
            nn.Linear(c2, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.seq(x)


class ModelClassification(nn.Module):
    ### 1) init: define initial params
    def __init__(self, model_name=None, in_channel=1, hidden_channel=128, emb_channel=64, num_classes=11, weight=None, img_size=112):
        super().__init__()
        self.model_name = model_name
        if model_name == 'VGG-16':
            self.backbone = models.vgg16(pretrained=False) # 1) backbone => feature extraction
            print(self.backbone)
            head = self.backbone.classifier # 2) head => classification
            new_head = nn.Sequential(
                nn.Linear(head[0].in_features, hidden_channel),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channel, hidden_channel),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channel, num_classes)
            )
            self.backbone.classifier = new_head
            if weight:
                self.backbone.load_state_dict(torch.load(weight))
            for params in self.backbone.parameters():
                params.requires_grad = True
            for params in self.backbone.classifier.parameters():
                params.requires_grad = True

        elif model_name == 'VGG-19':
            self.backbone = models.vgg19(pretrained=False)
            head = self.backbone.classifier
            new_head = nn.Sequential(
                nn.Linear(head[0].in_features, hidden_channel),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channel, hidden_channel),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channel, num_classes)
            )
            self.backbone.classifier = new_head
            if weight:
                self.backbone.load_state_dict(torch.load(weight))
            for params in self.backbone.parameters():
                params.requires_grad = True
            for params in self.backbone.classifier.parameters():
                params.requires_grad = True

        elif model_name == 'resnext101_32x8d':
            self.backbone = models.resnext101_32x8d(pretrained=False)
            head = self.backbone.fc
            new_head = nn.Sequential(
                nn.Linear(head.in_features, hidden_channel),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channel, hidden_channel),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channel, num_classes)
            )
            self.backbone.fc = new_head
            if weight:
                self.backbone.load_state_dict(torch.load(weight))
            for params in self.backbone.parameters():
                params.requires_grad = True
            for params in self.backbone.fc.parameters():
                params.requires_grad = True

        elif model_name == 'swin_base_mini':
            self.backbone = models.swin_v2_t(weights=None)
            ### adjust the head
            linear = self.backbone.head
            new_linear = nn.Sequential(
                nn.Linear(linear.in_features, hidden_channel),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channel, num_classes),
            )
            self.backbone.head = new_linear
            ### adjust the feature0
            feature0 = self.backbone.features[0]
            feature0[0] = nn.Conv2d(in_channel, 96, kernel_size=(3, 3))
            self.backbone.features[0] = feature0
            for param in self.backbone.parameters():
                param.requires_grad = True

        elif model_name == 'resnet_50':
            self.backbone = models.resnet50(pretrained=False)
            linear = self.backbone.fc
            new_linear = nn.Sequential(
                nn.Linear(linear.in_features, hidden_channel),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channel, hidden_channel),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channel, num_classes),
                # nn.Sigmoid()
            )
            self.backbone.fc = new_linear
            if weight:
                self.backbone.load_state_dict(torch.load(weight))
            # Freeze all the parameters except the last and second last layer
            for param in self.backbone.parameters():
                param.requires_grad = True

        
        elif model_name == 'resnet_18':
            self.backbone = models.resnet18(pretrained=False)
            linear = self.backbone.fc
            new_linear = nn.Sequential(
                nn.Linear(linear.in_features, hidden_channel),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channel, emb_channel),
                nn.ReLU(inplace=True),
                #nn.Linear(hidden_channel, num_classes),
                # nn.Sigmoid()
            )
            self.backbone.fc = new_linear
            self.head = nn.Sequential(
                nn.Linear(emb_channel, num_classes)
            )
            if weight:
                self.backbone.load_state_dict(torch.load(weight))
            # Freeze all the parameters except the last and second last layer
            for param in self.backbone.parameters():
                param.requires_grad = True
                
        else:
            raise "This model is not implemented!"

        #if weight:
        #    print(f'Load the model from the checkpoint: {weight}')
        #    self.load_state_dict(torch.load(weight))

    ### forward: execute the model
    def forward(self, input, feature_return=False):
        if feature_return:
            feature = self.backbone(input)
            out = self.head(feature)
            return feature, out
        else:
            feature = self.backbone(input)
            out = self.head(feature)
            return out

################ Testing
'''
model_name = "..."
in_channel = 1
hidden_channel = 64
emb_channel = 64
num_classes = 7
model = ModelClassification(model_name=model_name, in_channel=in_channel, hidden_channel=hidden_channel, emb_channel=emb_channel,
         num_classes=num_classes, weight=None)
print(model)
'''
