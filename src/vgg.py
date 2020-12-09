import torchvision
import torch.nn as nn

def load_feature_extractor():
    vgg19 = torchvision.models.vgg19(pretrained = True)

    first_features = nn.Sequential(*list(vgg19.children())[0][:6]).cuda()
    second_features = nn.Sequential(*list(vgg19.children())[0][:15]).cuda()
    third_features = nn.Sequential(*list(vgg19.children())[0][:26]).cuda()
    first_features.eval()
    second_features.eval()
    third_features.eval()
    first_features.requires_grad_(False)
    second_features.requires_grad_(False)
    third_features.requires_grad_(False)

    feature_extractors = [first_features, second_features, third_features]
    feature_weights = [0.5, 0.5, 0.5]

    return feature_extractors, feature_weights