import torch
import torchvision.models as models


def get_feature_extractor():

    # load pretrained DenseNet121
    model = models.densenet121(pretrained=True)

    # remove classification layer
    model.classifier = torch.nn.Identity()

    model.eval()

    return model