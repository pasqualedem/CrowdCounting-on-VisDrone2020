import torch
import torch.nn as nn
from models.MobileCount import MobileCount

MBVersions = {
    'MobileCountx0_5': [16, 32, 64, 128],
    'MobileCountx0_75': [32, 48, 80, 160],
    'MobileCount': [32, 64, 128, 256],
    'MobileCountx1_25': [64, 96, 160, 320],
    'MobileCountx2': [64, 128, 256, 512],
}


class CrowdCounter(nn.Module):
    """
    Container class for MobileCount networks
    """
    def __init__(self, gpus, model_name):
        super(CrowdCounter, self).__init__()

        self.CCN = MobileCount(MBVersions[model_name])
        if len(gpus) > 1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
            self.CCN = self.CCN.cuda()
        self.loss_mse_fn = nn.MSELoss().cuda()

    @property
    def loss(self):
        return self.loss_mse

    def f_loss(self):
        return self.loss_mse

    def forward(self, img):
        return self.CCN(img)

    def predict(self, img):
        return self(img)

    def load(self, model_path):
        try:
            self.load_state_dict(torch.load(model_path)['model_state_dict'])
        except KeyError:
            self.load_state_dict(torch.load(model_path))  # Retrocompatibility

    def build_loss(self, density_map, gt_data):
        self.loss_mse = self.loss_mse_fn(density_map.squeeze(), gt_data.squeeze())
        return self.loss_mse

