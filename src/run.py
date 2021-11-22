import torch
import torchvision
from dataset.run_datasets import VideoDataset
from models.CC import CrowdCounter
from config import cfg
from dataset.run_datasets import make_dataset
from dataset.visdrone import cfg_data
from callbacks import call_dict


def run_model(model, dataset, batch_size, n_workers, callbacks):
    """
    Run the model on a given dataset

    @param model: Torch model
    @param dataset: torch Dataset object
    @param batch_size: batch size for parallel computing
    @param n_workers: n° of workers for parallel process
    @param callbacks: list of callback function to execute after each item is processed
    @return:
    """
    device = cfg.DEVICE
    print('Running using device: ' + str(device))

    # Setup the data loader
    if type(dataset) == VideoDataset:
        n_workers = 0
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers
    )

    # Make sure the model is set to evaluation mode
    model.eval()

    with torch.no_grad():
        for input, other in data_loader:
            input = input.to(device)
            predictions = model.predict(input)

            input = input.to('cpu')
            predictions = predictions.to('cpu')
            for i in range(input.shape[0]):
                for callback in callbacks:
                    callback(input[i], predictions[i], other[i])


def run_transforms(mean, std, size):
    """
    Run the necessary transformation for running the image in the net

    @param mean: mean for normalizing
    @param std: standard devation for normalizing
    @param size: resize input for the net
    """
    return torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(mean=mean,
                                                                            std=std),
                                           torchvision.transforms.Resize(size)
                                           ])  # normalize to (-1, 1)


def load_CC_run():

    """
    Load CrowdCounter model net for testing mode
    """
    if cfg.DEVICE == 'cuda' and torch.cuda.is_available():
        device = [0]
    else:
        device = 'cpu'
        cfg.DEVICE = 'cpu'

    cc = CrowdCounter(device, cfg.NET)
    if cfg.PRE_TRAINED:
        cc.load(cfg.PRE_TRAINED)
    return cc


def run_net(in_file, callbacks, model=None):
    """
    Run the model on a given file or folder

    @param in_file: media file or folder of images
    @param callbacks: list of callbacks to be called after every forward operation
    @param model: torch model
    """
    dataset = make_dataset(in_file)

    transforms = run_transforms(cfg_data.MEAN, cfg_data.STD, cfg_data.SIZE)
    dataset.set_transforms(transforms)

    if model == None:
        model = load_CC_run()

    callbacks_list = [(call_dict[call] if type(call) == str else call) for call in callbacks]

    run_model(model, dataset, cfg.TEST_BATCH_SIZE, cfg.N_WORKERS, callbacks_list)
