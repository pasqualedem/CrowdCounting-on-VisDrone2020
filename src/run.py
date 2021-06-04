import torch
import torchvision
from dataset.run_datasets import VideoDataset


def run_model(model_fun, dataset, batch_size, n_workers, callbacks):
    """
    Run the model on a given dataset

    @param model_fun: function that returns the model
    @param dataset: torch Dataset object
    @param batch_size: batch size for parallel computing
    @param n_workers: n° of workers for parallel process
    @param callbacks: list of callback function to execute after each item is processed
    @return:
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running using device: ' + str(device))

    model = model_fun()
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
