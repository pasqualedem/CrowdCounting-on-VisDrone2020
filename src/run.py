import torch
import torchvision


def run_model(model_fun, dataset, batch_size, n_workers, callback):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running using device: ' + str(device))

    model = model_fun()
    # Setup the data loader
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
                callback(input[i], predictions[i], other[i])


def run_transforms(mean, std, size):
    return torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(mean=mean,
                                                                            std=std),
                                           torchvision.transforms.Resize(size)
                                           ])  # normalize to (-1, 1)
