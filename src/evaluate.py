import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from pathlib import Path

def test_model(
        model,
        data_test,
        batch_size=128,
        n_workers=4,
        device=None,
        out_prediction=None
):
    """
    Test the given model on a given dataset

    @param model: torch model to test
    @param data_test: torch dataset for testing the madel
    @param batch_size: batch size for parallel computation
    @param n_workers: n° workers for parallel processing
    @param device: device where to compute the network calculations (cuda or cpu)
    @param out_prediction: boolean that specify if saving the heatmaps generated
    @return: y_true and y_pres
    """
    # Get the device to use
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Test using device: ' + str(device))

    # Setup the data loader
    test_loader = torch.utils.data.DataLoader(
        data_test, batch_size=batch_size, shuffle=False, num_workers=n_workers
    )

    # Make sure the model is set to evaluation mode
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for i, (*inputs, targets) in enumerate(tqdm(test_loader)):
            inputs = [inp.to(device) for inp in inputs]
            targets = targets.to(device)
            predictions = model.predict(*inputs)

            count_pr = torch.sum(predictions.squeeze(), dim=(1, 2)) / 2550
            count_gt = torch.sum(targets.squeeze(), dim=(1, 2))

            if out_prediction:
                for img in range(predictions.shape[0]):
                    out_dir = os.path.join(os.path.dirname(out_prediction), 'preds')
                    Path(out_dir).mkdir(exist_ok=True)
                    plt.imsave(
                        os.path.join(out_dir, str(i)) + '.png',
                        predictions[img].squeeze().cpu().numpy(),
                        cmap='jet', )

            y_pred.extend(count_pr.cpu().tolist())
            y_true.extend(count_gt.cpu().tolist())

        return y_true, y_pred


def evaluate_model(model_function, data_function, bs, n_workers, losses, device=None, out_prediction=None):
    """
    Evaluate a given model on a given dataset using the given loss functions

    @param model_function: function that returns the torch model
    @param data_function: function that return the torch dataset
    @param bs: batch size for parallel computation
    @param n_workers: n° workers for parallel processing
    @param losses: list of loss functions
    @param device: device where to compute the network calculations (cuda or cpu)
    @param out_prediction: boolean that specify if saving the heatmaps generated
    @return: list of loss values
    """
    ds = data_function()
    net = model_function()
    net = net.to(device)
    y_true, y_pred = test_model(net, ds, bs, n_workers, device, out_prediction)

    results = {}
    for loss in losses:
        results[loss] = losses[loss](y_pred, y_true)

    return results
