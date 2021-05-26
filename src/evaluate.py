import torch
from tqdm import tqdm

def test_model(
        model,
        data_test,
        batch_size=128,
        n_workers=4,
        device=None,
):
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
        for *inputs, targets in tqdm(test_loader):
            inputs = [inp.to(device) for inp in inputs]
            targets = targets.to(device)
            predictions = model.predict(*inputs)

            count_pr = torch.sum(predictions, dim=(1, 2))
            count_gt = torch.sum(targets, dim=(1, 2))

            y_pred.extend(count_pr.cpu().tolist())
            y_true.extend(count_gt.cpu().tolist())

        return y_true, y_pred


def evaluate_model(model_function, data_function, bs, n_workers, losses, device=None):
    ds = data_function()
    net = model_function()
    net = net.to(device)
    y_true, y_pred = test_model(net, ds, bs, n_workers, device)

    results = {}
    for loss in losses:
        results[loss] = losses[loss](y_pred, y_true)

    return results
