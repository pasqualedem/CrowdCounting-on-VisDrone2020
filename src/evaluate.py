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
        for inputs, prev_inputs, targets in tqdm(test_loader):
            inputs, prev_inputs, targets = inputs.to(device), prev_inputs.to(device), targets.to(device)
            predictions = model.predict(inputs, prev_inputs)
            y_pred.extend(predictions.cpu().tolist())
            y_true.extend(targets.cpu().tolist())
    return y_true, y_pred


def evaluate_model(model_folder, data_folder, bs, n_workers, losses, device=None):
    from dataset import load_test_dataset
    from model import CANNet2s
    ds = load_test_dataset(data_folder)
    net = CANNet2s()
    net.load(model_folder, testing=True)
    net = net.cuda()
    y_true, y_pred = test_model(net, ds, bs, n_workers, device)

    results = {}
    for loss in losses:
        results[loss] = losses[loss](y_pred, y_true)

    return results