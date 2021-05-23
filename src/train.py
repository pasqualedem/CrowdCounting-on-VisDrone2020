import time
import torch
import numpy as np

from tqdm import tqdm
from utils import EarlyStopping, RunningAverageMetric, get_optimizer


def train_classifier(
        model,
        train_data,
        valid_data,
        lr=1e-3,
        optimizer='adam',
        batch_size=128,
        epochs=100,
        patience=5,
        weight_decay=0.0,
        n_workers=4,
        device=None,
        verbose=True
):
    # Get the device to use
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Train using device: ' + str(device))

    # Setup the data loaders
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=n_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size, shuffle=False, num_workers=n_workers
    )

    # Compute the class weights (due to dataset im-balance)
    _, class_counts = np.unique(train_data.get_targets(), return_counts=True)
    class_weights = np.min(class_counts) / class_counts

    # Instantiate the NLL losses (with weights)
    nll_loss = torch.nn.NLLLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
        reduction='sum'
    )

    # Move the model to device
    model.to(device)

    # Instantiate the optimizer
    optimizer_kwargs = dict()
    optimizer_class = get_optimizer(optimizer)
    if optimizer_class == torch.optim.SGD:
        # If using SGD, introduce Nesterov's momentum
        optimizer_kwargs['momentum'] = 0.9
        optimizer_kwargs['nesterov'] = True
    optimizer = optimizer_class(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay, **optimizer_kwargs
    )

    # Instantiate the train history
    history = {
        'train': {'loss': [], 'accuracy': []},
        'validation': {'loss': [], 'accuracy': []}
    }

    # Instantiate the early stopping callback
    early_stopping = EarlyStopping(model, patience=patience)

    for epoch in range(epochs):
        start_time = time.time()

        # Initialize the tqdm train data loader, if verbose is enabled
        if verbose:
            tk_train = tqdm(
                train_loader, leave=False, bar_format='{l_bar}{bar:32}{r_bar}',
                desc='Train Epoch %d/%d' % (epoch + 1, epochs)
            )
        else:
            tk_train = train_loader

        # Make sure the model is set to train mode
        model.train()

        # Training phase
        running_train_loss = RunningAverageMetric(train_loader.batch_size)
        running_train_hits = RunningAverageMetric(train_loader.batch_size)
        for inputs, targets in tk_train:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = torch.log_softmax(model(inputs), dim=1)
            loss = nll_loss(outputs, targets)
            running_train_loss(loss.item())
            loss /= train_loader.batch_size
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                predictions = torch.argmax(outputs, dim=1)
                hits = torch.eq(predictions, targets).sum()
                running_train_hits(hits.item())

        # Initialize the tqdm validation data loader, if verbose is specified
        if verbose:
            tk_val = tqdm(
                valid_loader, leave=False, bar_format='{l_bar}{bar:32}{r_bar}',
                desc='Validation Epoch %d/%d' % (epoch + 1, epochs)
            )
        else:
            tk_val = valid_loader

        # Make sure the model is set to evaluation mode
        model.eval()

        # Validation phase
        running_val_loss = RunningAverageMetric(valid_loader.batch_size)
        running_val_hits = RunningAverageMetric(valid_loader.batch_size)
        with torch.no_grad():
            for inputs, targets in tk_val:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = torch.log_softmax(model(inputs), dim=1)
                loss = nll_loss(outputs, targets)
                running_val_loss(loss.item())
                predictions = torch.argmax(outputs, dim=1)
                hits = torch.eq(predictions, targets).sum()
                running_val_hits(hits.item())

        # Get the average train and validation losses and accuracies and print it
        end_time = time.time()
        train_loss = running_train_loss.average()
        train_accuracy = running_train_hits.average()
        val_loss = running_val_loss.average()
        val_accuracy = running_val_hits.average()
        print('Epoch %d/%d - train_loss: %.4f, validation_loss: %.4f, train_acc: %.1f%%, validation_acc: %.1f%% [%ds]' %
              (epoch + 1, epochs, train_loss, val_loss, train_accuracy*100, val_accuracy*100, end_time - start_time))

        # Append losses and accuracies to history data
        history['train']['loss'].append(train_loss)
        history['train']['accuracy'].append(train_accuracy)
        history['validation']['loss'].append(val_loss)
        history['validation']['accuracy'].append(val_accuracy)

        # Check if training should stop according to early stopping
        early_stopping(val_loss)
        if early_stopping.should_stop:
            print('Early Stopping... Best Loss: %.4f' % early_stopping.best_loss)
            break

    return history, early_stopping.best_state
