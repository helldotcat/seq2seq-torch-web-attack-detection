import time

import matplotlib.pyplot as plt
import torch
import tqdm
from IPython.display import clear_output
from torch import nn


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    
    return elapsed_mins, elapsed_secs


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    
    print('Evaluate')
    with torch.no_grad():
        for _, batch in tqdm.tqdm(enumerate(iterator)):
            output = model.forward(batch, teacher_forcing_ratio=1.0)

            # batch = [request len, batch size]
            # output = [request len, batch size, output dim]

            output = output[1:].permute(1, 2, 0)
            trg = batch[1:].permute(1, 0)

            # trg = [(request len - 1) * batch size]
            # output = [(request len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            epoch_loss += loss.item()
            
    return epoch_loss / len(iterator)


def train(model, iterator, optimizer, criterion, train_history=None, valid_history=None, teacher_forcing_ratio=0.5):
    model.train()
    
    epoch_loss = 0
    history = []
    
    for i, batch in tqdm.tqdm(enumerate(iterator)):
        optimizer.zero_grad()
        output = model.forward(batch, teacher_forcing_ratio=teacher_forcing_ratio)

        # trg = [request len, batch size]
        # output = [request len, batch size, output dim]
        
        # output = output[1:].view(-1, output.shape[-1])
        # trg = trg[1:].view(-1)
        output = output[1:].permute(1, 2, 0)
        trg = batch[1:].permute(1, 0)

        # output = [batch size, output dim, request len]
        # trg = [batch size, request len]
        
        loss = criterion(output, trg)
        loss.backward()
        
        # Let's clip the gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        
        epoch_loss += loss.item()
        history.append(loss.cpu().data.numpy())
        
        if (i + 1) % 10 == 0:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
            clear_output(True)
            
            ax[0].plot(history, label='train loss')
            ax[0].set_xlabel('Batch')
            ax[0].set_yscale('log')
            ax[0].set_title('Train loss')
            
            ax[1].plot(train_history, label='general train history')
            ax[1].plot(valid_history, label='general valid history')
            ax[1].set_yscale('log')
            ax[1].set_xlabel('Epoch')

            plt.legend()
            plt.show()

    return epoch_loss / len(iterator)


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.modules.loss._Loss,
    
    epochs: int,
    teacher_forcing_ratio: float,
    
    train_iterator: torch.utils.data.dataloader.DataLoader,
    val_iterator: torch.utils.data.dataloader.DataLoader,
):
    train_history = []
    valid_history = []

    best_valid_loss = float('inf')

    training_start_time = time.time()

    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, train_history, valid_history, teacher_forcing_ratio)
        valid_loss = evaluate(model, val_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'model_{best_valid_loss:.3f}.pt')

        train_history.append(train_loss)
        valid_history.append(valid_loss)

        print(f'Epoch: {epoch+1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}')
    
    training_stop_time = time.time()
    training_mins, training_secs = epoch_time(training_start_time, training_stop_time)
    print(f'Training time: {training_mins}m {training_secs}s')
