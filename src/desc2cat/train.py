import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset, DataLoader
from icecream import ic


def train_model(model, train_dataloader, val_dataloader, device="mps",
                learning_rate=1e-5, patience=5, epochs=2):
    ic.disable()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    category_loss_fn = nn.CrossEntropyLoss(reduction='sum')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    model = model.to(device)
    best_val_loss = float('inf')
    no_improvement_epochs = 0
    batch_print_interval = 5
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        correct_train = 0
        for i, batch in enumerate(train_dataloader):
            input_ids, y_cat, mask = [item.to(device) for item in batch]
            optimizer.zero_grad()
            cat_probs = model(input_ids, attention_mask=mask)
            cat_loss = category_loss_fn(cat_probs, y_cat)
            total_train_loss += cat_loss.item()
            ic(cat_probs.shape)
            ic(y_cat.shape)
            correct_train += (cat_probs.argmax(dim=1) == y_cat.argmax(dim=1)).sum().item()
            cat_loss.backward()
            optimizer.step()
            if (i + 1) % batch_print_interval == 0:
                # Validation phase inside the batch loop
                model.eval()
                total_val_loss = 0
                correct_val = 0
                with torch.no_grad():
                    for batch in val_dataloader:
                        input_ids, y_cat = [item.to(device) for item in batch[:2]]
                        cat_probs = model(input_ids)
                        cat_loss = category_loss_fn(cat_probs, y_cat)
                        total_val_loss += cat_loss.item()
                        correct_val += (cat_probs.argmax(dim=1) == y_cat.argmax(dim=1)).sum().item()
                avg_val_loss = total_val_loss / len(val_dataloader)
                val_acc = correct_val / len(val_dataloader.dataset)
                avg_train_loss = total_train_loss / (i + 1)  # current average for this epoch up to batch i
                train_acc = (correct_train / ((i + 1) * len(batch))) / 100
                print(f"Epoch {epoch}/{epochs} - Batch {i + 1}/{len(train_dataloader)} "
                      f"- Training loss: {avg_train_loss:.4f}, Training Acc: {train_acc:.4f}, "
                      f"Validation loss: {avg_val_loss:.4f}, Validation Acc: {val_acc:.4f}")
                model.train()  # Switch back to training mode
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        # Update learning rate
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
        if no_improvement_epochs >= patience:
            print(f"Stopping early due to no improvement after {patience} epochs.")
            break
    return history


def train_test_split_alt(dataset, test_size=0.2, shuffle=True, random_state=None):
    # Determine the size of dataset
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    # Calculate the split index
    test_split_size = int(np.floor(test_size * dataset_size))

    # Shuffle indices if required
    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        np.random.shuffle(indices)

    # Split indices into training and test subsets
    train_indices, test_indices = indices[test_split_size:], indices[:test_split_size]

    # Create PyTorch data subsets
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, test_dataset
