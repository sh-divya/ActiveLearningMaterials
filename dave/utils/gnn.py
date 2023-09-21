import torch
from tqdm import tqdm


class GaussianSmearing(torch.nn.Module):
    r"""Smears a distance distribution by a Gaussian function."""

    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1).to(dist.device)
        return torch.exp(self.coeff * torch.pow(dist, 2))


def preprocess_data(batch, preproc_method):
    """Preprocess training data

    Args:
        batch (data): batch of data objects
        preproc_method (str): name of preprocessing mehtod

    Returns:
        data, tensor: correct data and target
    """
    if preproc_method == "graph":
        x = batch
        y = batch.y
    else:
        x, y = batch
    return x, y


def custom_fit(
    model, train_dataloader, val_dataloader, criterion, max_epochs, optimizer, device
):
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0

        # Training loop
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/ {max_epochs}"):
            batch = batch.to(device)
            batch, y = preprocess_data(batch, "graph")

            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch)

            # Compute loss
            loss = criterion(outputs, y)  # Calculate your loss here

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Update training loss
            train_loss += loss.item()

        # Calculate and log average training loss for the epoch
        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} - Avg. Training Loss: {avg_train_loss:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                batch = batch.to(device)
                outputs = model(batch)
                loss = criterion(
                    outputs, batch.y
                )  # Calculate your validation loss here
                val_loss += loss.item()

        # Calculate and log validation loss
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch + 1} - Avg. Validation Loss: {avg_val_loss:.4f}")

    print("Training complete.")
