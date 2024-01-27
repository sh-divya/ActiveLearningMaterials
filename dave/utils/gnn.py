import torch
from tqdm import tqdm
import warnings

from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher


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


class Pyxtal_loss:
    def __init__(self):
        super().__init__()
        self.matcher = StructureMatcher(primitive_cell=False)

    def __call__(self, pred, target, batch):
        rmsd_distance = 0
        loss = torch.nn.MSELoss()

        for i in range(len(batch)):
            # Compute pymatgen structure for predicted graph
            lat = Lattice.from_parameters(*batch.lp[i].tolist())
            s_pred = Structure(
                lattice=lat,
                species=batch.atomic_numbers[batch.batch == i].cpu(),
                coords=pred[batch.batch == i].detach().cpu(),
            )  # coords_are_cartesian=True)
            # Get pymatgen structure for target graph -- initial matbench
            for item in batch.struct[i][0]:  # fix label issue 
                item.label = None 
            s_target = Structure.from_sites(batch.struct[i][0])
            rmsd = self.matcher.get_rms_dist(s_pred, s_target)
            if rmsd: 
                rmsd_distance += rmsd
            else: 
                warnings.warn("RMSD not calculated for {i}th datapoint of the batch")
                rmsd_distance += loss(target[batch.batch == i], pred[batch.batch == i])

        return rmsd_distance / len(batch)
