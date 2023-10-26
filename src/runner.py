from torch import nn


class Runner:
    """Runner class that is in charge of implementing routine training functions such as running epochs or doing inference time"""

    def __init__(self, split, train_set, train_loader, model, optimizer):

        self.split = split

        # Initialize class attributes
        self.train_set = train_set

        # Prepare opt, model, and train_loader (helps accelerator auto-cast to devices)
        self.optimizer, self.model, self.train_loader = optimizer, model, train_loader

        # Since data is for targets, use Mean Squared Error Loss
        self.criterion = nn.MSELoss()

    def next(self):
        # Turn the model to training mode (affects batchnorm and dropout)
        if self.split == 'train':
            self.model.train()

        if self.split == 'dev':
            self.model.eval()

        running_loss = 0.0

        # Make sure there are no leftover gradients before starting training an epoch
        if self.split == 'train':
            self.optimizer.zero_grad()

        for sample, target in self.train_loader:
            prediction = self.model(sample)  # Forward pass through model
            loss = self.criterion(prediction, target)  # Error calculation
            running_loss += loss  # Increment running loss
            loss.backward()
            if self.split == 'train':
                self.optimizer.step()  # Update model weights
                self.optimizer.zero_grad()  # Reset gradients to 0

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss
