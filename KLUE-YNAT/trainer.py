import torch
import torch.nn as nn
from tqdm import tqdm
from torchmetrics.classification import MulticlassF1Score




class Trainer:
    def __init__(self, model, learning_rate, num_epochs, device):
        self.model = model
        self.model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        self.device = device

        self.score_list = []

    def train(self, train_loader):
        for epoch in range(self.num_epochs):
            self.model.train()
            for i, (input, label) in enumerate(tqdm(train_loader, desc=f"{epoch + 1}/{self.num_epochs} epoch.. Training...")):
                input = input.to(self.device)
                label = label.to(self.device)

                # Forward pass
                outputs = self.model(input)
                loss = self.criterion(outputs, label)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"* Train Loss: {loss.item():.2f}")

    def evaluation(self, val_loader, num_classes):
        # Test the model
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            f1_score = 0
            f1 = MulticlassF1Score(num_classes)
            for (input, label) in val_loader:
                input = input.to(self.device)
                label = label.to(self.device)
                outputs = self.model(input)
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
                f1_score += f1(predicted, label)

            acc = 100 * correct / total
            f1_score = 100 * f1_score / total

            # print('acc:', acc)
            # print('f1_score:', f1_score)

        return acc, f1_score


