import os
import torch
import wandb
import random
import string
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from assignment.utils.validation import Validation


class Training:
    def __init__(self, debug: bool = False):
        self.validation = Validation(debug)
        self.debug = debug
        self.run_name = ""

    def __call__(self, epochs, device, optimizer, model, loss, train_loader, validation_loader, threshold, validation_scheduler=None):
        if wandb.run is not None:
            self.run_name = wandb.run.name
        else:
            self.run_name = "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(15))

        print(f"Training run {self.run_name}")

        model = model.to(device)

        x = torch.tensor([])
        y = torch.tensor([])

        old_validation_value = self.validation(0, validation_loader, device, model, loss)
        counter = 0
        torch.save(obj=model.state_dict(), f=os.path.join(".", "model_params", f"run-{self.run_name}-params.pth"))
        losses = []

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0

            x = torch.tensor([])
            y = torch.tensor([])

            for batch_idx, (data, labels, item_name) in enumerate(train_loader):
                data = data.to(device)
                labels = labels.to(device)

                prediction = model(data)

                current_loss = loss(prediction, labels)

                optimizer.zero_grad()
                current_loss.backward()
                optimizer.step()
                epoch_loss += current_loss.item()

                if self.debug:
                    print(f"train item: {item_name}, prediction: {prediction.cpu().detach().numpy().tolist()}, "
                          f"ground truth: {labels.cpu().detach().numpy().tolist()}")

                x = torch.cat((x, prediction.cpu()))
                y = torch.cat((y, labels.cpu()))

            train_loss = epoch_loss / len(train_loader)

            mae = mean_absolute_error(y.detach().numpy(), x.detach().numpy())

            # RMSE
            rmse = np.sqrt(mean_squared_error(y.detach().numpy(), x.detach().numpy()))

            # R² Score
            r2 = r2_score(y.detach().numpy(), x.detach().numpy())

            print(f"Epoch {epoch + 1} train_report: {dict({'MAE': mae, 'RMSE': rmse, 'R2': r2, 'train_loss': train_loss})}")

            if wandb.run is not None:

                wandb.log({f"train_report": {'loss': train_loss, 'MAE': mae, 'RMSE': rmse, 'R2': r2},
                           "epoch": epoch + 1,
                           }, step=epoch + 1)

            current_validation_value = self.validation(epoch + 1, validation_loader, device, model, loss, validation_scheduler)

            losses.append(current_validation_value)

            if threshold is None or current_validation_value < old_validation_value:
                old_validation_value = current_validation_value
                counter = 0
                torch.save(obj=model.state_dict(), f=os.path.join(".", "model_params", f"run-{self.run_name}-params.pth"))
            elif counter < threshold:
                counter += 1
            else:
                model.load_state_dict(torch.load(os.path.join(".", "model_params", f"run-{self.run_name}-params.pth")))
                print(f"Risk of over fitting parameters, ending learning curve at epoch {epoch + 1}, reverting back to epoch {epoch - counter}.")

                print("labels: ", y.detach().numpy().tolist())
                print("predictions: ", x.detach().numpy().tolist())
                print("labels meanings: ", train_loader.dataset.classes)

                return losses[: -counter]

        model.load_state_dict(torch.load(os.path.join(".", "model_params", f"run-{self.run_name}-params.pth")))

        print("labels: ", y.detach().numpy().tolist())
        print("predictions: ", x.detach().numpy().tolist())
        print("labels meanings: ", train_loader.dataset.classes)

        return losses
