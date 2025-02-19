import torch
import wandb
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Validation:
    def __init__(self, debug: bool = False):
        self.debug = debug

    def __call__(self, epoch, validation_loader, device, model, loss, scheduler=None):
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            epoch_validation_loss = 0

            x = torch.tensor([])
            y = torch.tensor([])

            for validation_loader_data, validation_labels, item_name in validation_loader:
                validation_loader_data = validation_loader_data.to(device)
                validation_labels = validation_labels.to(device)

                val_prediction = model(validation_loader_data)

                validation_loss = loss(val_prediction, validation_labels)
                epoch_validation_loss += validation_loss.item()

                if self.debug:
                    print(f"validation item: {item_name}, prediction: {val_prediction.cpu().detach().numpy().tolist()}, "
                          f", ground truth: {validation_labels.cpu().detach().numpy().tolist()}")

                x = torch.cat((x, val_prediction.cpu()))
                y = torch.cat((y, validation_labels.cpu()))

            val_loss = epoch_validation_loss / len(validation_loader)

            if scheduler is not None:
                scheduler.step(val_loss)

            mae = mean_absolute_error(y.detach().numpy(), x.detach().numpy())

            # RMSE
            rmse = np.sqrt(mean_squared_error(y.detach().numpy(), x.detach().numpy()))

            # R² Score
            r2 = r2_score(y.detach().numpy(), x.detach().numpy())

            print(f"Epoch {epoch + 1} validation_report: {dict({'MAE': mae, 'RMSE': rmse, 'R2': r2, 'validation_loss': val_loss})}")

            if wandb.run is not None:
                wandb.log({f"validation_report": {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'loss': val_loss},
                           }, step=epoch)

            return val_loss
