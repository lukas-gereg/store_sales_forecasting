import torch
import wandb
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Evaluation:
    def __init__(self, debug: bool = False):
        self.debug = debug

    def __call__(self, loss, test_loader, model, device):
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            total_loss = 0
            results = []

            x = torch.tensor([])
            y = torch.tensor([])

            for test_loader_data, labels, item_name in test_loader:
                test_loader_data = test_loader_data.to(device)
                labels = labels.to(device)

                outputs = model(test_loader_data)
                total_loss += loss(outputs, labels).item()

                x = torch.cat((x, outputs.cpu()))
                y = torch.cat((y, labels.cpu()))

                if self.debug:
                    print(
                        f"evaluation item: {item_name}, prediction: {outputs.cpu().detach().numpy().tolist()}, "
                        f"ground truth: {labels.cpu().detach().numpy().tolist()}")

                results.extend(zip(labels.cpu(), outputs.cpu(), item_name))

            eval_loss = total_loss / len(test_loader)

            mae = mean_absolute_error(y.detach().numpy(), x.detach().numpy())

            # RMSE
            rmse = np.sqrt(mean_squared_error(y.detach().numpy(), x.detach().numpy()))

            # R² Score
            r2 = r2_score(y.detach().numpy(), x.detach().numpy())

            print(f"evaluation_report: {dict({'MAE': mae, 'RMSE': rmse, 'R2': r2, 'evaluation_loss': eval_loss})}")

            if wandb.run is not None:
                wandb.log({f"evaluation_report": {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'loss': eval_loss}})

            return eval_loss, results
