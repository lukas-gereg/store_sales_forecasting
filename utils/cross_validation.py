import os
import json
import copy
import torch
import wandb
import string
import shutil
import random
import itertools
import numpy as np
from pathlib import Path
from io import TextIOWrapper
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import torch.utils.data as torch_utils
from sklearn.model_selection import StratifiedKFold

from models.base_model import BaseModel
from utils.training import Training
from utils.evaluation import Evaluation
from datasets.custom_concat_dataset import CustomConcatDataset


class CrossValidation:
    def __init__(self, batch_size: int, k_folds: int, debug: bool = False, random_seed: int | None = None):
        self.debug = debug
        self.batch_size = batch_size
        self.k_folds = k_folds
        self.seed = random_seed

    @staticmethod
    def reset_weights(module: BaseModel) -> BaseModel:
        return module.__class__(module.defaults)

    @staticmethod
    def set_dataset_transforms(dataset, data_transforms, label_transforms):
        for sub_dataset in dataset.datasets:
            if isinstance(sub_dataset, Subset):
                sub_dataset = sub_dataset.dataset

            sub_dataset.transform = data_transforms
            sub_dataset.target_transform = label_transforms

        return dataset

    def __call__(self, epochs, device, optimizer, model, loss, train_dataset, validation_dataset, test_dataset, threshold, lr_scheduler=None,
                 wandb_config=None):
        k_fold = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)

        train_data = train_dataset.dataset if isinstance(train_dataset, Subset) else train_dataset
        train_data_transforms = train_data.transform
        train_label_transforms = train_data.target_transform

        validation_data = validation_dataset.dataset if isinstance(validation_dataset, Subset) else validation_dataset
        validation_data_transforms = validation_data.transform
        validation_label_transforms = validation_data.target_transform

        base_dataset = CustomConcatDataset([train_dataset, validation_dataset])

        train_dataset = self.set_dataset_transforms(copy.deepcopy(base_dataset), train_data_transforms, train_label_transforms)
        validation_dataset = self.set_dataset_transforms(copy.deepcopy(base_dataset), validation_data_transforms, validation_label_transforms)

        # TODO when reusing, make sure find_y_by_index is defined, as this is preventing load of x data and making preparation run faster
        labels_list = [[sub_dataset.find_y_by_index(idx) for idx in range(len(sub_dataset))] for sub_dataset in base_dataset.base_datasets]

        labels = list(itertools.chain.from_iterable(labels_list))

        if lr_scheduler is not None:
            scheduler_initial_state = copy.deepcopy(lr_scheduler.state_dict())
        else:
            scheduler_initial_state = dict()

        if wandb_config is not None:
            wandb.init(**wandb_config)
            run_name = wandb.run.name
        else:
            run_name = "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(15))

        print(f"Cross validation run {run_name}")

        splits = [(train_ids.tolist(), validation_ids.tolist()) for (train_ids, validation_ids) in k_fold.split(np.zeros(len(base_dataset)), labels)]

        path = Path(".", "model_params", "folds", f"run-{run_name}", "splits.json")
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w") as outfile:
            outfile: TextIOWrapper = outfile
            label_types_zips = zip([("train_ids", "test_ids") for _ in range(len(splits))], splits)
            reformatted_splits = [dict(zip(*vals)) for vals in label_types_zips]

            json.dump(dict(enumerate(reformatted_splits)), outfile)

        for fold, (train_ids, validation_ids) in enumerate(splits):
            if wandb.run is not None:
                wandb.run.name += f"-fold-{fold}"
            elif wandb_config is not None:
                wandb.init(name=f"{run_name}-fold-{fold}", **wandb_config)

            if wandb.run is None:
                new_weights_name = f"{run_name}-fold-{fold}"
            else:
                new_weights_name = wandb.run.name

            print(f"FOLD {fold}")

            train_sub_sampler = torch_utils.SubsetRandomSampler(train_ids)
            validation_sub_sampler = torch_utils.SubsetRandomSampler(validation_ids)

            train_loader = torch_utils.DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sub_sampler)
            validation_loader = torch_utils.DataLoader(validation_dataset, batch_size=self.batch_size,
                                                       sampler=validation_sub_sampler)
            test_loader = torch_utils.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

            model = self.reset_weights(model)
            optimizer = optimizer.__class__(model.parameters(), **optimizer.defaults)

            if lr_scheduler is not None:
                lr_scheduler.load_state_dict(copy.deepcopy(scheduler_initial_state))

                assert isinstance(optimizer, torch.optim.Optimizer), \
                    f"optimizer should be an instance of {torch.optim.Optimizer.__name__}, instead is {optimizer.__class__.__name__}"

                lr_scheduler.optimizer = optimizer

            training = Training(self.debug)
            losses = training(epochs, device, optimizer, model, loss, train_loader, validation_loader,
                              threshold, lr_scheduler)

            total_loss, results = Evaluation(self.debug)(loss, test_loader, model, device)

            filename = os.path.join(".", "model_params", f"run-{training.run_name}-params.pth")

            file = Path(filename)
            file = file.rename(Path(file.parent, f"run-{new_weights_name}-params.pth"))
            shutil.move(file, path.parent)

            if wandb.run is not None:
                wandb.finish()

            print(f"results of fold {fold}: {results}")
            print(f"Loss per item in test fold {fold}: {total_loss}")

            plt.title(f"Loss Graph of fold {fold}")
            plt.xlabel("Epochs")
            plt.ylabel("Loss per item")

            plt.plot(losses)

            plt.show()