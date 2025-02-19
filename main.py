import os
import torch
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.rmsle import RMSLE
from datasets.sales_dataset import SalesDataSet
from datasets.custom_subset import CustomSubset
from data_preprocessing import DataPreprocessing
from utils.cross_validation import CrossValidation
from models.store_sales_gru_model import StoreSalesGRUModel

if __name__ == '__main__':
    debug = False
    folds = 5
    random_seed = 42

    scheduler = None
    early_stopping = 25

    BATCH_SIZE = 32

    epochs = 10000
    lr = 0.0001

    temporal_length = 14
    output_size = 1
    hidden_size = 256


    # dataframe = DataPreprocessing.create_dataset(os.path.join(os.getcwd(), 'store-sales-time-series-forecasting', 'train.csv'))
    # dataframe.to_csv(os.path.join(os.getcwd(), 'output_train.csv'), index=False)
    dataframe = pd.read_csv(os.path.join(os.getcwd(), 'output_train.csv'))
    dataframe, encoders = DataPreprocessing.preprocess_dataset(dataframe)
    dataframes = DataPreprocessing.group_by_gap(dataframe, encoders)
    # DataPreprocessing.plot_seasonality(dataframes, ['sales'])

    base_dataset = SalesDataSet(dataframes, temporal_length)
    y_ids = [i for i in range(len(base_dataset))]

    train_ids, test_ids = train_test_split(y_ids, test_size=0.3, random_state=random_seed)
    train_ids, validation_ids = train_test_split(train_ids, test_size=0.3, random_state=random_seed)

    train_dataset = CustomSubset(base_dataset, train_ids)
    test_dataset = CustomSubset(base_dataset, test_ids)
    validation_dataset = CustomSubset(base_dataset, validation_ids)

    df = next(iter(dataframes))

    model_properties = {
        'input_size': len(df.columns),
        'hidden_size': hidden_size,
        # 'num_layers': 8,
        'output_size': output_size,
        # 'dropout': 0.1
    }

    model = StoreSalesGRUModel(model_properties)
    rand_val = torch.rand(4, temporal_length, len(df.columns))
    model(rand_val)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss = RMSLE()

    wandb_config = dict(project="MPIS-2024", entity="MPIS-2024", config={
        "model properties": model_properties,
        "temporal length": temporal_length,
        "learning rate": lr,
        "epochs": epochs,
        "early stopping": early_stopping,
        "model": str(model),
        "optimizer": str(optimizer),
        "loss calculator": str(loss),
        "LR reduce scheduler": str(scheduler),
        "debug": debug,
        "batch size": BATCH_SIZE,
        "random seed": random_seed
    })

    wandb_login_key = "a9f105e8b3bc98e07700e93201d4b02c1c75106d"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if wandb_login_key is not None:
        wandb.login(key=wandb_login_key)

    CrossValidation(BATCH_SIZE, folds, debug, random_seed)(epochs, device, optimizer, model, loss,
                                                           train_dataset, validation_dataset, test_dataset,
                                                           early_stopping, scheduler, wandb_config)

    print(f"training of model complete")



    