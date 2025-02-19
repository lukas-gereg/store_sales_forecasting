import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error

from assignment.models.store_sales_gru_model import StoreSalesGRUModel
from assignment.datasets.sales_dataset import SalesDataset


def main():
    # Load and preprocess dataset
    data_path = './MyData/final_train_dataset.csv'
    df = pd.read_csv(data_path)

    # Prekonvertovanie stĺpca 'date' na datetime formát
    df['date'] = pd.to_datetime(df['date'])

    # Filtrovanie dát len pre roky 2016 a 2017
    df = df[(df['date'].dt.year >= 2016) & (df['date'].dt.year <= 2017)]

    # Vytvorenie stĺpca 'days_from_start', ktorý počíta dni od najstaršieho dátumu
    df['days_from_start'] = (df['date'] - df['date'].min()).dt.days

    # Odstránenie stĺpca 'date', ak už nie je potrebný
    df = df.drop(columns=['date'])

    # Define sequence length and split into train/validation/test sets
    sequence_length = 100
    test_size = 0.1  # 10% of the data for testing
    val_size = 0.2   # 20% of the data for validation
    test_split_size = int(len(df) * test_size)
    val_split_size = int(len(df) * val_size)

    # Split dataset
    train_df = df.iloc[:-(test_split_size + val_split_size)]
    val_df = df.iloc[-(test_split_size + val_split_size):-test_split_size]
    test_df = df.iloc[-test_split_size:]

    # Create datasets and data loaders
    train_dataset = SalesDataset(train_df, sequence_length=sequence_length)
    val_dataset = SalesDataset(val_df, sequence_length=sequence_length)
    test_dataset = SalesDataset(test_df, sequence_length=sequence_length)

    batch_size = 1024
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define network parameters
    model_properties = {
        'input_size': train_df.drop(columns=['sales']).shape[1],  # Number of input features
        'hidden_size': 128,
        'num_layers': 8,
        'output_size': 1,  # Predicting one "sales" value
        'dropout': 0.1
    }

    # Initialize the StoreSalesGRUModel
    model = StoreSalesGRUModel(model_properties)

    # Select device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.MSELoss()

    # Early stopping parameters
    num_epochs = 10
    patience = 5  # Stop training if validation loss doesn't improve for 5 consecutive epochs
    best_val_loss = float('inf')
    trigger_times = 0

    # Training loop with validation and early stopping
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs).squeeze()

            # Calculate loss
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.8f}, Validation Loss: {val_loss:.8f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            torch.save(model.state_dict(), 'best_gru_model.pth')  # Save the best model
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping triggered.")
                break

    # Load the best model
    model.load_state_dict(torch.load('best_gru_model.pth'))
    model.eval()

    # Predict sales for the simulated test dataset
    predictions = []
    true_sales = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            predictions.extend(outputs.cpu().numpy())
            true_sales.extend(targets.cpu().numpy())

    # Convert predictions and true sales to NumPy arrays
    predictions = np.array(predictions, dtype=np.float64)
    true_sales = np.array(true_sales, dtype=np.float64)

    # Calculate RMSLE
    rmsle = np.sqrt(mean_squared_log_error(true_sales, predictions))
    print(f"RMSLE on simulated test set: {rmsle:.8f}")

    # Ensure predictions length matches the expected submission format
    predictions = np.array(predictions, dtype=np.float64).flatten()  # Ensure 1D array

    # Adjust predictions to match the test IDs
    submission = pd.DataFrame({
        'id': range(len(predictions)),  # Dummy IDs for simulated test set
        'sales': predictions  # Ensure 1D predictions
    })

    # Save to CSV
    submission.to_csv('submission.csv', index=False)
    print(f"Submission file created with {len(submission)} rows: submission.csv")


if __name__ == '__main__':
    main()
