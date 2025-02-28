
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from src.data.dataset import Preprocessor,FT_Dataset
from src.modules.ft_tabpfn import FT_TabPFN
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats
import pandas as pd

def train_one_epoch(
    train_loader, 
    model, 
    device,
    criterion,
    optimizer
):
    
    model.train()
    model.to(device)

    batch_loss = 0

    for _, (X_num_batch,X_cat_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        
        X_num_batch, X_cat_batch, y_batch = X_num_batch.to(device), X_cat_batch.to(device), y_batch.to(device)
        y_preds, y_query = model(X_num_batch, X_cat_batch, y_batch)
        y_preds = y_preds.softmax(dim=-1)

        train_loss = criterion(y_preds, y_query)
        reg_loss = model.catagorical_encoder.orthogonal_regularization_bias(regularization_strength=1e-5)
        total_loss = train_loss + reg_loss
        batch_loss+= total_loss.item()
        
        total_loss.backward()
        optimizer.step()

    batch_loss /= len(train_loader)
    return batch_loss


def evaluation(
    eval_dataloader,
    model,
    device,
    criterion
):
    model.eval()
    batch_loss = 0
    with torch.no_grad():
        for _, (X_num_batch,X_cat_batch, y_batch) in enumerate(eval_dataloader):
            
            X_num_batch, X_cat_batch, y_batch = X_num_batch.to(device), X_cat_batch.to(device), y_batch.to(device)
            y_preds, y_query = model(X_num_batch, X_cat_batch, y_batch)
            y_preds = y_preds.softmax(dim=-1)
            loss = criterion(y_preds, y_query)
            batch_loss += loss.item()
        
    batch_loss /= len(eval_dataloader)
    return batch_loss
    
def plot_losses(
    train_loss, 
    eval_loss, 
    file_path="plots/loss_plot.png", 
    title="Training vs Evaluation Loss"
):

    epochs = range(1, len(train_loss) + 1) 
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss', marker='o', linestyle='-')
    plt.plot(epochs, eval_loss, label='Evaluation Loss', marker='s', linestyle='--')
    
    plt.title(title, fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    

    plt.savefig(file_path)
    print("Plot Saved!")

def main(
    X,
    y, 
    epochs=0,
    save_path = "checkpoints/ft_tabpfn.pt"
):  
    preprocessor = Preprocessor()
    X, y = preprocessor.fit(X,y)
    print(f"Num Nans: {preprocessor.nan_numerical} Cat Nans: {preprocessor.nan_categorical}")
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # X_train_num, X_train_cat, y_train = preprocessor.transform(X_train,y_train)
    # X_test_num, X_test_cat, y_test = preprocessor.transform(X_test, y_test)

    # train_dataset = FT_Dataset(X_train_num, X_train_cat, y_train)
    # train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # eval_dataset = FT_Dataset(X_test_num, X_test_cat, y_test)
    # eval_dataloader = DataLoader(eval_dataset, batch_size=128, shuffle=False)
    
    # model = FT_TabPFN(
    #     preprocessor.n_features,
    #     preprocessor.cardinalities,
    #     preprocessor.num_cls
    # )

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # criterion = nn.CrossEntropyLoss()
    # optimiser = torch.optim.Adam(model.parameters(), lr=0.00001)

    # train_loss_arr = []
    # eval_loss_arr = []

    # for e in tqdm(range(epochs), desc="Training Progress"):
    #     train_loss = train_one_epoch(
    #         train_loader,
    #         model,
    #         device,
    #         criterion,
    #         optimiser
    #     )
    #     print(f"Epoch: {e} Train Loss: {train_loss}")
    #     train_loss_arr.append(train_loss)

    #     eval_loss = evaluation(
    #         eval_dataloader,
    #         model,
    #         device,
    #         criterion
    #     )

    #     print(f"Epoch: {e} Eval Loss: {eval_loss}")
    #     eval_loss_arr.append(eval_loss)
    
    # print("Finished Training!")

    # torch.save(model.state_dict(), save_path)
    # print("Model saved!")


def run_repetition(X, y, dataset_name, seed, epochs, batch_size=32, lr=1e-3):
    # Set seeds for reproducibility.
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    
    preprocessor = Preprocessor()
    X,y = preprocessor.fit(X, y)

    # Create the train-test split (50/50) using the current seed.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)

    X_train_num, X_train_cat, y_train = preprocessor.transform(X_train, y_train)
    X_test_num, X_test_cat, y_test = preprocessor.transform(X_test, y_test)

    # Create datasets and loaders.
    train_dataset = FT_Dataset(X_train_num, X_train_cat, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataset = FT_Dataset(X_test_num, X_test_cat, y_test)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate the model.
    model = FT_TabPFN(
        preprocessor.n_features,
        preprocessor.cardinalities,
        preprocessor.num_cls
    )

    # Set device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define loss and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop.
    train_loss_arr = []
    eval_loss_arr = []
    for epoch in tqdm(range(epochs), desc=f"Seed {seed} Training Progress"):
        train_loss = train_one_epoch(
            train_loader,
            model,
            device,
            criterion,
            optimiser
        )
        train_loss_arr.append(train_loss)
        eval_loss = evaluation(
            eval_loader,
            model,
            device,
            criterion
        )
        eval_loss_arr.append(eval_loss)
        print(f"Seed {seed} Epoch {epoch}: Train Loss = {train_loss:.4f}, Eval Loss = {eval_loss:.4f}")

    # Save the model if desired.
    torch.save(model.state_dict(), f"checkpoints/ft_tabpfn_seed_{seed}.pt")
    
    # After training, get predictions on the test set and compute the ROC AUC.
    model.eval()
    all_y_preds = []
    all_y_true = []
    with torch.no_grad():
        for X_num_batch, X_cat_batch, y_batch in eval_loader:
            X_num_batch = X_num_batch.to(device)
            X_cat_batch = X_cat_batch.to(device)
            y_batch = y_batch.to(device)
            # Assuming your model returns logits and maybe something else; adjust if needed.
            y_preds, y_query = model(X_num_batch, X_cat_batch, y_batch)
            # Convert logits to probabilities.
            probs = torch.softmax(y_preds, dim=-1)
            all_y_preds.append(probs.cpu().numpy())
            all_y_true.append(y_query.cpu().numpy())
    
    all_y_preds = np.concatenate(all_y_preds, axis=0)
    all_y_true = np.concatenate(all_y_true, axis=0)
    
    # Calculate ROC AUC.
    # If this is a multi-class problem, use one-vs-one.
    unique_classes = np.unique(all_y_true)
    if len(unique_classes) > 2:
        # Multi-class case: use one-vs-one (or one-vs-rest) strategy
        roc_auc = roc_auc_score(all_y_true, all_y_preds, multi_class="ovo")
    else:
        # Binary classification case: no need for the multi_class parameter
        roc_auc = roc_auc_score(all_y_true, np.max(all_y_preds, axis=-1))
    print(f"Seed {seed}: ROC AUC = {roc_auc:.4f}")
    # Optionally, you could also return the loss arrays or save the loss plots.
    plot_losses(train_loss_arr, eval_loss_arr, file_path=f"plots/loss_plot_seed_{lr}_{dataset_name}_{batch_size}_{seed}.png",
                title=f"Training vs Evaluation Loss (Seed {seed})")
    
    return roc_auc

# A helper function to compute the 95% confidence interval.
def compute_confidence_interval(data, confidence=0.95):
    data = np.array(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    margin = sem * stats.t.ppf((1 + confidence) / 2., len(data)-1)
    return mean, mean - margin, mean + margin

def main_experiment(X, y, dataset_name = "", epochs=10, seeds=[0, 42, 7, 123, 2025], batch_size=32, lr=1e-3):
    roc_auc_scores = []

    for seed in seeds:
        roc_auc = run_repetition(X, y, dataset_name, seed, epochs, batch_size, lr)
        roc_auc_scores.append(roc_auc)
    
    # Aggregate results.
    mean_auc, lower_ci, upper_ci = compute_confidence_interval(roc_auc_scores)
    print("\n--- Aggregated Results ---")
    print(f"ROC AUC scores across seeds: {roc_auc_scores}")
    print(f"Mean ROC AUC: {mean_auc:.4f}")
    print(f"95% Confidence Interval: ({lower_ci:.4f}, {upper_ci:.4f})")
    return mean_auc, roc_auc_scores, lower_ci, upper_ci

if __name__ == "__main__":

    dataset_names = ["cmc", "credit-approval", "cylinder-bands", "dresses-sales"]
    for dataset_name in dataset_names:
        X, y = fetch_openml(name=dataset_name, version=1, return_X_y=True,as_frame=True)
        print(dataset_name)
        main(X,y)

    
    # lrs = [1e-2, 1e-3, 1e-6]
    # batch_sizes = [128,256]
    # epochs = [30, 60, 90, 120]
    # all_columns = ["Dataset", "Epoch", "LR", "Batch-Size", "Mean-ROC", "ROC-per-Seed", "CI-L", "CI-U"]
    # results_csv_path = "results.csv"
    
    # if not os.path.exists(results_csv_path):
    #     pd.DataFrame(columns=[all_columns]).to_csv(results_csv_path, index=False)
    
    # # for epoch in epochs:
    # #     for dataset_name in dataset_names:
    # #         for lr in lrs:
    # #             for batch_size in batch_sizes:
    # # 1) lr 2) batch 3) epoch
    # epoch = 30
    # dataset_name = "cmc"
    # lr = 1e-3
    # batch_size = 128
    

    # X, y = fetch_openml(name=dataset_name, version=1, return_X_y=True,as_frame=True)
    # mean_auc, roc_auc_scores, lower_ci, upper_ci = main_experiment(
    #     X=X,
    #     y=y,
    #     dataset_name = dataset_name,
    #     epochs=epoch,
    #     batch_size=batch_size,
    #     lr = lr
    # )

    # report = {
    #     "Dataset": dataset_name,
    #     "Epoch" : epoch,
    #     "LR" : lr,
    #     "Batch-Size": batch_size,
    #     "Mean-ROC" : f"{mean_auc:.4f}",
    #     "ROC-per-Seed" : str(roc_auc_scores),
    #     "CI-L" : str(lower_ci),
    #     "CI-U" : str(upper_ci)
    # }
    # print(report)
    # pd.DataFrame([report]).to_csv(results_csv_path, mode='a', index=False, header=False)






        




        









    


   

    