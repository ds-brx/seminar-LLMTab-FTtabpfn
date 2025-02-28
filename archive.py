class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.y_encoder = LabelEncoder()
        self.n_features = 0  # Number of numerical features
        self.num_cls = 0  # Number of classes in target variable y
        self.cardinalities = []  # List to store cardinality for each categorical feature

    def fit(self, X, y):
        # Identify numerical and categorical features
        self.num_features = X.select_dtypes(include=['number']).columns
        self.cat_features = X.select_dtypes(include=['category', 'object']).columns
        

        # Store the number of numerical features
        self.n_features = len(self.num_features)
        
        # Drop rows with missing numerical values
        X_clean = X.dropna(subset=self.num_features)
        y = y.loc[X_clean.index]
        
        # Fit scaler
        self.scaler.fit(X_clean[self.num_features])
        
        # Fit label encoders for categorical features
        for col in self.cat_features:
            X_clean[col] = X_clean[col].astype('category').cat.add_categories(['Unknown']).fillna('Unknown')
            le = LabelEncoder()
            le.fit(X_clean[col])
            self.label_encoders[col] = le
        
        # Record cardinality for each categorical feature
        self.cardinalities = [len(le.classes_) for le in self.label_encoders.values()]
        
        # Fit label encoder for target variable
        self.y_encoder.fit(y)
        self.num_cls = len(self.y_encoder.classes_)
        return X_clean, y

    def transform(self, X, y=None):

        X_num = self.scaler.transform(X[self.num_features])
        
        X_cat = X[self.cat_features].copy()
        
        for col, le in self.label_encoders.items():
            X_cat[col] = le.transform(X_cat[col])
        
        y_transformed = self.y_encoder.transform(y) if y is not None else None
        
        return X_num, X_cat.values, y_transformed

    def inverse_transform_y(self, y_encoded):
        return self.y_encoder.inverse_transform(y_encoded)

def ft_encoder(X):
    d_embedding = 512
    # Numerical Embeddings
    numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
    X_numerical = X[numerical_features].values
    scaler = StandardScaler()
    X_numerical = scaler.fit_transform(X_numerical)
    X_tensor = torch.tensor(X_numerical, dtype=torch.float32)
    n_features = X_tensor.shape[1]
    model = LinearEmbeddings(n_features=n_features, d_embedding=d_embedding)
    numerical_output = model(X_tensor)

    # Categorical Embeddings
    categorical_features = X.select_dtypes(include=['category', 'object']).columns
    X_encoded = X.copy()
    for col in categorical_features:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col])
    X_encoded = X_encoded.to_numpy()
    X_categorical = X_encoded[:, X.columns.isin(categorical_features)]
    cardinalities = [int(np.max(X_categorical[:, i]) + 1) for i in range(X_categorical.shape[1])]
    model = CategoricalEmbeddings(cardinalities=cardinalities, d_embedding=d_embedding)
    X_categorical_tensor = torch.tensor(X_categorical, dtype=torch.long)
    categorical_output = model(X_categorical_tensor)
    
    combined_features = torch.cat([categorical_output, numerical_output], dim=1)
    
    output = combined_features.sum(dim=1)
    
    return output

def prepare_data(X,y,single_eval_pos, device):
    X_encoded = ft_encoder(X).to(device)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y) 
    y_encoded = torch.tensor(y_encoded,dtype=torch.float32).to(device)
    y_encoded = y_encoded.unsqueeze(1)
    
    y_data = y_encoded[:single_eval_pos]
    y_query = y_encoded[single_eval_pos:]
    
    return X_encoded,y_data,y_query






def model_training(X,y, device):
    
    train_dataloader = 

   
    
    
    classifier = TabPFNClassifier(device='cpu', N_ensemble_configurations=1)
    model = classifier.model[2]
    model.to(device)
    model.eval()

    criterion = CrossEntropyLoss()
    optimiser = Adam(model.parameters())

class FT_Dataset(Dataset):
    def __init__(self, X, y, train=True, test_size=0.33, random_state=42):
        """
        Initializes the dataset by splitting into train/test, preprocessing features, and encoding labels.
        Args:
            X (pd.DataFrame): Feature dataframe.
            y (pd.Series or np.ndarray): Target variable.
            train (bool): Whether this dataset is for training or testing.
            test_size (float): Proportion of the dataset to use as test set.
            random_state (int): Random state for train/test splitting.
        """
        self.num_cls = y.nunique()
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        
        if train:
            self.X, self.y = X_train, y_train
        else:
            self.X, self.y = X_test, y_test

        # Separate numerical and categorical features
        self.numerical_features = self.X.select_dtypes(include=['float64', 'int64']).columns
        self.categorical_features = self.X.select_dtypes(include=['category', 'object']).columns

        # Preprocess numerical features
        self.scaler = StandardScaler()
        if train:
            self.X_numerical = self.scaler.fit_transform(self.X[self.numerical_features].values)
        else:
            self.X_numerical = self.scaler.transform(self.X[self.numerical_features].values)
        
        self.n_features = self.X_numerical.shape[1]
        
        # Encode categorical features
        self.label_encoders = {col: LabelEncoder() for col in self.categorical_features}
        for col in self.categorical_features:
            if train:
                self.X[col] = self.label_encoders[col].fit_transform(self.X[col])
            else:
                self.X[col] = self.label_encoders[col].transform(self.X[col])
        
        self.X_categorical = self.X[self.categorical_features].values

        
        self.cardinalities = [int(np.max(self.X_categorical[:, i]) + 1) for i in range(self.X_categorical.shape[1])]

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.y)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            tuple: (X_numerical[idx], X_categorical[idx], y[idx])
        """
        X_num = torch.tensor(self.X_numerical[idx], dtype=torch.float32)
        X_cat = torch.tensor(self.X_categorical[idx], dtype=torch.long)
        target = torch.tensor(self.y[idx], dtype=torch.float32)
        return X_num, X_cat, target
 
def main(
    X,
    y, 
    epochs,
    save_path = "checkpoints/ft_tabpfn.pt"
):  
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    preprocessor = Preprocessor()
    preprocessor.fit(X_train,y_train)

    X_train_num, X_train_cat, y_train = preprocessor.transform(X_train,y_train)
    X_test_num, X_test_cat, y_test = preprocessor.transform(X_test, y_test)

    train_dataset = FT_Dataset(X_train_num, X_train_cat, y_train)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    eval_dataset = FT_Dataset(X_test_num, X_test_cat, y_test)
    eval_dataloader = DataLoader(eval_dataset, batch_size=128, shuffle=False)
    
    metadata_train = calculate_metadata(X_train_num, X_train_cat, y_train)
    metadata_test = calculate_metadata(X_test_num, X_test_cat, y_test)

    assert metadata_train == metadata_test, f"Metadata Inconsistencies in splits! metadata_train = {metadata_train} metadata_test = {metadata_test}"

    model = FT_TabPFN(
        metadata_train["n_features"],
        metadata_train["cardinalities"],
        metadata_train["num_cls"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.00001)

    train_loss_arr = []
    eval_loss_arr = []

    for e in tqdm(range(epochs), desc="Training Progress"):
        train_loss = train_one_epoch(
            train_loader,
            model,
            device,
            criterion,
            optimiser
        )
        print(f"Epoch: {e} Train Loss: {train_loss}")
        train_loss_arr.append(train_loss)

        eval_loss = evaluation(
            eval_dataloader,
            model,
            device,
            criterion
        )

        print(f"Epoch: {e} Eval Loss: {eval_loss}")
        eval_loss_arr.append(eval_loss)
    
    print("Finished Training!")

    torch.save(model.state_dict(), save_path)
    print("Model saved!")

    plot_losses(
        train_loss_arr, 
        eval_loss_arr
    ) 
def main(
    X,
    y, 
    epochs,
    save_path = "checkpoints/ft_tabpfn.pt"
):  
    preprocessor = Preprocessor()
    preprocessor.fit(X,y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    X_train_num, X_train_cat, y_train = preprocessor.transform(X_train,y_train)
    X_test_num, X_test_cat, y_test = preprocessor.transform(X_test, y_test)

    train_dataset = FT_Dataset(X_train_num, X_train_cat, y_train)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    eval_dataset = FT_Dataset(X_test_num, X_test_cat, y_test)
    eval_dataloader = DataLoader(eval_dataset, batch_size=128, shuffle=False)
    
    # metadata_train = calculate_metadata(X_train_num, X_train_cat, y_train)
    # metadata_test = calculate_metadata(X_test_num, X_test_cat, y_test)

    # assert metadata_train == metadata_test, f"Metadata Inconsistencies in splits! metadata_train = {metadata_train} metadata_test = {metadata_test}"

    model = FT_TabPFN(
        preprocessor.n_features,
        preprocessor.cardinalities,
        preprocessor.num_cls
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.00001)

    train_loss_arr = []
    eval_loss_arr = []

    for e in tqdm(range(epochs), desc="Training Progress"):
        train_loss = train_one_epoch(
            train_loader,
            model,
            device,
            criterion,
            optimiser
        )
        print(f"Epoch: {e} Train Loss: {train_loss}")
        train_loss_arr.append(train_loss)

        eval_loss = evaluation(
            eval_dataloader,
            model,
            device,
            criterion
        )

        print(f"Epoch: {e} Eval Loss: {eval_loss}")
        eval_loss_arr.append(eval_loss)
    
    print("Finished Training!")

    torch.save(model.state_dict(), save_path)
    print("Model saved!")

    plot_losses(
        train_loss_arr, 
        eval_loss_arr
    ) 

if __name__ == "__main__":


    df, y = fetch_openml(name='adult', version=2, return_X_y=True,as_frame=True)
    X,y = df[:500], y[:500]

    X_cleaned = X.dropna()
    y_cleaned = y.loc[X_cleaned.index]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluation(X_cleaned,y_cleaned,device)
