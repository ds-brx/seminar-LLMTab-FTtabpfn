import torch
import torch.nn as nn
import torch.nn.functional as F
from tabpfn import TabPFNClassifier

from src.modules.ft_encoders import CategoricalEmbeddings


class FT_TabPFN(nn.Module):
    def __init__(self,n_features, cardinalities, num_cls, d_embedding=512):
        super(FT_TabPFN, self).__init__()
        
        self.classifier = TabPFNClassifier()
        self.model = self.classifier.model[2]
        self.numerical_encoder = self.model.encoder
        self.catagorical_encoder = CategoricalEmbeddings(cardinalities=cardinalities, d_embedding=d_embedding)
        self.y_encoder = self.model.y_encoder
        self.transformer_encoder = self.model.transformer_encoder
        self.decoder = self.model.decoder
        
        self.num_cls = num_cls


    def forward(self, X_numerical, X_categorical, y):
        
        single_eval_pos = int(0.8 * y.shape[0])
        y_context = y[:single_eval_pos]
        y_query = y[single_eval_pos:]
                
        # numerical_features = torch.stack([self.numerical_encoder(F.pad(X_numerical[:,i].unsqueeze(1), (0,99))) for i in range(X_numerical.shape[-1])], dim=-2)
        X_numerical_padded = F.pad(X_numerical, (0, 100 - X_numerical.shape[-1]))  # Pad along last dim
        numerical_features = self.numerical_encoder(X_numerical_padded)
        
        categorical_features = self.catagorical_encoder(X_categorical)
        categorical_features = categorical_features.sum(dim=1)
        # X_encoded = torch.cat((numerical_features, categorical_features)) 

        X_encoded = torch.add(numerical_features, categorical_features)

        y_encoded = self.y_encoder(y_context)

        X_train = X_encoded[:single_eval_pos] + y_encoded[:single_eval_pos]
        src = torch.cat([X_train, X_encoded[single_eval_pos:]], 0)
        
        output = self.transformer_encoder(src)
        output = self.decoder(output)

        y_preds = output[single_eval_pos:][:, :self.num_cls]

        return y_preds, y_query.long().flatten()