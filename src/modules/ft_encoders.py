import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor
from typing import List

"""
Citation:
@inproceedings{gorishniy2021revisiting,
    title={Revisiting Deep Learning Models for Tabular Data},
    author={Yury Gorishniy and Ivan Rubachev and Valentin Khrulkov and Artem Babenko},
    booktitle={{NeurIPS}},
    year={2021},
}
"""

class CategoricalEmbeddings(nn.Module):
    def __init__(self, cardinalities: List[int], d_embedding: int, bias: bool = True):
        """
        Args:
            cardinalities: the number of distinct values for each feature.
            d_embedding: the embedding size.
            bias: if `True`, for each feature, a trainable vector is added to the
                embedding regardless of a feature value. For each feature, a separate
                non-shared bias vector is allocated.
                In the paper, FT-Transformer uses `bias=True`.
        """
        super().__init__()
        if not cardinalities:
            raise ValueError('cardinalities must not be empty')
        if any(x <= 0 for x in cardinalities):
            i, value = next((i, x) for i, x in enumerate(cardinalities) if x <= 0)
            raise ValueError(
                'cardinalities must contain only positive values, '
                f'however: cardinalities[{i}]={value}'
            )
        if d_embedding <= 0:
            raise ValueError(f'd_embedding must be positive, however: {d_embedding}')

        # Remove the NaN token handling; use cardinalities as given.
        self.embeddings = nn.ModuleList(
            [nn.Embedding(x, d_embedding) for x in cardinalities]
        )
        
        self.bias = (
            Parameter(torch.empty(len(cardinalities), d_embedding)) if bias else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_rsqrt = self.embeddings[0].embedding_dim ** -0.5
        for m in self.embeddings:
            nn.init.uniform_(m.weight, -d_rsqrt, d_rsqrt)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -d_rsqrt, d_rsqrt)

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        if x.ndim < 2:
            raise ValueError(
                f'The input must have at least two dimensions, however: {x.ndim}'
            )
        n_features = len(self.embeddings)
        if x.shape[-1] != n_features:
            raise ValueError(
                'The last input dimension (the number of categorical features) must be '
                f'equal to the number of cardinalities passed to the constructor. '
                f'However: {x.shape[-1]}, len(cardinalities)={n_features}'
            )

        # Directly use the input values (assumed to be non-NaN)
        x = torch.stack(
            [self.embeddings[i](x[..., i].long()) for i in range(n_features)], dim=-2
        )
        
        if self.bias is not None:
            x = x + self.bias
        
        return x

    def orthogonal_regularization_bias(self, regularization_strength: float = 1e-5) -> Tensor:
        if self.bias is None:
            return torch.tensor(0.0, device=self.bias.device)
        
        # Gram matrix of the bias vectors
        bias_gram = torch.matmul(self.bias, self.bias.T)
        
        # Identity matrix
        identity = torch.eye(self.bias.size(0), device=self.bias.device)
        
        # Frobenius norm of the difference
        orthogonal_loss = torch.norm(bias_gram - identity, p='fro')
        
        return regularization_strength * orthogonal_loss

class LinearEmbeddings(nn.Module):
    """Linear embeddings for continuous features.

    **Shape**

    - Input: `(*, n_features)`
    - Output: `(*, n_features, d_embedding)`

    **Examples**

    >>> batch_size = 2
    >>> n_cont_features = 3
    >>> x = torch.randn(batch_size, n_cont_features)
    >>> d_embedding = 4
    >>> m = LinearEmbeddings(n_cont_features, d_embedding)
    >>> m(x).shape
    torch.Size([2, 3, 4])
    """

    def __init__(self, n_features: int, d_embedding: int) -> None:
        """
        Args:
            n_features: the number of continuous features.
            d_embedding: the embedding size.
        """
        if n_features <= 0:
            raise ValueError(f'n_features must be positive, however: {n_features}')  # Updated for Python 3.7
        if d_embedding <= 0:
            raise ValueError(f'd_embedding must be positive, however: {d_embedding}')  # Updated for Python 3.7

        super().__init__()
        self.weight = Parameter(torch.empty(n_features, d_embedding))
        self.bias = Parameter(torch.empty(n_features, d_embedding))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_rqsrt = self.weight.shape[1] ** -0.5
        nn.init.uniform_(self.weight, -d_rqsrt, d_rqsrt)
        nn.init.uniform_(self.bias, -d_rqsrt, d_rqsrt)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 2:
            raise ValueError(
                f'The input must have at least two dimensions, however: {x.ndim}'  # Updated for Python 3.7
            )

        x = x[..., None] * self.weight
        x = x + self.bias[None]
        return x