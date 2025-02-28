# FT-TabPFN: Reproduction and Analysis  

## Abstract  
FT-TabPFN is an extension of TabPFN designed to improve the handling of categorical data in tabular classification. The original paper claims that FT-TabPFN enhances robustness by introducing feature tokenization, feature identifiers, and orthogonal regularization. However, the lack of an official implementation raises questions about the reproducibility of these results. This project reimplements FT-TabPFN from scratch, evaluates its performance across multiple datasets, and compares it against the original TabPFN. Our findings reveal inconsistencies in the reported improvements, highlighting the importance of transparent benchmarking and rigorous empirical validation.  

## Introduction  
Tabular data classification remains a challenging task in deep learning, often requiring models to effectively handle both numerical and categorical features. TabPFN, a transformer-based model, has shown promise in solving small-scale tabular classification problems but struggles with categorical feature representation.  

FT-TabPFN aims to address this limitation by:  
- **Feature Tokenization:** Using a lookup table to generate embeddings for categorical features.  
- **Feature Identifiers:** Assigning feature-specific identifiers to categorical embeddings for better differentiation.  
- **Orthogonal Regularization:** Enforcing feature separation to minimize overlap in learned representations.  
- **Fusion of Numerical and Categorical Features:** Processing numerical and categorical embeddings separately before combining them.  

This project independently reproduces FT-TabPFN to verify these claims and assess its effectiveness. Through extensive testing, we compare FT-TabPFN against TabPFN and analyze its actual impact on classification performance.  

## Performance Comparison

The following plot shows the performance comparison between FT-TabPFN and TabPFN across datasets:

![Performance Comparison](images/performance_comparison.png)

## Learning Curves

The learning curves for different models over training epochs:

![Learning Curve](images/learning_curve.png)

## Code Reproduction Steps  

Follow these steps to set up and run the FT-TabPFN model:  

### **1. Clone the repository**  
`git clone https://github.com/ds-brx/seminar-LLMTab-FTtabpfn.git`
### **2. Navigate into the project folder**  
`cd seminar-LLMTab-FTtabpfn`
### **3. Create a Conda environment**  
`conda create -n FT python=3.7`
### **4. Activate the environment**  
`conda activate FT`
### **5. Install dependencies**  
`pip install -r requirements.txt`
### **6. Run the model**  
`python main.py`

This setup ensures all dependencies are installed correctly and allows you to run the model for evaluation and further analysis.  

## Results Summary  
- The reimplementation was successfully completed, adhering to the methodology described in the FT-TabPFN paper.  
- Performance evaluations showed **no consistent improvement** over TabPFN, contradicting the original paper’s claims.  
- A **significant performance drop (~25.75%)** was observed in real-world testing.  
- Hyperparameter tuning helped mitigate some losses, but **FT-TabPFN did not outperform TabPFN** even after optimization.  

## Limitations  
- **No Official Codebase:** The model was reconstructed based on the paper’s descriptions, potentially leading to minor deviations.  
- **Undocumented Hyperparameters:** Key training details (e.g., batch size) were missing, requiring assumptions.  
- **Computational Constraints:** Limited hyperparameter tuning was performed; further optimization may yield different results.  
- **Dataset Scope:** Testing was conducted on five datasets, but broader validation is needed.  
- **Evaluation Metrics:** Performance was primarily measured using ROC-AUC; additional metrics could provide deeper insights.  

## Conclusion  
This project highlights the importance of **reproducibility in machine learning research**. Despite FT-TabPFN’s proposed enhancements, our independent evaluation found no substantial benefit over TabPFN. Future research should focus on refining categorical feature handling, improving model generalization, and ensuring transparency in deep learning model evaluations.  

## References  
- Liu, Q., Yang, W., Liang, C., Pang, L., & Zou, Z. (2024). *Tokenize Features, Enhancing Tables: The FT-TabPFN Model for Tabular Classification.* arXiv preprint arXiv:2406.06891.  
- Hollmann, N., Müller, S., Eggensperger, K., & Hutter, F. (2022). *TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second.* arXiv preprint arXiv:2207.01848.  
---
