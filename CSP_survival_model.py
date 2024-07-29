import numpy as np
import random
import torch
import os
import pandas as pd

torch.manual_seed(123)
np.random.seed(123)
random.seed(123)
os.environ['PYTHONHASHSEED']=str(123)
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

cGAS_STING_centric_pathways=pd.read_csv("Deep_learning_input_pathways.csv")
cGAS_STING_centric_pathways_copy=cGAS_STING_centric_pathways.copy()
cGAS_STING_centric_pathways_copy[cGAS_STING_centric_pathways_copy.isnull()]="null_gene"

gene_set=[]
for i in range(cGAS_STING_centric_pathways_copy.shape[1]):
    gene_set=gene_set+list(cGAS_STING_centric_pathways_copy.iloc[:,i])

all_gene_mean_expression = 4992.7214 #all_gene_mean_expression is used for data correction.

class MultitaskDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labelsA, labelsB):
        self.inputs = inputs
        self.labelsA = labelsA
        self.labelsB = labelsB
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        inputs = self.inputs[index]
        labelsA = self.labelsA[index]
        labelsB = self.labelsB[index]
        return inputs, labelsA, labelsB

import torch.nn as nn
import torch.nn.functional as F

class GeneTransformer(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads, dropout, is_TPM_format):
        super(GeneTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(hidden_size, num_heads, dropout) 
            for _ in range(num_layers)])
        
        #Average
        self.avg=nn.AvgPool2d(kernel_size=(1,200))
        
        # Output layer
        self.output_layer0 = nn.Linear(7, 4)
        self.output_layer1 = nn.Linear(4, 1)
        
    def forward(self, x):
        # Embedding layer
        x=x.view(-1,7,self.hidden_size)
        x1=x
        x_copy=x
        if is_TPM_format: x1=torch.log2(x1+1)
        gene_ratio=torch.sum(x_copy.reshape(-1))/all_gene_mean_expression
        x1=x1/gene_ratio
        x1=nn.LayerNorm(self.hidden_size)(x1)

        # Transformer layers
        for layer in self.transformer_layers:
            x1= layer(x1)

        x1 = self.avg(x1)
        x1 =x1.view(x.size(0),-1)
        hidden_x1 = x1
        x1 = self.output_layer0(x1)
        x1 = nn.ReLU()(x1)
        x1 = self.output_layer1(x1)
        x1 = 4*nn.Tanh()(x1)
        x1 = torch.sigmoid(x1)

        return x1

    
class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super(TransformerLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout

        # Multi-head attention layer
        self.multi_head_attention = nn.MultiheadAttention(hidden_size, num_heads, dropout,batch_first=True)

        # Feedforward layer
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size))

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        


    def forward(self, x):
        # Multi-head attention layer
        residual = x
        x, _ = self.multi_head_attention(x, x, x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer_norm1(x + residual)

        # Feedforward layer
        residual = x
        x = self.feedforward(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer_norm2(x + residual)

        return x
    

hidden_size = 200
num_layers = 2
num_heads = 4
dropout = 0.2
is_TPM_format=True

gene_transformer = GeneTransformer(hidden_size, num_layers, num_heads, dropout, is_TPM_format)

# initialization
net = gene_transformer
net=torch.load("Trained_deep_learning_model.pth")

cGAS_STING_centric_pathways_copy1=cGAS_STING_centric_pathways.copy()
cGAS_STING_centric_pathways_copy2=np.array(cGAS_STING_centric_pathways_copy1.T)
cGAS_STING_centric_pathways_copy3=pd.DataFrame(cGAS_STING_centric_pathways_copy2.reshape(-1))
cGAS_STING_centric_pathways_copy4=np.array(cGAS_STING_centric_pathways_copy3.dropna())

gene_expression_test_data=pd.read_csv("Input_File.csv",index_col=0) #Please ensure that the gene expression format is TPM.
gene_expression_test_data1=gene_expression_test_data.T
gene_expression_test_data1["null_gene"]=0
gene_expression_test_data2=pd.DataFrame(gene_expression_test_data1,index=gene_expression_test_data1.index,columns=gene_expression_test_data1.columns)
gene_expression_test_data2_copy=gene_expression_test_data2.copy()
selected_genes_raw=np.array(gene_set)
selected_genes_raw1=selected_genes_raw.reshape(-1,200)
selected_genes_dropna=cGAS_STING_centric_pathways_copy4.reshape(-1)
not_available_genes=list(set(selected_genes_dropna)-set(gene_expression_test_data.index))
for gene_vector in selected_genes_raw1:
    for not_available_gene in not_available_genes:
        if not_available_gene in gene_vector:
            gene_expression_test_data2[not_available_gene]=np.power(2,pd.DataFrame(np.log2(gene_expression_test_data2_copy[list(set(gene_vector)-set(not_available_genes))]+1)).mean(axis=1))-1
gene_expression_test_data3=gene_expression_test_data2.loc[:,selected_genes_raw]
gene_expression_test_data4=np.array(gene_expression_test_data3)
gene_expression_test_data5=torch.tensor(np.array(gene_expression_test_data4))
test_OS=torch.zeros(gene_expression_test_data5.shape[0],1) #OS placeholders
test_OS=test_OS.view(-1)
test_OS_time=torch.ones(gene_expression_test_data5.shape[0],1) #OS time placeholders
test_OS_time=test_OS_time.view(-1)
testset=MultitaskDataset(gene_expression_test_data5,test_OS,test_OS_time)

test_loader = torch.utils.data.DataLoader(testset,batch_size=test_OS.shape[0])

total=0
correct=0
net.eval()
with torch.no_grad():
    for data in test_loader:
        inputs, OS, OS_time = data
        inputs = inputs.to(torch.float32)
        outputs = net(inputs)

result=pd.DataFrame({"risk_probability":np.array(outputs.view(-1))},index=gene_expression_test_data.columns)
result["binary_risk"]=np.where(result['risk_probability']>0.5,"high","low")
result.to_csv("Output_flie.csv",index=True)
