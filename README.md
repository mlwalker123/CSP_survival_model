# CSP_survival_model
Deep learning survival model for hepatocellular carcinoma patients based on cGAS-STING-centric pathways
## Step 1: Configure  the  working  environment.
        We recommend the use of conda for managing Python version and Python module versions.
        python 3.8.16  
        numpy 1.24.3  
        torch 2.0.0  
        pandas 2.0.1  
## Step 2: Establish a new working directory including following files.
        /new working directory  
          |_ CSP_survival_model.py  
          |_ Deep_learning_input_pathways.csv  
          |_ Input_File.csv  
          |_ Trained_deep_learning_model.pth  
       Tip: The "Input_File.csv" document is fundamentally a gene expression matrix (TPM format);    
            when using it, please replace the provided file with your own.  
## Step 3: Directly execute the Python script.
        python CSP_survial_model.py  
## Step 4: Inspect the output file.
After executing the aforementioned Python script, you will obtain a file named "Output_file.csv"; "risk_probability" denotes the survival risk   probability of patients with hepatocellular carcinoma, with higher risk probability indicating poorer prognosis. We employ 0.5 as the threshold to    determine the survival risk probability of patients with hepatocellular carcinoma.  

## Tip: our code and data are only for academic researches.


