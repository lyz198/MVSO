# MVSO-PPIS
## 1 Description
    Predicting protein–protein interaction (PPI) sites is essential for advancing our understanding of protein interactions,  as accurate predictions can significantly reduce experimental costs and time. While considerable progress has been made in identifying binding sites at the level of individual amino acid residues,  the prediction accuracy for residue subsequences at transitional boundaries—such as those represented by patterns like singular structures (mutation characteristics of contiguous interacting-residue segments) or edge structures (boundary transitions between Interacting/non-Interacting residue segments)—still requires improvement. To address this challenge,  we propose a novel PPI site prediction method named MVSO-PPIS. This method integrates two complementary feature extraction modules,  a subgraph-based module and an enhanced graph attention module. The extracted features are fused using an attention-based fusion mechanism,  producing a composite representation that captures both local protein substructures and global contextual dependencies. MVSO-PPIS is trained to jointly optimize three objectives: overall PPI site prediction accuracy,  edge structural consistency,  and recognition of unique structural patterns in PPI site sequences. Experimental results on benchmark datasets demonstrate that MVSO-PPIS outperforms existing baseline models in both accuracy and structural interpretability. 
## 2 Installation
### 2.1 system requirements
  For fast prediction and training process, we recommend using a GPU. To use MVSO-PPIS with GPUs, you will need: cuda >= 10.1
### 2.2 virtual environment requirements
    (1) python 3.6
    (2) torch-1.7.0+cu101
    (3) torchaudio-0.7.0
    (4) torchvision-0.8.0
    (5) dgl_cu101.0.7.0
    (6) cudatoolkit-10.1.168
    (7) pandas
    (8) sklearn
## 3 Datasets
  The files in "./Dataset" include the datasets used in this experiment(Test_315-28.pkl, Test_60.pkl, Train_335.pkl, UBtest_31-6.pkl, bound_unbound_mapping31-6.txt, PP-1001_Train.txt, PP-250_Test.txt).<br>
  All the processed pdb files of the protein chains used in this experiment are put in the directory "./Dataset/pdb/".
## 4 Features
    The extracted features are in the directory "./Feature". The specific meanings are listed as follows.
        (1) distance_map_C: using the centroid of residues as the pseudo position of resiudes, and using them to calculate the distance matrix of the protein chain.
        (2) distance_map_CA: using the position of the alpha-C atom of residues as the pseudo position of resiudes, and using them to calculate the distance matrix of the protein chain.
        (3) distance_map_SC: using the centroid of the residue side chain as the pseudo position of resiudes, and using them to calculate the distance matrix of the protein chain.
        (4) dssp: the DSSP matrix of the protein chains used in this experiment.
        (5) hmm: the HMM matrix of the protein chains used in this experiment.
        (6) psepos: the resiude pseudo positions of the protein chains in those datasets, with SC, CA, C standing for centriod of side chain, alpha-C atom and centroid of the residue, respectively.
        (7) pssm: the PSSM matrix of the protein chains used in this experiment.
        (8) resAF: the atom features of the residues for each protein used in the experiment.
## 5 The trained model
  The models with trained parameters are put in the directory "./Model/model/" and the predicted results of the test datasets are put in the directory "./Model/result_metrics".
## 6 Usage
  The construction of the model is in the "MVSOPPIS_model.py".<br>
  You can run "train.py" to train the deep model from stratch and use the "test.py" to test the test datasets with the trained model.
## 7 Access for the paper of MVSO-PPIS
  Paper title: "MVSO-PPIS:A Structured Objective Learning Model for Protein-Protein Interaction Sites Prediction via Multi-View Graph Information Integration". <br>
  Figshare link: https://doi.org/10.6084/m9.figshare.29580308.v1  
  Paper link: https://doi.org/
  
