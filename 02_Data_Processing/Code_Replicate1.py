#!/usr/bin/env python
# coding: utf-8

# # P0 Het and Ko analysis Replicate 1
# AT Updated 8/8/24



# ## Section 1: Load in data and concatenate datasets if applicable



# Import libraries

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings("ignore") # suppress warnings

get_ipython().run_line_magic('matplotlib', 'inline')

sc.settings.verbosity = 3             
sc.logging.print_header()

sc.settings.set_figure_params(format = "pdf", vector_friendly = False, dpi_save = 80, transparent = True)
sc.set_figure_params(vector_friendly = False, format = "eps", dpi = 80, dpi_save = 80, transparent = True)
plt.rcParams['svg.fonttype'] = 'none'



results_file = 'write/P0KOR1.h5ad' 




# Read in control data from replicate 1

adata_het = sc.read_10x_mtx('/Users/atrevisa/Desktop/KO_sc/ForceCells/P0HetR1_filtered_feature_bc_matrix',
                            var_names='gene_symbols',cache=True)
adata_het.var_names_make_unique()
adata_het





# Read in Ko data from replicate 1
adata_ko = sc.read_10x_mtx('/Users/atrevisa/Desktop/KO_sc/ForceCells/P0KoR1_filtered_feature_bc_matrix',
                            var_names='gene_symbols',cache=True)
adata_ko.var_names_make_unique()
adata_ko





# Merge the data (no batch correction)

adata_all = adata_het.concatenate(adata_ko, batch_categories=['het', 'ko'])
adata_all


# ## Section 2: Define functions used for analaysis




def QC_graphs (adata):
    
    # QC graphs generates graphs of basic QC data including the highest expressing genes, total counts, genes, % mt genes
    # example
    # Q_graphs(adata_all)
    # if you have not yet run preprocessing1 you will only see the top genes and then get an error
    # run preprocessing1 on the data to get the other QC graphs

    # Graph highest expressing genes overall
    sc.pl.highest_expr_genes (adata, n_top = 20)
    
    # If you have not yet run preprocessing1 the following will give an error because these values have not been calculated yet
    if 'batch' in adata.obs.columns:
        sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], groupby = 'batch',jitter=0.4, multi_panel=True, size = 0)
        sc.pl.scatter(adata, x='log1p_total_counts', y='log1p_pct_counts_mt', color = 'batch')
        sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', color = 'batch')
        sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', color = 'batch')
        sc.pl.scatter(adata, x='log1p_total_counts', y='log1p_n_genes_by_counts', color = 'batch')
    
    else:
        sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True, size = 0)
        sc.pl.scatter(adata, x='log1p_total_counts', y='log1p_pct_counts_mt')
        sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
        sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')
        sc.pl.scatter(adata, x='log1p_total_counts', y='log1p_n_genes_by_counts')





def preprocessing1 (adata):
    
    # preprocessing 1 elimates genes with less than 3 counts and cells with less than 200 reads


    # preprocessing 1 also calculates QC stats
    # example
    # preprocessing(adata_all)
    
    # First compute the number of genes per cell and add to adata.obs in a column called 'n_genes'
    sc.pp.filter_cells(adata, min_genes=0, inplace = True) 
    # Manually filter 
    adata_copy = adata[adata.obs['n_genes'] >= 1000, :]
    
    # First compute the number of cells expressing each gene and add to adata.var in a column called 'n_cells'
    sc.pp.filter_genes(adata_copy, min_cells=0, inplace = True)
    # Manually filter
    adata_copy = adata_copy[:, adata_copy.var['n_cells'] >= 10]
    
    # Calculate qc statistics
    # note that log1p is the natural log
    adata_copy.var['mt'] = adata_copy.var_names.str.startswith('mt-')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(adata_copy, qc_vars=['mt'], percent_top=None, log1p=True, inplace=True)
    adata_copy.obs['log1p_pct_counts_mt'] = np.log1p(adata_copy.obs.pct_counts_mt)

    return(adata_copy)





def preprocessing2 (adata):
    
    # preprocessing2 eleminates cells above and below 2 MADs from the median of the natiral log total counts and genes
    # preprocessing2 also gets rid of cells with > 5% mt genes expressed
    # example
    # adata_filtered = preprocessing2(adata_all)
    
    if 'batch' in adata.obs.columns:
        cell_ID2 = []

        for batch in adata.obs.batch.unique():
            print(batch)
            currentbatch = adata[adata.obs.batch == batch]
            print(currentbatch)
    
            upper_counts = currentbatch.obs.log1p_total_counts.median() + 2.5*stats.median_abs_deviation(currentbatch.obs.log1p_total_counts)
            lower_counts = currentbatch.obs.log1p_total_counts.median() - 2.5*stats.median_abs_deviation(currentbatch.obs.log1p_total_counts)
            upper_genes = currentbatch.obs.log1p_n_genes_by_counts.median() + 2.5*stats.median_abs_deviation(currentbatch.obs.log1p_n_genes_by_counts)
            lower_genes = currentbatch.obs.log1p_n_genes_by_counts.median() - 2.5*stats.median_abs_deviation(currentbatch.obs.log1p_n_genes_by_counts)

            currentbatch = currentbatch[currentbatch.obs.log1p_total_counts < upper_counts, :] 
            currentbatch = currentbatch[currentbatch.obs.log1p_total_counts > lower_counts, :] 
            currentbatch = currentbatch[currentbatch.obs.log1p_n_genes_by_counts > lower_genes, :] 
            currentbatch = currentbatch[currentbatch.obs.log1p_n_genes_by_counts < upper_genes, :] 
            
            currentbatch = currentbatch[currentbatch.obs.pct_counts_mt < 5, :]
    
            cell_ID2.extend(currentbatch.obs.index.tolist())
        
        #len(cell_ID2)
        cell_ID2_series = pd.Series(cell_ID2)
        adata = adata[cell_ID2_series]
        
        return(adata)
    
    # use the following code for only 1 batch - significantly simpler
    else:
        upper_counts = adata.obs.log1p_total_counts.median() + 2.5*stats.median_abs_deviation(adata.obs.log1p_total_counts)
        lower_counts = adata.obs.log1p_total_counts.median() - 2.5*stats.median_abs_deviation(adata.obs.log1p_total_counts)
        upper_genes = adata.obs.log1p_n_genes_by_counts.median() + 2.5*stats.median_abs_deviation(adata.obs.log1p_n_genes_by_counts)
        lower_genes = adata.obs.log1p_n_genes_by_counts.median() - 2.5*stats.median_abs_deviation(adata.obs.log1p_n_genes_by_counts)
        
        adata = adata[adata.obs.log1p_total_counts < upper_counts, :] 
        adata = adata[adata.obs.log1p_total_counts > lower_counts, :] 
        adata = adata[adata.obs.log1p_n_genes_by_counts > lower_genes, :] 
        adata = adata[adata.obs.log1p_n_genes_by_counts < upper_genes, :] 
        
        adata = adata[adata.obs.pct_counts_mt < 5, :]
        
        return(adata)





def normalization (adata, regress=None):
    
    # normalization normalizes reads to total counts per cell, logarithmizes the data, identifies HVG, and scales the data
    # optionally, normalization can regress out variables of choice using the optional argument regress

    # options for regression
    # Continuous variables: ['total_counts', 'pct_counts_mt', 'n_genes_by_counts']
    # Categorical variable such as ['batch'] but this cannot be combined with continuous variables

    # you cannot run normalization twice on the same dataset

    # examples
    # adata_filtered = normalization(adata_filtered) # no regression
    # adata_filtered = normalization(adata_filtered, ['batch'])
    # adata_filtered = normalization(adata_filtered, ['total_counts', 'pct_counts_mt', 'n_genes_by_counts'])
    
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    
    # IF YOU HAVE ERRORS DO NOT RUN THE NEXT LINE
    adata.raw = adata # this may mess up some heatmaps later
    
    adata = adata[:, adata.var.highly_variable]
    
    if regress is not None:
        sc.pp.regress_out(adata, regress)
    
    sc.pp.scale(adata, max_value=10)
    
    return(adata)





def pca (adata):
    
    # pca performs pca dimensionality reduction on the data and outputs several graphs to reflect this
    # example
    # pca(adata_filtered)
    
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pl.pca(adata, color='total_counts')
    sc.pl.pca(adata, color='batch')
    sc.pl.pca_variance_ratio(adata, log=True)
    sc.pl.pca_loadings(adata) # genes contributing to each PC





def cluster (adata, neighbors, pcs, res):
    
    # cluster performs non-linear dimensionality reduction and clustering
    # example
    # cluster(adata_filtered, neighbors = 30, pcs = 15, res = 1)
    
    sc.pp.neighbors(adata, n_neighbors=neighbors, n_pcs=pcs)
    
    # sc.tl.paga(adata_all_all)
    # sc.pl.paga(adata_all_all, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph
    # sc.tl.umap(adata_all_all, init_pos='paga')
    
    sc.tl.umap(adata)
    #sc.pl.umap(adata)
    
    sc.tl.leiden(adata, resolution = res) # Default resolution is 1
    sc.pl.umap(adata, color=['leiden'], legend_loc='on data')
    sc.pl.umap(adata, color=['leiden'])





def filtering (adata):
    
    # filtering shows some basic  info need for getting rid of non-V1 cells
    # example
    # filtering(adata)

    sc.pl.umap(adata, color=['total_counts', 'pct_counts_mt', 'n_genes_by_counts',
                             'log1p_total_counts', 'log1p_pct_counts_mt', 'log1p_n_genes_by_counts'], ncols=3)
    
    sc.pl.umap(adata, color=['Foxp2', 'St18', 'Calb1', 'Pou6f2', 'Sall3', 'Nr5a2', 'Rnf220', 'Bcl11a', 'Nos1', 'Piezo2', 'Ntn1', 'Sp8'], ncols=3)
    
    sc.pl.umap(adata, color=['leiden'], legend_loc='on data')


# ## Section 3: Preliminary analysis -1




adata_all


adata_filtered = preprocessing1(adata_all)


adata_filtered



QC_graphs(adata_filtered)


adata_filtered = preprocessing2(adata_filtered)


adata_filtered


QC_graphs(adata_filtered)


adata_filtered = normalization(adata_filtered)


adata_filtered





# ## Negative Selection 1




pca(adata_filtered)



cluster(adata_filtered, neighbors = 30, pcs = 15, res = 1)



sc.pl.umap(adata_filtered, color = 'batch')



filtering(adata_filtered)



# Glia

# General
# Astrocytes
# Oligodendrocyte-lineage
# Microglia
sc.pl.umap(adata_filtered, color=["Sox6",
                                 "Aldh1l1", "Fgfr3", "Aqp4", "Gfap", "Glul", "Gja1", "Slc1a3", "Slc4a4", "Sox2", "Slc1a2", "S100b", "Ndrg2", "Nkx6-1", "Sox9",
                                 "Olig2", "Mbp", "Mog", "Mag", "Bcas1", "Pdgfra", "Mbp", "Plp1", "Pmp22", "Prx", "Cspg4", "Gpr17", "Cnp", "Sox10", "Olig1", "Cd9", "Zfp488", "Zfp536",  "Nkx6-2", "Nkx2-2", "Cd82", "Mal", "Bmp4", "Aspa", "St18",
                                 "Aif1", "Trem2", "Inpp5d", "Ctss", "Itgam", "Ptprc", "Cx3cr1", "Cd68", "Adgre1", "Mertk", "Fcer1g", "Fcrls", "Hexb"], ncols = 3)





# Other non-neuronal cells

# Vasculature
# CSF-contacting cells
# Meninges
# Ependymal cells
sc.pl.umap(adata_filtered, color=["Cldn5", "Rgs5", "Flt1", 'Slco1c1', 'Fli1', "Sox17", "Fermt3", "Klf1", "Car2", "Pecam1", "Tek", "Egflam", "Dlc1", "Cald1", "Rapgef5", "Flt4",
                                 "Pkd2l1", "Pkd1l2", "Myo3b",
                                  "Dcn",  "Col3a1", "Igf2",
                                  "Sox9", "Sox2",
                                 "Dnah12", "Spef2", "Ccdc114", "Ddo", "Cfap65", "Ak9", "Fam216b", "Zfp474", "Wdr63", "Ccdc180",
                                 "Lmx1a", "Msx1", "Pax3", "Wnt1"], ncols = 3)



# Neurons

# General excitatory markers
# Cholinergic markers
# Neural Crest derived
sc.pl.umap(adata_filtered, color=["Slc17a6", "Lmx1b", "Ebf2", "Sox5", "Slc17a7", "Ebf1", "Ebf3", "Cacna2d1",
                                 "Chat", "Slc5a7", "Slc18a3", "Isl1", "Ret", "Slit3", "Prph", "Lhx3", "Isl2", "Mnx1", "Slc8a3",
                                  "Sox10", "Sox2", 
                                 "Enc1", "Dlg4", "Eno2",
                                  "Lhx1", "Lhx5", "Pax8", "Lbx1", "Pax2", "Gbx1", "Bhlhe22", "Sall3",
                                 "Gad1", "Gad2", "Slc32a1", 
                                 "Tal1", "Gata2", "Gata3",
                                 "Evx1", "Evx2",
                                 "Dmrt3", "Wt1", "En1"], ncols = 3)


# dI4
# dIlA
sc.pl.umap(adata_filtered, color=["Lhx1", "Lhx5", "Pax8", "Lbx1", "Gbx1", "Bhlhe22",
                                 "Gbx1",  "Pax2","Sall3", "Rorb", "Pdzd2"], ncols = 3)




# remove clusters

# First round contaminants
adata_filtered = adata_filtered[~adata_filtered.obs.leiden.isin(["11", "6", "17", "15", "8", "16", "13", "10", "7", "12", "8", "18", "5", "9"])]


adata_filtered


filtering(adata_filtered)








# ## Negative Selection 2




pca(adata_filtered)


cluster(adata_filtered, neighbors = 30, pcs = 15, res = 1)


sc.pl.umap(adata_filtered, color = 'batch')


filtering(adata_filtered)


# Glia

# General
# Astrocytes
# Oligodendrocyte-lineage
# Microglia
sc.pl.umap(adata_filtered, color=["Sox6",
                                 "Aldh1l1", "Fgfr3", "Aqp4", "Gfap", "Glul", "Gja1", "Slc1a3", "Slc4a4", "Sox2", "Slc1a2", "S100b", "Ndrg2", "Nkx6-1", "Sox9",
                                 "Olig2", "Mbp", "Mog", "Mag", "Bcas1", "Pdgfra", "Mbp", "Plp1", "Pmp22", "Prx", "Cspg4", "Gpr17", "Cnp", "Sox10", "Olig1", "Cd9", "Zfp488", "Zfp536",  "Nkx6-2", "Nkx2-2", "Cd82", "Mal", "Bmp4", "Aspa", "St18",
                                 "Aif1", "Trem2", "Inpp5d", "Ctss", "Itgam", "Ptprc", "Cx3cr1", "Cd68", "Adgre1", "Mertk", "Fcer1g", "Fcrls", "Hexb"], ncols = 3)


# other non-neuronal

# Vasculature
# CSF-contacting cells
# Meninges
# Ependymal cells
sc.pl.umap(adata_filtered, color=["Cldn5", "Rgs5", "Flt1", 'Slco1c1', 'Fli1', "Sox17", "Fermt3", "Klf1", "Car2", "Pecam1", "Tek", "Egflam", "Dlc1", "Cald1", "Rapgef5", "Flt4",
                                 "Pkd2l1", "Pkd1l2", "Myo3b",
                                  "Dcn",  "Col3a1", "Igf2",
                                  "Sox9", "Sox2",
                                 "Dnah12", "Spef2", "Ccdc114", "Ddo", "Cfap65", "Ak9", "Fam216b", "Zfp474", "Wdr63", "Ccdc180",
                                 "Lmx1a", "Msx1", "Pax3", "Wnt1"], ncols = 3)


# neurons

# General excitatory markers
# Cholinergic markers
# NC derived
#
sc.pl.umap(adata_filtered, color=["Slc17a6", "Lmx1b", "Ebf2", "Sox5", "Slc17a7", "Ebf1", "Ebf3", "Cacna2d1",
                                 "Chat", "Slc5a7", "Slc18a3", "Isl1", "Ret", "Slit3", "Prph", "Lhx3", "Isl2", "Mnx1", "Slc8a3",
                                  "Sox10", "Sox2", 
                                 "Enc1", "Dlg4", "Eno2",
                                  "Lhx1", "Lhx5", "Pax8", "Lbx1", "Pax2", "Gbx1", "Bhlhe22", "Sall3",
                                 "Gad1", "Gad2", "Slc32a1", 
                                 "Tal1", "Gata2", "Gata3",
                                 "Evx1", "Evx2",
                                 "Dmrt3", "Wt1", "En1"], ncols = 3)

# dI4
# dIlA
sc.pl.umap(adata_filtered, color=["Lhx1", "Lhx5", "Pax8", "Lbx1", "Gbx1", "Bhlhe22",
                                 "Gbx1",  "Pax2","Sall3", "Rorb", "Pdzd2"], ncols = 3)


adata_filtered # before removing contaminates


# remove clusters


# Second round
adata_filtered = adata_filtered[~adata_filtered.obs.leiden.isin(["9", "13", "0", "14"])]


adata_filtered


filtering(adata_filtered)




# ## Select for V1's and perform analysis again




adata_filtered


V1 = pd.DataFrame(adata_filtered.obs.index)
V1_ID = V1[0]
len(V1_ID)


adata_all


adata_V1 = adata_all[V1_ID]


adata_V1


adata_V1 = preprocessing1(adata_V1)


adata_V1


QC_graphs(adata_V1)


adata_V1 = normalization(adata_V1, ['batch']) # regress out batch effects`


adata_V1


pca(adata_V1)


cluster(adata_V1, neighbors = 50, pcs = 12, res = 1)


sc.pl.umap(adata_V1, color = 'batch') # optional



for batch in ['het', 'ko']:
    sc.pl.umap(adata_V1, color='batch', groups=[batch])


filtering(adata_V1)


