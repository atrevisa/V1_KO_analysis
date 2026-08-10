# V1 Interneuron snRNA-seq  KO analysis

## Overview

This repository contains the code used to generate the single-nucleus RNA sequencing (snRNA-seq) atlas of mouse spinal V1 interneurons in global global En1 KO P0 pups and littermate controls. 

Trevisan AJ, Han K, Chapman P, Kulkarni AS, Hinton JM, Ramirez C, Klein I, Gatto G, Gabitto MI, Menon V, Bikoff JB. The transcriptomic landscape of spinal V1 interneurons reveals a role for En1 in specific elements of motor output. bioRxiv [Preprint]. 2024 Oct 26:2024.09.18.613279. doi: 10.1101/2024.09.18.613279. PMID: 39345580; PMCID: PMC11429899.

The analysis workflow includes quality control, data integration, clustering, cell-type annotation, differential gene expression analysis, and generation of figures used in the manuscript.


---

## Associated Publication

**Preprint:** 

Trevisan AJ, Han K, Chapman P, Kulkarni AS, Hinton JM, Ramirez C, Klein I, Gatto G, Gabitto MI, Menon V, Bikoff JB. The transcriptomic landscape of spinal V1 interneurons reveals a role for En1 in specific elements of motor output. bioRxiv [Preprint]. 2024 Oct 26:2024.09.18.613279. doi: 10.1101/2024.09.18.613279. PMID: 39345580; PMCID: PMC11429899.

**Final publication:**

Trevisan, A.J., Han, K., Chapman, P.D. et al. Transcriptomic analysis of spinal V1 interneurons informs their multifunctional role in motor output. Nat Commun (2026). https://doi.org/10.1038/s41467-026-76522-3


---

## Data Availability

Raw sequencing data, count matrices, and associated metadata are available through the Gene Expression Omnibus (GEO).

**GEO Accession:** GSE275595

**Link to GEO page:** <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE275595>

Input data should be downloaded and placed in the appropriate data directories prior to executing the analysis scripts.

---

## Analysis Summary

The workflow performs:

1. Import and preprocessing of snRNA-seq count matrices
2. Quality-control filtering of nuclei
3. Removal of ambient RNA contamination
4. Doublet identification and exclusion
5. Normalization and scaling
6. Batch correction and dataset integration
7. Dimensionality reduction and clustering
8. Cell-type annotation and marker identification


---

## Repository-to-Manuscript Mapping

The figures below were generated using the code in this repository

**Figure 6B - 6E:** Analysis of En1 control and knock-out (KO) V1 nuclei

**Supplementary Figure 7B, 7E:** Differential gene expression analysis between En1 control and En1 KO V1 nuclei


---

## Software Requirements

Analyses were performed using:

* R (version specified in SessionInfo)
* Python
* Seurat
* Scanpy

Additional package versions and dependency information are provided in the accompanying SessionInfo files.

---

## Running the Analysis

Scripts are intended to be run sequentially according to their filenames.

A typical workflow consists of:

1. Data preprocessing and quality control
2. Integration and clustering
3. Cell-type annotation
4. Differential expression analysis
5. Figure generation

Refer to individual scripts for required inputs and outputs.

---

## Expected Outputs

The workflow generates:

* Processed Seurat objects
* Cluster annotations
* Marker gene tables
* Differential expression results


---

## Reproducibility

All analyses were conducted using the software versions documented in the provided SessionInfo files.

Minor differences in numerical values or visualization appearance may occur when using different package versions or computing environments.

---

## Contact

Questions regarding the analysis workflow may be directed to:

**Jay B. Bikoff**

<Jay.Bikoff@STJUDE.ORG>  
Department of Developmental Neurobiology  
St. Jude Children's Research Hospital  
Memphis, TN, USA

---

## License

This repository is provided for academic and research use.

If you use this code or derived results, please cite the associated publication.
