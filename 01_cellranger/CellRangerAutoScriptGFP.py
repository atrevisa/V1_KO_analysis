#!/usr/bin/env python3

import pandas as pd, os, sys, argparse

#for line in open(sys.argv[1], "r"):
metadata = pd.read_csv(sys.argv[1])

for row in range(len(metadata)):
    SAMPLE = str(metadata.loc[row,"SRM_sample_number"]) + "_" + str(metadata.loc[row,"Submitted_sample_name"])
    ID = str(metadata.loc[row,"Actual_file_name"]) + "_cellranger"
    FASTQ = str(metadata.loc[row,"File_path"])
    REF_PATH = "/research/groups/bikoffgrp/home/atrevisa/RNA/Analysis/01_Cellranger/refdata-gex-mm10-2020-A_GFP"
    # SingleCommand = 'bsub -P CellRangerCount -J CellRangerCount_' + ID + ' -q standard -o /research/groups/bikoffgrp/home/atrevisa/RNA/Analysis/KO_analysis/01_cellranger/cellranger_outs/' + ID + '.out -e /research/groups/bikoffgrp/home/atrevisa/RNA/Analysis/KO_analysis/01_cellranger/cellranger_errors/' + ID + '.err -n 4 -R "rusage[mem=4000] span[hosts=1]" ' + 'cellranger count --id=' + ID + ' --fastqs=' + FASTQ + ' --sample=' + SAMPLE + ' --transcriptome=' + REF_PATH + ' --force-cells=9000 --jobmode=lsf'
    SingleCommand = 'bsub -P CellRangerCount -J CellRangerCount_' + ID + ' -q standard -o /research/groups/bikoffgrp/home/atrevisa/RNA/Analysis/KO_analysis/01_cellranger/cellranger_outs/' + ID + '.out -e /research/groups/bikoffgrp/home/atrevisa/RNA/Analysis/KO_analysis/01_cellranger/cellranger_errors/' + ID + '.err -n 4 -R "rusage[mem=4000] span[hosts=1]" ' + 'cellranger count --id=' + ID + ' --fastqs=' + FASTQ + ' --sample=' + SAMPLE + ' --transcriptome=' + REF_PATH + ' --jobmode=lsf'
    
    print(SingleCommand)
    os.system(SingleCommand)
    

