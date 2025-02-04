# load data

# genetic perturbations
chem1_syd_cds = load_monocle_objects("/net/seahub_zfish/vol1/data/annotated/v2.2.0/CHEM1_SYD/CHEM1_SYD_projected_cds_v2.2.0/", matrix_control = list(matrix_class="BPCells", matrix_path="/net/trapnell/vol1/home/elizab9/tmp_files/nobackup/"))
chem3_atoh7_cds = load_monocle_objects("/net/seahub_zfish/vol1/data/annotated/v2.2.0/ATOH7/CHEM3_ATOH7_projected_cds_v2.2.0/", matrix_control = list(matrix_class="BPCells", matrix_path="/net/trapnell/vol1/home/elizab9/tmp_files/nobackup/"))
gap16_cds = load_monocle_objects("/net/seahub_zfish/vol1/data/annotated/v2.2.0/GAP16/GAP16_projected_cds_v2.2.0/", matrix_control = list(matrix_class="BPCells", matrix_path="/net/trapnell/vol1/home/elizab9/tmp_files/nobackup/"))
kmt_cds = load_monocle_objects("/net/seahub_zfish/vol1/data/preprocessed/KMT/KMT/", matrix_control = list(matrix_class="BPCells", matrix_path="/net/trapnell/vol1/home/elizab9/tmp_files/nobackup/"))
lmx1b_cds = load_monocle_objects("/net/seahub_zfish/vol1/data/annotated/v2.2.0/LMX1B/LMX1B_projected_cds_v2.2.0/", matrix_control = list(matrix_class="BPCells", matrix_path="/net/trapnell/vol1/home/elizab9/tmp_files/nobackup/"))
prdm_cds = load_monocle_objects("/net/seahub_zfish/vol1/data/annotated/v2.2.0/PRDM/PRDM_projected_cds_v2.2.0/", matrix_control = list(matrix_class="BPCells", matrix_path="/net/trapnell/vol1/home/elizab9/tmp_files/nobackup/"))
HF5_cds = load_monocle_objects("/net/seahub_zfish/vol1/data/annotated/v2.2.0/HF5/HF5_projected_cds_v2.2.0/", matrix_control = list(matrix_class="BPCells", matrix_path="/net/trapnell/vol1/home/elizab9/tmp_files/nobackup/"))
GENE1_cds = load_monocle_objects("/net/seahub_zfish/vol1/data/annotated/v2.2.0/GENE1/GENE1_projected_cds_v2.2.0/", matrix_control = list(matrix_class="BPCells", matrix_path="/net/trapnell/vol1/home/elizab9/tmp_files/nobackup/"))
GENE2_cds = load_monocle_objects("/net/seahub_zfish/vol1/data/annotated/v2.2.0/GENE2/GENE2_projected_cds_v2.2.0/", matrix_control = list(matrix_class="BPCells", matrix_path="/net/trapnell/vol1/home/elizab9/tmp_files/nobackup/"))
GENE3_cds = load_monocle_objects("/net/seahub_zfish/vol1/data/annotated/v2.2.0/GENE3/GENE3_projected_cds_v2.2.0/", matrix_control = list(matrix_class="BPCells", matrix_path="/net/trapnell/vol1/home/elizab9/tmp_files/nobackup/"))

# chemical and environmental perturbations
c646_cds = load_monocle_objects("/net/seahub_zfish/vol1/data/annotated/v2.2.0/C646/C646_projected_cds_v2.2.0/", matrix_control = list(matrix_class="BPCells", matrix_path="/net/trapnell/vol1/home/elizab9/tmp_files/nobackup/"))
chem1.0_cds = load_monocle_objects("/net/seahub_zfish/vol1/data/annotated/v2.2.0/CHEM1.0/CHEM1.0_projected_cds_v2.2.0/", matrix_control = list(matrix_class="BPCells", matrix_path="/net/trapnell/vol1/home/elizab9/tmp_files/nobackup/"))
chem1.1_cds = load_monocle_objects("/net/seahub_zfish/vol1/data/annotated/v2.2.0/CHEM1.1/CHEM1.1_projected_cds_v2.2.0/", matrix_control = list(matrix_class="BPCells", matrix_path="/net/trapnell/vol1/home/elizab9/tmp_files/nobackup/"))
chem9_cds = load_monocle_objects("/net/seahub_zfish/vol1/data/annotated/v2.2.0/CHEM9/CHEM9_projected_cds_v2.2.0/", matrix_control = list(matrix_class="BPCells", matrix_path="/net/trapnell/vol1/home/elizab9/tmp_files/nobackup/"))
HF_cds = load_monocle_objects("/net/seahub_zfish/vol1/data/annotated/v2.2.0/HF/HF_projected_cds_v2.2.0/", matrix_control = list(matrix_class="BPCells", matrix_path="/net/trapnell/vol1/home/elizab9/tmp_files/nobackup/"))

# reference
REF1_cds = load_monocle_objects("/net/seahub_zfish/vol1/data/annotated/v2.2.0/REF1/REF1_projected_cds_v2.2.0", matrix_control = list(matrix_class="BPCells", matrix_path="/net/trapnell/vol1/home/elizab9/tmp_files/nobackup/"))
REF2_cds = load_monocle_objects("/net/seahub_zfish/vol1/data/annotated/v2.2.0/REF2/REF2_projected_cds_v2.2.0", matrix_control = list(matrix_class="BPCells", matrix_path="/net/trapnell/vol1/home/elizab9/tmp_files/nobackup/"))
REF_cds = load_monocle_objects("/net/seahub_zfish/vol1/data/reference_cds/v2.2.0/reference_cds_fixed_colData_updated_241203/", matrix_control = list(matrix_class="BPCells", matrix_path="/net/trapnell/vol1/home/elizab9/tmp_files/nobackup/"))