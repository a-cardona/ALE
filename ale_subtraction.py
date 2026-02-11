
from nimare.meta.cbma import ALESubtraction
from nimare.correct import FWECorrector
from nilearn.reporting import get_clusters_table
from nimare import io
import os
#path setup 
BASE = os.getcwd()
print("PWD:", BASE)

# Add dataset info check
print("\n data set info")
# Load existing datasets
attention_dset = io.convert_sleuth_to_dataset("cleaned_sleuth/Attention.txt")
cognitive_dset = io.convert_sleuth_to_dataset("cleaned_sleuth/Cognitive_PT.txt")
spatial_dset   = io.convert_sleuth_to_dataset("cleaned_sleuth/Spatial_PT_lvl2.txt")
affective_dset = io.convert_sleuth_to_dataset("cleaned_sleuth/Affective_PT.txt")

for name, dset in [("Attention", attention_dset), 
                    ("Cognitive", cognitive_dset),
                    ("Spatial", spatial_dset),
                    ("Affective", affective_dset)]:
    print(f"{name}: {len(dset.ids)} studies")
    print(f"  Coordinates shape: {dset.coordinates.shape}")
print("=" * 40 + "\n")

# %% subtraction lock in 
def run_subtraction(
    dset1, dset2, name1, name2,
    n_iters=10000,
):
    print(f"\n{'='*50}")
    print(f"Running subtraction: {name1} vs {name2}")
    print(f"Dataset 1: {len(dset1.ids)} studies")
    print(f"Dataset 2: {len(dset2.ids)} studies")
    
    sub = ALESubtraction(
        n_iters=n_iters,
    )
    sub_results = sub.fit(dset1, dset2)
    sub_corr = sub_results 
    prefix = f"{name1}_vs_{name2}"
    outdir = os.path.join(
        "/home/tur31606@tu.temple.edu/ALE/results",
        prefix
    )
    os.makedirs(outdir, exist_ok=True)
    
    print(">>> SAVING MAPS <<<")
    sub_corr.save_maps(
        output_dir=outdir,
        prefix=prefix
    )
    
    files_created = os.listdir(outdir)
    print(f"Files created: {files_created}")
    print('='*50 + '\n')
    
    return sub_corr

#
contrasts = [
    (attention_dset, cognitive_dset, "attention", "cognitive"),
    (attention_dset, spatial_dset,   "attention", "spatial"),
    (attention_dset, affective_dset, "attention", "affective"),
    (cognitive_dset, spatial_dset,   "cognitive", "spatial"),
    (cognitive_dset, affective_dset, "cognitive", "affective"),
    (spatial_dset,   affective_dset, "spatial", "affective"),
]

# naming results 
results_dict = {}
for d1, d2, name1, name2 in contrasts:
    key = f"{name1}_vs_{name2}"
    results_dict[key] = run_subtraction(d1, d2, name1, name2)

# cluster tables using z-map that was created :P
for contrast_name, sub_corr in results_dict.items():
    if sub_corr is None:
        continue
    
    # Use the actual z-map filenameeeee  
    nii_file = os.path.join(
        "/home/tur31606@tu.temple.edu/ALE/results",
        contrast_name,
        f"{contrast_name}_z_desc-group1MinusGroup2.nii.gz"  # Fixed filename
    ) 
    
    if not os.path.exists(nii_file):
        print(f"WARNING: Z-map not found for {contrast_name}: {nii_file}")
        continue
    
    table = get_clusters_table(
        nii_file,
        stat_threshold=1.96,  # two-sided z > 1.96 (p < 0.05)
    )
    
    out_csv = os.path.join(
        "/home/tur31606@tu.temple.edu/ALE/results",
        contrast_name,
        "clusters.csv"
    )
    table.to_csv(out_csv, index=False) 
    print(f"Cluster table saved: {out_csv}")
