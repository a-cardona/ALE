# %%
from nimare.meta.cbma import ALESubtraction
from nimare.correct import FWECorrector
from nilearn.reporting import get_clusters_table
from nimare import io
import os

# %%
BASE = os.getcwd()
print("PWD:", BASE)
# Load existing ALE coordinate datasets
attention_dset = io.convert_sleuth_to_dataset("cleaned_sleuth/Attention.txt")
cognitive_dset = io.convert_sleuth_to_dataset("cleaned_sleuth/Cognitive_PT.txt")
spatial_dset   = io.convert_sleuth_to_dataset("cleaned_sleuth/Spatial_PT_lvl2.txt")
affective_dset = io.convert_sleuth_to_dataset("cleaned_sleuth/Affective_PT.txt")

# %%
def run_subtraction(
    dset1, dset2, name1, name2,
    n_iters=10000,
):
    print(f"Running subtraction: {name1} vs {name2}")

    sub = ALESubtraction(
        n_iters=n_iters,
        two_sided=True,
        random_state=42
    )
    sub_results = sub.fit(dset1, dset2)

    corr = FWECorrector(method="montecarlo")
    sub_corr = corr.transform(sub_results)

    prefix = f"{name1}_vs_{name2}"
    outdir = os.path.join(
        "/home/tur31606@tu.temple.edu/ALE/results",
        prefix
    )
    os.makedirs(outdir, exist_ok=True)

    print(">>> ABOUT TO SAVE MAPS <<<")
    print("Saving to:", outdir)

    sub_corr.save_maps(
        output_dir=outdir,
        prefix=prefix
    )

    print(">>> SAVE_MAPS FINISHED <<<")
    print("Files:", os.listdir(outdir))

    return sub_corr

# %%
contrasts = [
    (attention_dset, cognitive_dset, "attention", "cognitive"),
    (attention_dset, spatial_dset,   "attention", "spatial"),
    (attention_dset, affective_dset, "attention", "affective"),
    (cognitive_dset, spatial_dset,   "cognitive", "spatial"),
    (cognitive_dset, affective_dset, "cognitive", "affective"),
    (spatial_dset,   affective_dset, "spatial", "affective"),
]

# %%
results_dict = {}

for d1, d2, name1, name2 in contrasts:
    key = f"{name1}_vs_{name2}"
    results_dict[key] = run_subtraction(d1, d2, name1, name2)

# %%
# Generate cluster tables robustly
for contrast_name, sub_corr in results_dict.items():

    if sub_corr is None:
        continue

    # Find the corrected Z map programmatically
    nii_file = os.path.join(
        "/home/tur31606@tu.temple.edu/ALE/results",
        contrast_name,
        f"{contrast_name}_corr-FWE_z.nii.gz"
    ) 
    table = get_clusters_table(
        nii_file,
        stat_threshold=1.96,  # two-sided
    )

    out_csv = os.path.join(
        "results",
        contrast_name,
        "clusters.csv"
    )
    table.to_csv(out_csv, index=False) 
    print(f"Cluster table saved: {out_csv}")
