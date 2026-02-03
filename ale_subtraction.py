# %%
from nimare.meta.cbma import ALESubtraction
from nimare.correct import FWECorrector
from nilearn.reporting import get_clusters_table
from nimare import io
import os

print("RUNNING FILE:", __file__)
# %%
BASE = "/home/tur31606@tu.temple.edu/ALE"
# Load existing ALE coordinate datasets
attention_dset = io.convert_sleuth_to_dataset(f"{BASE}/cleaned_sleuth/Attention.txt")
cognitive_dset = io.convert_sleuth_to_dataset(f"{BASE}/cleaned_sleuth/Cognitive_PT.txt")
spatial_dset   = io.convert_sleuth_to_dataset(f"{BASE}/cleaned_sleuth/Spatial_PT_lvl2.txt")
affective_dset = io.convert_sleuth_to_dataset(f"{BASE}/cleaned_sleuth/Affective_PT.txt")

# %%
def run_subtraction(
    dset1, dset2, name1, name2,
    voxel_thresh=0.005,
    n_iters=10000,
):
    """
    Run ALE subtraction with FWE correction and save maps.
    Returns corrected results object.
    """
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
    outdir = f"{BASE}/results/{prefix}"
    os.makedirs(outdir, exist_ok=True)

    print("ABOUT TO SAVE:", outdir) 

    sub_corr.save_maps(
        output_dir=outdir,
        prefix=prefix
    )


    print("FILES:", os.listdir(outdir))
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
    z_maps = [
        fname for fname in sub_corr.maps
        if "z" in fname and "corr-FWE" in fname
    ]

    if not z_maps:
        print(f"No corrected Z map found for {contrast_name}")
        continue

    nii_file = z_maps[0]

    table = get_clusters_table(
        nii_file,
        stat_threshold=1.96,  # two-sided
    )

    out_csv = f"cluster_table_{contrast_name}.csv"
    table.to_csv(out_csv, index=False)

    print(f"Cluster table saved: {out_csv}")
