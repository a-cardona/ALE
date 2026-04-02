from nimare.meta.cbma import ALE, ALESubtraction
from nimare.correct import FWECorrector
from nilearn.reporting import get_clusters_table
from nimare import io
import os

# path setup
BASE = os.getcwd()
print("PWD:", BASE)
RESULTS_DIR = "/home/tur31606@tu.temple.edu/ALE/results"
SINGLE_DIR = os.path.join(RESULTS_DIR, "single_domain")
SUB_DIR    = os.path.join(RESULTS_DIR, "subtraction")
os.makedirs(SINGLE_DIR, exist_ok=True)
os.makedirs(SUB_DIR,    exist_ok=True)

print("\n data set info")
#load datasetsssss
attention_dset = io.convert_sleuth_to_dataset("cleaned_sleuth/Attention.txt")
cognitive_dset = io.convert_sleuth_to_dataset("cleaned_sleuth/Cognitive_PT.txt")
spatial_dset   = io.convert_sleuth_to_dataset("cleaned_sleuth/Spatial_PT_all.txt")
affective_dset = io.convert_sleuth_to_dataset("cleaned_sleuth/Affective_PT.txt")

#alefit lock in
ale = ALE(null_method="approximate")
corr = FWECorrector(method="montecarlo", voxel_thresh=0.001, n_iters=10000, n_cores=7)
attention_results = ale.fit(attention_dset)
cognitive_results = ale.fit(cognitive_dset)
spatial_results = ale.fit(spatial_dset)
affective_results = ale.fit(affective_dset)
#correct ale
attention_corr = corr.transform(attention_results)
cognitive_corr = corr.transform(cognitive_results)
spatial_corr = corr.transform(spatial_results)
affective_corr = corr.transform(affective_results)


# extracted corrected z mapssss
map_key = "z_desc-size_level-cluster_corr-FWE_method-montecarlo"
attention_img = attention_corr.get_map(map_key)
cognitive_img = cognitive_corr.get_map(map_key)
spatial_img = spatial_corr.get_map(map_key)
affective_img = affective_corr.get_map(map_key)
#single domain map
attention_corr.save_maps(ouput_dir=f"{RESULTS_DIR}/single_domain", prefix="attention")
cognitive_corr.save_maps(ouput_dir=f"{RESULTS_DIR}/single_domain", prefix="cognitive")
spatial_corr.save_maps(ouput_dir=f"{RESULTS_DIR}/single_domain", prefix="spatial")
affective_corr.save_maps(ouput_dir=f"{RESULTS_DIR}/single_domain", prefix="affective")

#subtraction mode unlocked
contrasts = [
    (attention_dset, cognitive_dest, "attention", "cognitive"),
    (attention_dset, spatial_dset,   "attention", "spatial"),
    (attention_dset, affective_dset, "attention", "affective"),
    (cognitive_dest, spatial_dset,   "cognitive", "spatial"),
    (cognitive_dset, affective_dset, "cognitive", "affective"),
    (spatial_dset,   affective_dset, "spatial",   "affective"),
]

for d1, d2, n1, n2 in contrasts:
    print(f"\nSubtraction: {n1} vs {n2}")
    sub = ALESubtraction(n_iters=10000, null_method="approximate")
    res_sub = sub.fit(d1, d2)

    outdir = f"{RESULTS_DIR}/subtraction/{n1}_vs_{n2}"
    os.makedirs(oputdir, exist_ok=True)
    res_sub.save_maps(ouput_dir=outdir, prefix=f"{n1}_vs_{n2}")
    print(f"  Saved: {os.listdir(outdir)}")


   #cluster table
    for f in os.listdir(outdir):
        if f.endswith(".nii.gz") and "_z_" in f:
            table = get_clusters_table(os.path.join(outdir, f), stat_threshold=1.96)
            if not table.empty:
                table.to_csv(os.path.join(outdir, f.replace(".nii.gz", "_clusters.csv")), index=False)

