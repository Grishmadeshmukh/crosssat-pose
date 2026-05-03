# Generalizable Pose Estimation

quick coverage and appearance sanity check:

```bash
python3 scripts/run_visual_inspection.py --satellites hst juno
```

```bash
python3 scripts/run_visual_inspection.py --satellites jason-1_final ostm_jason-2
```

## Hand-Crafted Geometry Search

Same-satellite geometry-aware search:

```bash
python3 scripts/run_geometry_overfit.py \
  --query-satellite jason-1_final \
  --mesh-satellite jason-1_final \
  --candidate-satellite jason-1_final
```

Cross-satellite geometry-aware search:

```bash
python3 scripts/run_geometry_overfit.py \
  --query-satellite jason-1_final \
  --mesh-satellite ostm_jason-2 \
  --candidate-satellite ostm_jason-2
```

## Learned Mesh-Conditioned Scorer

Train the learned scorer on one satellite:

```bash
python3 scripts/train_learned_mesh_scorer.py \
  --query-satellite jason-1_final \
  --mesh-satellite jason-1_final \
  --candidate-satellite jason-1_final
```

this writes the checkpoint and training curves to:

```bash
outputs/learned_pose_scorer/jason-1_final__mesh_jason-1_final__train/
```

eval the learned scorer on the same satellite:

```bash
python3 scripts/evaluate_learned_mesh_scorer.py \
  --checkpoint outputs/learned_pose_scorer/jason-1_final__mesh_jason-1_final__train/best_model.pt \
  --query-satellite jason-1_final \
  --mesh-satellite jason-1_final \
  --candidate-satellite jason-1_final \
  --candidate-mode hybrid
```

eval cross-satellite transfer with the same checkpoint:

```bash
python3 scripts/evaluate_learned_mesh_scorer.py \
  --checkpoint outputs/learned_pose_scorer/jason-1_final__mesh_jason-1_final__train/best_model.pt \
  --query-satellite jason-1_final \
  --mesh-satellite ostm_jason-2 \
  --candidate-satellite ostm_jason-2 \
  --candidate-mode hybrid
```

## Benchmark-Style Shortlist Refiner

Train the benchmark-aligned shortlist refiner. This uses:
- a geometry-based coarse stage to build the real shortlist
- object-centric crops for query and render pairs
- listwise shortlist supervision instead of binary scoring
- rotation-only refinement, since translation refinement was unstable in the trial repo
- optional symmetry-aware supervision from the query satellite's detected mesh rotation symmetry group

```bash
python3 scripts/train_benchmark_refiner.py \
  --query-satellite jason-1_final \
  --mesh-satellite jason-1_final \
  --candidate-satellite jason-1_final \
  --use-dataset-bank \
  --use-structured-bank
```

this writes the checkpoint and training curves to:

```bash
outputs/benchmark_refiner/jason-1_final__mesh_jason-1_final__train/
```

training now logs one line per epoch with:

- train/eval loss
- train/eval shortlist accuracy
- eval mean rotation error
- eval folded half-turn rotation error

train the symmetry-aware version that uses the query satellite's detected mesh symmetry group:

```bash
python3 scripts/train_benchmark_refiner.py \
  --query-satellite jason-1_final \
  --mesh-satellite jason-1_final \
  --candidate-satellite jason-1_final \
  --use-dataset-bank \
  --use-structured-bank \
  --use-mesh-symmetry-group \
  --mesh-symmetry-threshold 0.03 \
  --output-dir outputs/benchmark_refiner_meshsym
```

eval the benchmark refiner on the same satellite:

```bash
python3 scripts/evaluate_benchmark_refiner.py \
  --checkpoint outputs/benchmark_refiner/jason-1_final__mesh_jason-1_final__train/best_model.pt \
  --query-satellite jason-1_final \
  --mesh-satellite jason-1_final \
  --candidate-satellite jason-1_final \
  --use-dataset-bank \
  --use-structured-bank \
  --iterations 3
```

eval cross-satellite transfer on a similar satellite:

```bash
python3 scripts/evaluate_benchmark_refiner.py \
  --checkpoint outputs/benchmark_refiner/jason-1_final__mesh_jason-1_final__train/best_model.pt \
  --query-satellite jason-1_final \
  --mesh-satellite ostm_jason-2 \
  --candidate-satellite ostm_jason-2 \
  --use-dataset-bank \
  --use-structured-bank \
  --iterations 3
```

eval the symmetry-aware benchmark and report both standard and symmetry-aware rotation errors:

```bash
python3 scripts/evaluate_benchmark_refiner.py \
  --checkpoint outputs/benchmark_refiner_meshsym/jason-1_final__mesh_jason-1_final__train/best_model.pt \
  --query-satellite jason-1_final \
  --mesh-satellite jason-1_final \
  --candidate-satellite jason-1_final \
  --use-dataset-bank \
  --use-structured-bank \
  --use-mesh-symmetry-group \
  --mesh-symmetry-threshold 0.03 \
  --iterations 3 \
  --output-dir outputs/benchmark_pose_search_meshsym
```

the benchmark evaluator now writes:

- standard rotation error
- folded half-turn rotation error
- mesh-symmetry-aware rotation error
- counts of likely symmetry-driven failures

## output folders

- `outputs/visual_inspection/`
- `outputs/geometry_search/`
- `outputs/learned_pose_scorer/`
- `outputs/learned_mesh_pose_search/`
- `outputs/benchmark_refiner/`
- `outputs/benchmark_pose_search/`

learned scorer workflow produces:

- `best_model.pt`
- `experiment_summary.json`
- `history.csv`
- `training_curves.png`
- `metrics.json`
- `retrievals.csv`
- `qualitative_gallery.png`

benchmark refiner workflow produces:

- `best_model.pt`
- `experiment_summary.json`
- `history.csv`
- `training_curves.png`
- `metrics.json`
- `retrievals.csv`
- `qualitative_gallery.png`

# Classification
Training (below is the best run so far)
```bash
python3 scripts/run_classification_train.py \
  --dataset-root spe3r \
  --classification-csv classification.csv \
  --output-dir outputs/classification_sat_holdout_v6 \
  --epochs 20 \
  --split-mode satellite \
  --learning-rate 2e-4 \
  --weight-decay 3e-4 \
  --label-smoothing 0.1 \
  --class-weight-power 0.8 \
  --use-class-weighted-loss \
  --use-weighted-sampler \
  --checkpoint-metric macro_acc \
  --strong-augmentation \
  --early-stopping-patience 6
```
Eval
```bash
python3 scripts/run_classification_eval.py \
  --dataset-root spe3r \
  --classification-csv classification.csv \
  --checkpoint outputs/classification_sat_holdout_v5/best_model.pt \
  --split-mode satellite \
  --output-dir outputs/classification_sat_holdout_v6_eval
```
1. Run classifier on an image
```bash
python3 scripts/run_classification_predict.py \
  --checkpoint outputs/classification_sat_holdout_v6/best_model.pt \
  --image /absolute/path/to/query_image.jpg \
  --top-k 1
```

2. Get all satellites in that predicted class
Suppose prediction is: Box Bus + Solar Wings
Then list satellites from classification.csv with this command:
```bash
python3 - <<'PY'
import csv
target_class = "Box Bus + Solar Wings"  # replace with predicted class string
with open("classification.csv", newline="") as f:
    rows = list(csv.DictReader(f))
matches = [r["satellite_name"] for r in rows if r["architecture_label"].strip() == target_class]
print(f"class: {target_class}")
print(f"count: {len(matches)}")
for name in matches:
    print(name)
PY
```
