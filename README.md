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

Predict the architecture class for one image:
```bash
python3 scripts/run_classification_predict.py \
  --checkpoint outputs/classification_sat_holdout_v6/best_model.pt \
  --image path/to/query.png
```

## Class-Conditioned Pose Routing

This script connects the classifier to the benchmark pose estimator:
- classifier predicts the satellite architecture class
- a hardcoded representative route is selected for that class
- the corresponding benchmark checkpoint is used to estimate pose for the query image

For now, the representative routes are hardcoded in [run_class_conditioned_pose.py](/Users/dawn/Documents/nyu/courses/cvse/crosssat-pose/scripts/run_class_conditioned_pose.py) because we do not yet have trained pose models for every class.

```bash
python3 scripts/run_class_conditioned_pose.py \
  --classification-checkpoint outputs/classification_sat_holdout_v6/best_model.pt \
  --query-image path/to/query.png \
  --query-mask path/to/query_mask.png \
  --max-candidate-samples 64 \
  --no-use-structured-bank
```

This writes:
- `outputs/class_conditioned_pose/class_conditioned_result.json`
- `outputs/class_conditioned_pose/summary.md`
- `outputs/class_conditioned_pose/pose_prediction/prediction.json`
- `outputs/class_conditioned_pose/pose_prediction/refined_render.png`
- `outputs/class_conditioned_pose/pose_prediction/refined_mask.png`
- `outputs/class_conditioned_pose/pose_prediction/overlay.png`
