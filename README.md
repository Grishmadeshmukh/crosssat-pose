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
