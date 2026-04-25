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

## output folders

- `outputs/visual_inspection/`
- `outputs/geometry_search/`
- `outputs/learned_pose_scorer/`
- `outputs/learned_mesh_pose_search/`

learned scorer workflow produces:

- `best_model.pt`
- `experiment_summary.json`
- `history.csv`
- `training_curves.png`
- `metrics.json`
- `retrievals.csv`
- `qualitative_gallery.png`
