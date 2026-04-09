# Generalizable Pose Estimation

```bash
python3 scripts/run_visual_inspection.py --satellites hst juno
```

```bash
python3 scripts/run_geometry_overfit.py \
  --query-satellite hst \
  --mesh-satellite hst \
  --candidate-satellite hst
```

```bash
python3 scripts/run_geometry_overfit.py \
  --query-satellite hst \
  --mesh-satellite juno \
  --candidate-satellite juno
```