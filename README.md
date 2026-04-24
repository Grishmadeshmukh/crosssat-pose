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

# Classifiy Satellites
Train:
```bash
python3 scripts/run_classification_train.py \
  --dataset-root spe3r \
  --classification-csv classification.csv \
  --output-dir outputs/classification_debug \
  --epochs 1 \
  --batch-size 16 \
  --image-size 128
```
Predict:
```bash
python3 scripts/run_classification_predict.py \
  --checkpoint outputs/classification/best_model.pt \
  --image /absolute/path/to/test.jpg \
  --top-k 3
```
