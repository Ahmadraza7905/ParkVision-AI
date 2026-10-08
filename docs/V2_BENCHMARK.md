# ParkVision AI — V2 Production Pipeline Benchmark

## Dataset

- Dataset: PKLot
- Validation images evaluated: 100
- Ground-truth parking spaces: 10,000

## Production Pipeline Results

- Predictions: 10,005
- Matched: 9,990
- False positives: 15
- False negatives: 10

### Detection

- Precision: 99.85%
- Recall: 99.90%
- F1: 99.88%

### Occupancy

- Occupancy accuracy: 99.85%
- Occupied precision: 100.00%
- Occupied recall: 99.76%
- Occupied F1: 99.88%
- Empty precision: 99.59%
- Empty recall: 100.00%
- Empty F1: 99.80%

### Occupancy Confusion

- Correct occupied: 6,317
- Correct empty: 3,658
- Occupied -> empty: 15
- Empty -> occupied: 0

## Important

These results are a benchmark on a 100-image PKLot validation subset. They must not be represented as guaranteed real-world or production accuracy.

The V2 model is currently the ParkVision baseline model.
