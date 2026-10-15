# Dynamic-ICU-Mortality-Prediction
Dynamic ICU Mortality Prediction: Local Updating
This repository contains two Python scripts supporting the target-domain updating analyses in the study Transportability and Local Updating of Dynamic Multi-Horizon ICU Mortality Prediction Across Healthcare Settings.
Included scripts

formal_full_head_fusion.py: Fine-tunes the nonlinear mortality prediction head using source and target data while keeping the encoder frozen. It also evaluates subsequent logistic recalibration.

formal_recalibration_only.py: Performs logistic recalibration of frozen-model predictions without neural-network fine-tuning.

Together, these scripts support comparison of four model states: frozen, recalibrated, fine-tuned, and fine-tuned followed by recalibration.