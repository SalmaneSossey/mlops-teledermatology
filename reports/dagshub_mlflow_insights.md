# DagsHub MLflow Run Insights

Pulled from `https://dagshub.com/SalmaneSossey/mlops-teledermatology.mlflow` on `2026-05-20T16:03:17.297214+00:00`.

- Experiments found: 6
- Runs found: 21
- Finished runs with logged test macro F1: 16
- Latest ISIC workflow runs found: yes

## Latest Finished Runs

| Started UTC | Experiment | Run | Status | Macro F1 | Balanced Acc | High-Risk Recall |
|---|---|---|---|---:|---:|---:|
| 2026-05-20 15:05 | pad-ufes-20-image-baseline-multimodal-isic-init | `efficientnet_b0_multimodal_20260520_150522` | FINISHED | 0.6902 | 0.6804 | 0.8902 |
| 2026-05-20 14:53 | pad-ufes-20-image-baseline-isic-init | `efficientnet_b0_20260520_145311` | FINISHED | 0.6432 | 0.6612 | 0.8049 |
| 2026-05-20 13:42 | pad-ufes-20-isic-2019-pretrain | `efficientnet_b0_20260520_134256` | FINISHED | 0.7053 | 0.7790 | 0.8843 |
| 2026-05-20 12:40 | pad-ufes-20-image-baseline-multimodal | `efficientnet_b0_multimodal_20260520_124000` | FINISHED | 0.6236 | 0.6707 | 0.8537 |
| 2026-05-20 12:10 | pad-ufes-20-image-baseline-multimodal | `efficientnet_b0_multimodal_20260520_121017` | FINISHED | 0.6383 | 0.6363 | 0.8659 |
| 2026-05-18 16:27 | pad-ufes-20-image-baseline-hparam-sweep | `efficientnet_b0_20260518_162756` | FINISHED |  |  |  |
| 2026-05-18 16:09 | pad-ufes-20-image-baseline-hparam-sweep | `efficientnet_b0_20260518_160913` | FINISHED | 0.6430 | 0.6463 | 0.8110 |
| 2026-05-18 15:50 | pad-ufes-20-image-baseline | `efficientnet_b0_20260518_155046` | FINISHED | 0.6430 | 0.6463 | 0.8110 |

## ISIC Workflow Result

| Model | Run ID | Macro F1 | Balanced Acc | High-Risk Recall | Test Selection |
|---|---|---:|---:|---:|---:|
| April image-only best | `4f185f85c76e497f94e429a568f03a04` | 0.6597 | 0.6746 | 0.8232 | n/a |
| ISIC pretrain on ISIC test | `eea7f072a40c4621a6d33ba4d5dfe991` | 0.7053 | 0.7790 | 0.8843 | n/a |
| PAD image, ISIC initialized | `6118b45498f1484db00df25d3186e3dd` | 0.6432 | 0.6612 | 0.8049 | n/a |
| PAD multimodal, ISIC initialized | `ef084927bef741f996894b8a0fdd63e3` | 0.6902 | 0.6804 | 0.8902 | 0.7902 |

Main conclusion: ISIC external pretraining helped most when combined with metadata fusion. The new PAD multimodal ISIC-initialized run is now the strongest final PAD run by macro F1, balanced accuracy, and high-risk recall.

## Post-Best Ablations

These later experiments were useful diagnostically, but they do not replace the final selected model.

| Experiment | Run ID / Output | Macro F1 | Balanced Acc | High-Risk Recall | SCC Recall | Decision |
|---|---|---:|---:|---:|---:|---|
| Final selected PAD multimodal, ISIC initialized | `ef084927bef741f996894b8a0fdd63e3` | 0.6902 | 0.6804 | 0.8902 | 0.2069 | Keep as final |
| Class-aware augmentation | `multimodal_class_aware_aug/isic_init` | 0.6127 | 0.6115 | 0.8232 | 0.1379 | Failed promotion gates |
| Derm8 eight-class pretraining/fine-tune ablation | `derm8` Colab run | 0.6145 | 0.6291 | 0.8659 | 0.3103 | Report as ablation only |

The class-aware augmentation run did not achieve its main goal: SCC recall dropped from `0.2069` to `0.1379`, while macro F1, balanced accuracy, and high-risk recall also fell beyond the allowed promotion margins.

The Derm8 label-space experiment improved SCC recall to `0.3103`, which is clinically interesting, but it hurt global metrics enough that it should be presented as an ablation rather than the deployed/final model. It supports the conclusion that broader dermatology label coverage can help weak high-risk classes, but the current pipeline still needs better balancing before replacing the six-class PAD model.

Final model selection therefore remains `ef084927bef741f996894b8a0fdd63e3`.

## PAD Class-Level Comparison

| Model | ACK F1 | BCC Recall/F1 | MEL Recall/F1 | SCC Recall/F1 | NEV F1 | SEK F1 |
|---|---:|---:|---:|---:|---:|---:|
| April image-only best | 0.7115 | 0.6378/0.7013 | 0.7500/0.7500 | 0.3793/0.2558 | 0.8095 | 0.7302 |
| PAD image, ISIC initialized | 0.7273 | 0.6614/0.7089 | 0.6250/0.6250 | 0.3793/0.2973 | 0.7470 | 0.7536 |
| PAD multimodal, ISIC initialized | 0.8038 | 0.7795/0.7765 | 0.7500/0.8571 | 0.2069/0.1935 | 0.7750 | 0.7353 |

## Interpretation

- The new ISIC-initialized multimodal run reached macro F1 `0.6902`, balanced accuracy `0.6804`, and high-risk recall `0.8902` on PAD-UFES-20.
- Compared with the previous April image-only best, macro F1 improved by about `+0.0305`, balanced accuracy by `+0.0058`, and high-risk recall by `+0.0671`.
- The improvement is mainly from stronger ACK, BCC, MEL, and NEV behavior. MEL is tiny in the PAD test set, so report that carefully.
- SCC is still the weak point: SCC recall is only `0.2069` and SCC F1 is `0.1935` in the best new run. So the model is better overall, but SCC remains a limitation to discuss in the final project.
- The ISIC-initialized image-only PAD run did not beat the old image-only baseline. The gain appears when ISIC initialization is combined with clinical metadata.

## Local Artifacts Pulled

- `reports/dagshub_mlflow_runs_snapshot.json`
- `reports/dagshub_mlflow_artifacts/latest_isic/`
