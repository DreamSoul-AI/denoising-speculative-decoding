# Numbers: k_affine_residual

Generated from `process.json`. The conclusion stays in `STUDY_REPORT.md`.

## Experiments

| factors | metric | mean | std | min | max | n |
|---|---|---:|---:|---:|---:|---:|
| - | cross_dataset_bias_only_document_mean_mse | 1.7159 | 0.0000 | 1.7159 | 1.7159 | 1 |
| - | cross_dataset_bias_only_mean_cosine | 0.2389 | 0.0000 | 0.2389 | 0.2389 | 1 |
| - | cross_dataset_bias_only_target_argmax_agreement | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 |
| - | cross_dataset_bias_only_target_kl | 0.0110 | 0.0000 | 0.0110 | 0.0110 | 1 |
| - | cross_dataset_full_k_document_mean_mse | 0.7699 | 0.0000 | 0.7699 | 0.7699 | 1 |
| - | cross_dataset_full_k_mean_cosine | 0.7815 | 0.0000 | 0.7815 | 0.7815 | 1 |
| - | cross_dataset_full_k_target_argmax_agreement | 0.2031 | 0.0000 | 0.2031 | 0.2031 | 1 |
| - | cross_dataset_full_k_target_kl | 0.0049 | 0.0000 | 0.0049 | 0.0049 | 1 |
| - | cross_dataset_none_document_mean_mse | 1.5148 | 0.0000 | 1.5148 | 1.5148 | 1 |
| - | cross_dataset_none_mean_cosine | 0.2372 | 0.0000 | 0.2372 | 0.2372 | 1 |
| - | cross_dataset_none_target_argmax_agreement | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 |
| - | cross_dataset_none_target_kl | 0.0097 | 0.0000 | 0.0097 | 0.0097 | 1 |
| - | fit_bias_only_document_mean_mse | 1.3657 | 0.0000 | 1.3657 | 1.3657 | 1 |
| - | fit_bias_only_mean_cosine | 0.2324 | 0.0000 | 0.2324 | 0.2324 | 1 |
| - | fit_bias_only_target_argmax_agreement | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 |
| - | fit_bias_only_target_kl | 0.0087 | 0.0000 | 0.0087 | 0.0087 | 1 |
| - | fit_full_k_document_mean_mse | 0.1334 | 0.0000 | 0.1334 | 0.1334 | 1 |
| - | fit_full_k_mean_cosine | 0.9487 | 0.0000 | 0.9487 | 0.9487 | 1 |
| - | fit_full_k_target_argmax_agreement | 0.7656 | 0.0000 | 0.7656 | 0.7656 | 1 |
| - | fit_full_k_target_kl | 0.0009 | 0.0000 | 0.0009 | 0.0009 | 1 |
| - | fit_none_document_mean_mse | 1.4804 | 0.0000 | 1.4804 | 1.4804 | 1 |
| - | fit_none_mean_cosine | 0.2539 | 0.0000 | 0.2539 | 0.2539 | 1 |
| - | fit_none_target_argmax_agreement | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 |
| - | fit_none_target_kl | 0.0095 | 0.0000 | 0.0095 | 0.0095 | 1 |
| - | holdout_bias_only_document_mean_mse | 1.3224 | 0.0000 | 1.3224 | 1.3224 | 1 |
| - | holdout_bias_only_mean_cosine | 0.2808 | 0.0000 | 0.2808 | 0.2808 | 1 |
| - | holdout_bias_only_target_argmax_agreement | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 |
| - | holdout_bias_only_target_kl | 0.0085 | 0.0000 | 0.0085 | 0.0085 | 1 |
| - | holdout_full_k_document_mean_mse | 0.4826 | 0.0000 | 0.4826 | 0.4826 | 1 |
| - | holdout_full_k_mean_cosine | 0.8447 | 0.0000 | 0.8447 | 0.8447 | 1 |
| - | holdout_full_k_target_argmax_agreement | 0.3750 | 0.0000 | 0.3750 | 0.3750 | 1 |
| - | holdout_full_k_target_kl | 0.0031 | 0.0000 | 0.0031 | 0.0031 | 1 |
| - | holdout_none_document_mean_mse | 1.3841 | 0.0000 | 1.3841 | 1.3841 | 1 |
| - | holdout_none_mean_cosine | 0.3038 | 0.0000 | 0.3038 | 0.3038 | 1 |
| - | holdout_none_target_argmax_agreement | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 |
| - | holdout_none_target_kl | 0.0089 | 0.0000 | 0.0089 | 0.0089 | 1 |

## Runs

| factors | seed | id | metrics | log |
|---|---:|---|---|---|
| - | 123 | `171d55477ad7c899` | cross_dataset_bias_only_document_mean_mse=1.7159, cross_dataset_bias_only_mean_cosine=0.2389, cross_dataset_bias_only_target_argmax_agreement=0.0000, cross_dataset_bias_only_target_kl=0.0110, cross_dataset_full_k_document_mean_mse=0.7699, cross_dataset_full_k_mean_cosine=0.7815, cross_dataset_full_k_target_argmax_agreement=0.2031, cross_dataset_full_k_target_kl=0.0049, cross_dataset_none_document_mean_mse=1.5148, cross_dataset_none_mean_cosine=0.2372, cross_dataset_none_target_argmax_agreement=0.0000, cross_dataset_none_target_kl=0.0097, fit_bias_only_document_mean_mse=1.3657, fit_bias_only_mean_cosine=0.2324, fit_bias_only_target_argmax_agreement=0.0000, fit_bias_only_target_kl=0.0087, fit_full_k_document_mean_mse=0.1334, fit_full_k_mean_cosine=0.9487, fit_full_k_target_argmax_agreement=0.7656, fit_full_k_target_kl=0.0009, fit_none_document_mean_mse=1.4804, fit_none_mean_cosine=0.2539, fit_none_target_argmax_agreement=0.0000, fit_none_target_kl=0.0095, holdout_bias_only_document_mean_mse=1.3224, holdout_bias_only_mean_cosine=0.2808, holdout_bias_only_target_argmax_agreement=0.0000, holdout_bias_only_target_kl=0.0085, holdout_full_k_document_mean_mse=0.4826, holdout_full_k_mean_cosine=0.8447, holdout_full_k_target_argmax_agreement=0.3750, holdout_full_k_target_kl=0.0031, holdout_none_document_mean_mse=1.3841, holdout_none_mean_cosine=0.3038, holdout_none_target_argmax_agreement=0.0000, holdout_none_target_kl=0.0089 | [run.log](../runs/171d55477ad7c899/assets/logs/run.log) |
