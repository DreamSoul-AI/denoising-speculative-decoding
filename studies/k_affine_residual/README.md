# K affine residual correction

Independent postcommit-replacement CPU Study, authorised2026-10-20. The [project plan](<../../docs/研究计划 Denoising Speculative Decoding.md#k-affine-residual-20261020>) / [English](<../../docs/Research Plan Denoising Speculative Decoding.md#k-affine-residual-20261020>) defines the question, data isolation, fit, metrics and budget. [Registration](docs/registration.json) and [input manifest](docs/data_manifest.json) precede outputs.

`z'=z(m)+A(E[y]−m)+b`, y is already committed at this same position; target is z(y). This is not future-slot hit reproduction. Random tied tiny LLaMA uses real cached text/token IDs; full K and separately fitted bias-only freeze before heldout/cross evaluation. Quality, performance and model/data independence remain unproven.

Native entry: `python -m spec_mtp make studies/k_affine_residual --round 1`, then `launch`, `process`, `status`, `report`. One Run, three paired arms; no new runner or shared source changes. Formal execution waits for root's immutable-plan commit and owned-tree guard. Raw artifacts stay local under runs; small reconstructible summaries belong in docs/STUDY_REPORT.md and docs/NUMBERS.md.

Use `python -m rpipe report studies/k_affine_residual` for the final generic numeric report. The SpecMTP thin `report` entry additionally invokes the old decoding-specific gate; this Study has no decoding acceptance/argmax protocol. Its native collect/summarize/write/process gates validate the separately registered latent-pair contract instead.
