# K affine residual correction — CPU mechanism study

Pre-execution implementation status. The [registered project plan](<../../../docs/研究计划 Denoising Speculative Decoding.md#k-affine-residual-20261020>) / [English](<../../../docs/Research Plan Denoising Speculative Decoding.md#k-affine-residual-20261020>) is committed at `ac1162448d7b129e5d8b50d97dc1bd46e6cfee96`. Registration and data selection preceded model construction/output inspection. The first registration check found the first SpecBench question had22tokens; before outputs, all splits' eight positions were fixed to4/6/8/10/12/14/16/20. This is an input-feasibility correction, not a negative model result.

Implementation and focused tests are new Study-local research. No shared algorithm or old Study changed. Unit arrays test formula/validation only; they are not observed model improvement. Native Study execution and scientific interpretation will be appended from its retained raw artifacts. Old technical debt repayment:0.
