# 2026-11-04 mask模块迁入 algorithm

步骤 3 只搬位置。provider 与 Kalman 类在 `src/spec_mtp/structure/algorithm/mask/providers.py`。K 拟合在 `fit.py`。`build_mask_provider` 在 `factory.py`。

同日删除顶层 `masks.py`。调用方从 `spec_mtp.structure.algorithm.mask` 导入。配置键没有单列到 `structure/algorithm/config.py`。见 [删除 masks 与 decode 顶层再导出](<2026-11-04 删除 masks 与 decode 顶层再导出.md>)。
