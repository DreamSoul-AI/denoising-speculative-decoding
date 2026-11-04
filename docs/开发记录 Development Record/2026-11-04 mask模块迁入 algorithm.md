# 2026-11-04 mask模块迁入 algorithm

步骤 3 只搬位置。provider 与 Kalman 类在 `src/spec_mtp/structure/algorithm/mask/providers.py`。K 拟合在 `fit.py`。`build_mask_provider` 在 `factory.py`。

`src/spec_mtp/masks.py` 仍再导出这些公开名，对象与新模块相同。函数正文和默认参数没有改。调用方这一步仍从 `spec_mtp.masks` 导入。配置键没有单列到 `structure/algorithm/config.py`。步骤 4–7 未做。步骤 8 里测试镜像改路径和 `.importlinter` 按两柱重写仍未做。
