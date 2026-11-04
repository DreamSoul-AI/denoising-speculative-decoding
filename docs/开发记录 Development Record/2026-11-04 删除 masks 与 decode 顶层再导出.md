# 2026-11-04 删除 masks 与 decode 顶层再导出

`src/spec_mtp/masks.py` 与 `src/spec_mtp/decode.py` 已删除。mask 从 `spec_mtp.structure.algorithm.mask` 导入，解码循环从 `spec_mtp.structure.algorithm.decode.loop` 导入。函数正文和默认参数没有改。这两份顶层文件不再占文件落点，legacy 清单去掉对应条目。步骤 5–7 未做。步骤 8 里测试镜像改路径和 `.importlinter` 按两柱重写仍未做。
