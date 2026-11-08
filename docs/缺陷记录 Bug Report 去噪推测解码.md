**去噪推测解码 缺陷记录**
更新日期：2026年11月8日。本文只维护活动缺陷，关闭经过与验证从 [开发记录](<开发记录 Development Record/2026-11-05 宿主结构收束.md>)、[运行记录](<运行记录 Run Record/2026-11-05 宿主结构CPU验证.md>) 定位。

# 一、最低支持版本的严格测试未验证
声明支持 Python 3.10，而 test_package 通过 importorskip 导入 tomllib，3.10 缺该标准库时会跳过模块，与严格 CPU 计划的缺收集阻断冲突。当前执行环境为3.13，最低版本测试依赖策略尚未确定，不能将模拟缺 file_digest 的通过写成全版本验收。

验收需要明确测试解析依赖，并在实际 Python 3.10 环境完整收集、执行必需计划。当前没有安装依赖或新建该环境；状态为待处理。

# 二、工程整理与证据缺口
专用启动入口的Run根接入、其余旧分析消费者、日志可靠性及测试元数据的剩余工作由 [开发计划](<开发计划 Development Plan 去噪推测解码.md>) 维护。缺失原运行矩阵或未知历史 helper 身份会阻断对应完整分析，不用当前源码、换行转换或新哈希替代原件。

当前检查覆盖两柱整理及CUDA工程合同，没有开展全库缺陷审计；已有方案核验与研究收束见 [研究计划](<研究计划 Research Plan 去噪推测解码.md>)。

# 三、本机PowerShell脚本执行受阻
生成launch.ps1的真实执行被本机有效Restricted策略阻断，失败回执从运行记录第五轮定位。当前只验证生成内容与正式Python CLI的CUDA调度，不把脚本记为功能通过；未绕过或修改策略。若未来环境允许执行，需独立验收脚本参数、中文路径和退出码。

# 四、静态校正跨prompt状态重置待核
只读审查发现VerifyCorrectionMaskProvider在线scalar的init重置轮次标记，但未见还原拟合的k/num/den；跨prompt复用与每prompt新实例可能不同。该项尚未CUDA复现，不能据此判定旧Run失效；若必要诊断对照用到它，先按真实调用方式验证并明确状态作用域。当前Kalman/EMA独立参考不覆盖该静态校正类。

# 五、R104四题收集薄绑定的冷导入失败

状态：待修复，阻塞q85–88观察收集，已撤回此前的收集CPU就绪判断。旧矩校准参数与只读影子数学合同不受此入口缺陷影响；新预测、接受与完整T0仍未执行。

真实worker在`collect_entry.observer_factory`返回工厂后，原`r101_entry.worker`已将`eval_native_observer.NativeObservation`替换为该函数；工厂的construct此时才导入cost_observer→causal_native，后者执行`class CausalObservation(NativeObservation)`，基类已是函数，触发`TypeError: function() argument 'code' must be code, not str`。此前FakeObserver测试替换了整个cost_observer模块，漏验真实冷导入。原入口在创建工厂时提前导入，新增薄绑定将它延后，是本缺陷的直接原因。

最小修复建议：在**worker获准后创建observer_factory时**导入原CausalObservation，早于替换基类；不得移到入口模块顶层而破坏未授权时不导torch的门。先用隔离CPU进程覆盖真实模块冷导入及基类身份，不用FakeObserver替代这条回归；同时保留未授权CLI在torch/child/resource前拒绝与作用域还原检查。无需更改算法、Job owner、模型或时间／资源门。

本批只登记原因与验收，冻结失败入口未修改。后续修复须保留旧源码／协议，使用新版本与新Run根，重新核来源和入口后再评有限授权；不得复用失败根、旧grant或自动重试。实际v1/v2及释放证据见[运行记录第一百一十一节](<运行记录 Run Record/2026-11-05 宿主结构CPU验证.md>)。
