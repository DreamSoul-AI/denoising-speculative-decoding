**去噪推测解码 缺陷记录**
更新日期：2026年11月6日。本文只维护活动缺陷，关闭经过与验证从 [开发记录](<开发记录 Development Record/2026-11-05 宿主结构收束.md>)、[运行记录](<运行记录 Run Record/2026-11-05 宿主结构CPU验证.md>) 定位。

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

# 五、CPU检查收集已修，整套设备选集仍待处理
原d79d0ac的远端cpu-checks与本地同命令均复现14个收集错误。实际缺失的是test_window_observer/test_proxy_transfer两个裸测试模块，生产spec_mtp可导入。原因是旧兄弟测试引用依赖prepend目录副作用，而当前pytest采用importlib模式。本次只将14文件17条引用改为tests.spec_mtp完整模块名，符号、fixture、测试体和断言不变。

1103项已完整收集，两个helper各只有一个模块身份。现有研究归档已补入test_docs的精确集合；本轮隐藏CUDA全跑为832通过、54失败、217错误，归档合同通过。剩余269项是显式CUDA合同，另2项为本机旧RPipe目录/安装入口；不删除禁区或修改共享环境。该本地Windows/Python3.13/CUDA torch在CUDA_VISIBLE_DEVICES空值下实测is_available=True、device_count=0，runtime计数1而NVML计数0；远端CPU wheel在可用性断言直接拒绝，不能混用两类栈或数量。

共享设备来源为cuda_decode的192项和cuda_checkpoint的25项，均含真实设备/RNG/张量合同；普通_system路径用例已过，不是默认设备引起这组失败。其余6函数8参数错误也均属CUDA恢复合同，在真实GPU定向回归通过。没有正确的单点CPU回退方案；不改CUDA合同或拆夹具求绿。工作流仍直接执行混合设备目录，后续候选需单独授权接已有cpu-pr及全新runner临时根，CUDA合同另在GPU执行；更换入口本身不证明全量通过。完整269项清单、现有54项GPU环境回归及新HEAD远端状态从运行记录与PR106定位，本轮不修改workflow、选择策略或skip/ignore。
