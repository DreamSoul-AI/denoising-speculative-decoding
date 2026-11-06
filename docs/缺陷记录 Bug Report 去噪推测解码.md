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

# 五、native BF16主工作树launch超时待定位
活动验收缺口：tests/studies/_shared/flow/test_cli.py::test_bf16_five_methods_and_independent_ar_native_cli在本轮主工作树两次均触发原launch子进程90秒门限，分别1失败、106.30秒和1失败、107.31秒；首次Timeout等待值89.853496秒。虽然六个Run的runtime result均为succeeded，pytest没有进入后续断言，不能按完整native通过记账。

隔离baseline-cuda-checkout的7bc46b0同用例1通过、105.26秒；隔离cpu-checkout中本轮八成员逐字节相同的当前源码，同D:/anaconda3 Python、同原90秒用例1通过、104.02秒。隔离通过执行到真实固定BF16加载、六Run及后续篡改拒绝断言末尾；主树两次失败和partial事实仍保留。当前根因未定，不能称同失败preexisting、源变化已证明无因果或问题已根治。原CPU/CUDA与诊断门的通过另行记账。

下一步使用同一用例、配置、门限与来源，分别定位launch启动、执行、退出及父进程等待的耗时；不提高90秒、不过滤/跳过，也不重签首红或改旧Run。输入和四次对照由同日运行记录第五十八章及D:/Grok Bot/tmp/grokbot/sd/round55的native-bf16.log/xml、native-bf16-repeat.log/xml、baseline-native-bf16.log/xml、isolated-current-native-bf16.log/xml定位；未完成定位与同根复验前保持活动。
