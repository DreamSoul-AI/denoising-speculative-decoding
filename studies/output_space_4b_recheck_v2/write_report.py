from pathlib import Path
import argparse
import json
import hashlib

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--study',required=True)
args=parser.parse_args()
study=ROOT/'studies'/args.study
docs=study/'docs'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
x=read(docs/'execution_v1.json')
assert x['quality_passed'] and x['decoding_timing_disabled']
methods=('input','zero','one')
labels={'input':'输入冻结均值','zero':'输出零阶','one':'输出一阶'}
primary=args.study=='output_space_4b_recheck_v2'
lines=[f'# Qwen3-4B 输出空间配对复验报告\n\n2026-10-15，{x["n"]}条固定prompt；全部原生Run及Study质量门通过。\n\n',
       '## 结论\n\n']
pair=x['paired_summaries']
for p in pair:
    lines.append(f"{labels[p['second']]}相对{labels[p['first']]}：接受数净差{p['accepted_total_delta']:+d}，主生成模型调用数净差{p['main_model_call_delta']:+d}；逐prompt接受增／平／减为{p['accepted_positive']}/{p['accepted_zero']}/{p['accepted_negative']}。\n\n")
lines.append(('在本批固定输入上，零阶与一阶相对输入基线均有总接受收益。' if x['H1_both_output_orders_gain_over_input'] else
    '本批结果未同时支持“输出零阶与一阶都比输入基线有总接受收益”。')+
    '这是当前同树／同prompt／同模型条件的事实，不代表各prompt均改善、速度提高或统计泛化。\n\n')
lines += ['## 协议与历史证据\n\n',
    '[冻结PLAN](<../../../docs/研究计划 Denoising Speculative Decoding.md#legacy-studies-output-space-4b-recheck-v2-docs-plan>)与[输入清单](prompts_manifest.json)：Qwen3-4B，FP32/eager/CUDA，T0，EOS/max100，非thinking，seed123+prompt序号；三方法同unpruned静态[1,1]、k2/BC9。输入基线是本研究ESP冻结输入均值；输出beta=.9、beta_prime=.99、缩放映射保持原科学实现。实际输入/输出共享权重由两output decoder的storage门核实；四Run实际model_digest相同。\n\n',
    '[历史检索](history_inventory.json)找到的4B数字是输入空间方法：first26单槽680/704/777，[7,2]双槽621/704，全480冻结均值12809与选定EMA15009；旧Run原始制品未保留，未找到能核验的旧output0/output1结果与配置。因此不宣称精确复现旧输出收益，亦不直接拿不同树的旧接受数与本次配对比较。旧输入EMA选定参数也不当作输出一阶参数。\n\n',
    '[模型快照](modelscope_snapshot.json)记录本机14文件SHA，文件名/大小与官方ModelScope列表一致，离线权重经Study shared junction复用。预存目录的原始下载来源及不可变remote revision未确认，不把名/大小相同当字节一致；实际模型摘要见[原生证据](execution_v1.json)。没有新下载或权重量化/卸载，直接CUDA载入同FP32模型仅避免完整host副本。\n\n',
    'eval.warmup=false；Study局部适配在execute期间替换decoder模块时钟为恒定0，不读取解码clock，全局时间模块不变。原生兼容timing/throughput/speedup字段若为0均为禁用占位，不能解释为零延迟或性能数据。这里仅比较调用计数，不测速。\n\n',
    '## 配对数字\n\n',
    '| 方法 | 接受draft | 主生成模型调用 | accept_d1 | accept_d2 | 生成新token |\n| --- | ---: | ---: | ---: | ---: | ---: |\n']
for m in methods:
    t=x['totals'][m]
    lines.append(f"| {labels[m]} | {t['accepted']} | {t['model_calls']} | {t['accept_d1']} | {t['accept_d2']} | {t['new_tokens']} |\n")
lines.append(f"| 独立AR | 不适用 | {x['AR_main_model_calls']} | 不适用 | 不适用 | {x['totals']['input']['new_tokens']} |\n\n")
lines.append('主生成调用：spec=prefill_calls+decode_calls，AR=ar_calls；每个spec额外独立AR审计计入实验总预算，排除在主路径配对调用数之外。accepted沿用预算/EOS最终裁剪前draft验证计数，不能将accepted差自动等同实际省下的调用差。所有三种输出逐token等于自身AR及第四独立AR Run；没有为了匹配而裁剪或改变输入。\n\n')
lines.append('| qid | 类别 | 输入 接受/调用 | 零阶 接受/调用 | 一阶 接受/调用 | AR调用 |\n| --- | --- | ---: | ---: | ---: | ---: |\n')
for row in x['paired_rows']:
    values=[f"{row['cells'][m]['accepted']}/{row['cells'][m]['model_calls']}" for m in methods]
    lines.append(f"| {row['question_id']} | {row['category']} | {' | '.join(values)} | {row['AR_main_model_calls']} |\n")
lines.append('\n配对接受差分统计（sample stdev，固定非随机prompt集合，不作显著性判决）：\n\n| 配对 | n | mean | stdev | min | max |\n| --- | ---: | ---: | ---: | ---: | ---: |\n')
for p in pair:
    lines.append(f"| {labels[p['second']]}−{labels[p['first']]} | {p['n']} | {p['accepted_mean_delta']:.4f} | {p['accepted_stdev_delta']:.4f} | {p['accepted_min_delta']} | {p['accepted_max_delta']} |\n")
lines.append('\n![实际配对接受数与调用数](figures/paired_counts_v1.png)\n\n图依据execution_v1.json逐prompt生成，左图接受差、右图主路径调用差；横向正负均保留，右图负值只代表更少调用，不是实测加速。\n\n')
lines.append('## 验收、失败保全与边界\n\n')
lines.append(f"四Run全部succeeded、source/model/input/完整性/AR硬门通过；科学source仍{x['source_digest']}，package/default未改。实际生成新token（含三份内部AR审计）{x['total_generated_tokens_including_audits']}，实际生成forward总数{x['total_generation_model_forwards_including_audits']}，零teacher、零warmup。完整Runs/cache/weights在本机，formal证据随本地提交，不推送/PR/合并。\n\n")
if primary:
    lines.append('[首次激活失败记录](first_run_failure.json)：v1基线在26条execute完成/全部AR相同后因Study collect入口导出模块而失败；原Run与锁未重签，v2仅修正原生phase callable导出，以新版本完成上述验收。旧失败额外26×spec/AR生成单列，不混作新的独立seed或成功Run。8项phase入口回归防止重现；关闭计时/调用语义与注册恢复另有回归。\n\n')
    lines.append('0.6B rare／单位置机理分析已暂停，结果归档原报告与1274bb2；其有限窗口结论不能代替本4B配对验收。此报告写回后继续预声明[4条新prompt确认](<../../../docs/研究计划 Denoising Speculative Decoding.md#legacy-studies-output-space-4b-holdout-docs-plan>)，不选择有利prompt或改参数。\n')
else:
    lines.append('这些4条是first26之外固定summarization/translation skip2/take2，按主批次启动前协议选择，不因主批次收益替换。与[主first26报告](STUDY_REPORT.md)分层，不当额外独立seed或总体显著性证据。下一步应在4B扩大固定任务范围或配方对照；当前不恢复0.6B罕见位置分析。\n')
lines.append('\n未验证：未定位旧output-space4B正结果的完整配方，历史最优输入EMA的同链配对，多seed采样／多设备／更长预算／速度与全量任务泛化。不能从调用减少直接推断加速。\n')
(docs/'STUDY_REPORT.md').write_text(''.join(lines),encoding='utf-8',newline='\n')

import torch  # noqa: F401 - Windows torch-first
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(13,5))
for j,m in enumerate(('zero','one')):
    offset=(j-.5)*.36
    positions=[i+offset for i in range(x['n'])]
    axes[0].bar(positions,[r['cells'][m]['accepted']-r['cells']['input']['accepted'] for r in x['paired_rows']],width=.34,label=m)
    axes[1].bar(positions,[r['cells'][m]['model_calls']-r['cells']['input']['model_calls'] for r in x['paired_rows']],width=.34,label=m)
for ax in axes:
    ax.axhline(0,color='black',linewidth=.7)
    ax.set_xticks(range(x['n']),[str(r['question_id']) for r in x['paired_rows']],rotation=90 if x['n']>10 else 0)
    ax.set_xlabel('Fixed question ID')
    ax.legend()
axes[0].set_title('Accepted drafts: output minus matched input')
axes[1].set_title('Main model calls: output minus matched input')
fig.suptitle('Qwen3-4B / FP32 / T0 / two-slot chain / exact AR outputs; no timing')
fig.tight_layout()
folder=docs/'figures'
folder.mkdir(exist_ok=True)
target=folder/'paired_counts_v1.png'
assert not target.exists()
fig.savefig(target,dpi=150)
plt.close(fig)
(folder/'source_v1.json').write_text(json.dumps({'execution_sha256':hashlib.sha256((docs/'execution_v1.json').read_bytes()).hexdigest(),
    'figure_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'plot_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'new_model_forwards':0,'timing_measured':False},indent=2)+'\n',encoding='utf-8')
print('report and actual paired count figure written',study)
