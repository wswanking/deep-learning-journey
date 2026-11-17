"""Original vector evaluation diagrams, toy statistics and actual frozen-model outcomes."""
from pathlib import Path
from xml.sax.saxutils import escape
import json
BLUE,ORANGE,GREEN,INK,GRID='#246cc0','#bd611a','#39805b','#172c44','#d0dbe7'
def text(x,y,value,size=22,extra=''):return f'<text x="{x}" y="{y}" font-size="{size}" {extra}>{escape(str(value))}</text>'
def line(x1,y1,x2,y2,color=GRID,width=1.5,extra=''):return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" {extra}/>'
def rect(x,y,w,h,fill='#fff',extra=''):return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" {extra}/>'
def path(points,color,width=3):return '<path d="'+' '.join(f'{"M" if i==0 else "L"} {x:.3f} {y:.3f}' for i,(x,y) in enumerate(points))+f'" fill="none" stroke="{color}" stroke-width="{width}"/>'
def box(x,y,w,h,title,sub):return rect(x,y,w,h,'#fff','rx="12" stroke="#8a9eb4" stroke-width="2"')+text(x+w/2,y+34,title,24,'text-anchor="middle" font-weight="700"')+text(x+w/2,y+68,sub,21,'text-anchor="middle"')
def save(folder,name,h,title,desc,body):
    source=f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{h}" viewBox="0 0 900 {h}" role="img" aria-labelledby="c27-{name}-title c27-{name}-desc"><title id="c27-{name}-title">{escape(title)}</title><desc id="c27-{name}-desc">{escape(desc)}</desc><defs><marker id="c27-{name}-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{BLUE}"/></marker></defs>'+rect(0,0,900,h,'#f7f9fc','rx="16"')+'<g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" fill="#172c44">'+body+'</g></svg>\n'
    (folder/f'{name}.svg').write_text(source,encoding='utf-8')


import math
def write_figures(report,folder):
    folder.mkdir(parents=True,exist_ok=True)
    body=text(30,42,'评估先写合同：模型、输入、指标与停止规则一起固定',25,'font-weight="700"')+text(30,82,'相同模型也会因模板、答案归一化、字节预算或数据切片得到不同分数。',21)
    marker='marker-end="url(#c27-evaluation-contract-arrow)"'
    boxes=[(30,130,370,'① 冻结评估对象','权重 / tokenizer / 来源SHA'),(500,130,370,'② 固定数据和协议','分组 / 模板 / 8字节 / LF停止'),(30,295,370,'③ 保存所有逐例结果','字节 / 终止 / NLL / 耗时'),(500,295,370,'④ 按切片汇总与比较','分母 / 基线 / 组重抽范围')]
    for x,y,w,a,b in boxes:body+=box(x,y,w,100,a,b)
    body+=line(400,180,495,180,BLUE,3,marker)+line(685,230,685,265,BLUE,3)+line(685,265,215,265,BLUE,3)+line(215,265,215,290,BLUE,3,marker)+line(400,345,495,345,BLUE,3,marker)
    body+=box(30,475,840,100,'⑤ 定位失败与决定下一轮开发','新假设使用开发数据；不根据这批测试答案调参后继续称盲测')
    body+=text(30,630,'本章不训练；22既有测试复评与新增域外诊断分别报告，不包装成新排行榜。',21)
    save(folder,'evaluation-contract',670,'图27-1：冻结对象与评估协议的完整流程','原创评估合同流程，冻结权重数据模板和停止规则，保存逐例结果，按切片基线与分组比较，再定位失败，不用反复测试反馈作未披露调参。',body)

    body=text(30,42,'混淆矩阵手算：先声明正类，再计算分母',28,'font-weight="700"')+text(30,82,'教学虚构例：真实正类表示应触发告警；共20个示例，仅用于手算。',21)
    for x,label in [(340,'预测正'),(620,'预测负')]:body+=text(x,145,label,25,'text-anchor="middle" font-weight="700"')
    for y,label in [(245,'真实正'),(365,'真实负')]:body+=text(130,y,label,25,'text-anchor="middle" font-weight="700"')
    for x,y,value in [(220,175,'TP = 8'),(500,175,'FN = 4'),(220,295,'FP = 2'),(500,295,'TN = 6')]:body+=rect(x,y,260,100,'#fff','rx="10" stroke="#b6c6d8"')+text(x+130,y+60,value,29,'text-anchor="middle"')
    body+=text(30,470,'准确率=(8+6)/20=70%；精确率=8/(8+2)=80%。',24)+text(30,520,'召回率=8/(8+4)=66.7%；F1=16/(16+2+4)=72.7%。',24)+text(30,570,'没有预测正类时精确率分母为0；本章记null，而不是100%。',22)
    save(folder,'confusion',610,'图27-2：二分类混淆矩阵与准确率精确率召回率F1','原创二十例手算矩阵，真实作为行预测作为列，八真阳两假阳四假阴六真阴，展示不同分母和零分母不适用。',body)

    summaries=report['language_models']['summaries'];body=text(30,42,'真实任务切片：格式变好，与正确率变好是两件事',27,'font-weight="700"')+text(30,82,'既有22测试24例；新增g..l域外12例；均固定8字节上限与LF停止。',21)
    for panel,(suite,title) in enumerate([('existing_22_test','既有测试：a..f'),('new_alphabet_diagnostic','新字母诊断：g..l')]):
        left=70+panel*440;body+=text(left,140,title,25,'font-weight="700"')
        for i,(key,label,color) in enumerate([('base','基座',BLUE),('sft','SFT',ORANGE),('echo_baseline','原样输出',GREEN)]):
            x=left+i*115;v=summaries[key][suite]['exact_match'];body+=rect(x,405-v*400,75,max(1,v*400),color,'rx="4"')+text(x+37,388-v*400,f'{v:.0%}',24,'text-anchor="middle"')+text(x+37,438,label,21,'text-anchor="middle"')
        body+=line(left-10,405,left+340,405,INK,2)
    body+=text(30,505,'SFT既有测试格式100%，答案只有6/24；新字母格式100%，答案0/12。',22)+text(30,552,'原样输出基线只会复制，不会反转；平衡两任务使它恰好得到50%。',22)+text(30,599,'简单基线更高也是结果；两套数据不能拼成一个“通用能力分”。',23)
    save(folder,'task-slices',640,'图27-6：冻结基座SFT与原样输出基线按数据切片的真实EM','真实两任务切片柱图，基座均零，SFT既有测试百分之二十五域外零，原样输出基线两套百分之五十，格式与内容分开。',body)

    comp=report['language_models']['paired_group_comparison']['existing_22_test'];body=text(30,42,'同一组同时比较两个模型，再重抽组',28,'font-weight="700"')+text(30,82,'既有测试只有3个无序字母组，每组8个变体；不能当成24个无关问题。',21)
    left,top,w,h=190,145,630,220;sx=lambda x:left+x/.5*w
    for value in [0,.125,.25,.375,.5]:body+=line(sx(value),top,sx(value),top+h)+text(sx(value),top+h+30,f'{100*value:g}%',21,'text-anchor="middle"')
    for i,(label,value) in enumerate(comp['differences_by_group'].items()):
        y=190+i*70;body+=text(170,y+7,label,23,'text-anchor="end"')+line(left,y,sx(value),y,BLUE,5)+f'<circle cx="{sx(value)}" cy="{y}" r="7" fill="{BLUE}"/>'
    body+=text(30,455,'平均差=SFT−基座=25个百分点；2000次组重抽的2.5%~97.5%范围',22)+text(30,500,'为12.5~37.5个百分点。仅有3个合成组，不能保证总体95%覆盖率。',22)+text(30,545,'新字母各组两模型均错：重抽区间[0,0]，也不表示真实差异必然为0。',21)+text(30,590,'配对保留“同一问题”的相关性；按组重抽保留组内变体关系。',23)
    save(folder,'paired-groups',630,'图27-4：按组配对比较与小样本bootstrap边界','真实三个字母组SFT减基座差值分别二十五十二点五三十七点五个百分点，组bootstrap范围十二点五至三十七点五，只做描述性范围，不保证总体覆盖或域外零差。',body)

    body=text(30,42,'校准手算：信心高不代表答得对',28,'font-weight="700"')+text(30,82,'独立的二分类虚构例：正类概率[0.9,0.8,0.6,0.1]，标签[1,0,1,0]。',21)
    for i,b in enumerate(report['hand']['bins']):
        x=165+i*400;body+=rect(x,405-b['confidence']*250,100,b['confidence']*250,BLUE,'rx="5"')+rect(x+115,405-b['accuracy']*250,100,b['accuracy']*250,ORANGE,'rx="5"')+text(x+50,388-b['confidence']*250,f'{b["confidence"]:.3f}',22,'text-anchor="middle"')+text(x+165,388-b['accuracy']*250,f'{b["accuracy"]:.3f}',22,'text-anchor="middle"')+text(x+105,443,['[0.5,0.75)','[0.75,1]'][i],24,'text-anchor="middle"')+text(x+105,480,f'n={b["count"]}',22,'text-anchor="middle"')
    body+=line(80,405,830,405,INK,2)+line(40,535,85,535,BLUE,4)+text(95,543,'平均预测类别信心',22)+line(460,535,505,535,ORANGE,4)+text(515,543,'实际类别准确率',22)
    body+=text(30,594,'二元Brier=(0.01+0.64+0.16+0.01)/4=0.205；这两个桶的ECE=0.25。',21)+text(30,641,'分桶改变ECE；小样本有噪声。它不是本章语言模型的实测校准曲线。',21)
    save(folder,'calibration-toy',680,'图27-3：虚构二分类概率的Brier与分桶校准手算','原创四例校准演示，两个桶的平均预测类别信心和准确率，二元Brier零点二零五，ECE零点二五。明确不是LM实测校准结论。',body)

    body=text(30,42,'RAG错误按证据链定位：规则读者与真实LM分别报告',25,'font-weight="700"')+text(30,82,'原始26章冻结测试：12题有答案+4题无资料；引用均核对原文区间。',21)
    for y,title,sub in [(135,'规则抽取器：12个有支持答案 + 4个正确拒答','单实体/字段语法限定；不代表神经模型已会读文档'),(290,'真实35k字节模型：12题证据在上下文，仍全部答错','reader_error=12；保留全部实际字节与非法UTF-8'),(445,'受控诊断：删去正确来源 → retrieval_miss','另将上下文预算设0 → context_lost；是主动干预，不混入自然测试')]:body+=box(30,y,840,110,title,sub)
    body+=text(30,620,'候选未命中、上下文丢失、读错、引用不支持、无证据乱答需要不同检查。',21)
    save(folder,'rag-failures',660,'图27-5：实际RAG组件结果与明确受控失效注入','规则读者十二回答四拒答成功，实际相同基座十二有证据仍错误；另主动删除来源和设零上下文验证错误分类，干预不计入自然测试成绩。',body)

    body=text(30,42,'单次本机质量—成本点图：输出长度影响耗时',27,'font-weight="700"')+text(30,82,'只画既有22测试切片；横轴为单次预热后中位耗时，毫秒，对数刻度。',21)
    left,top,w,h=100,145,715,270;latencies=[summaries[k]['existing_22_test']['latency_ms_median'] for k in ['base','sft','echo_baseline']];low_power=math.floor(math.log10(min(latencies)));high_power=max(low_power+1,math.ceil(math.log10(max(latencies))));sx=lambda x:left+(math.log10(x)-low_power)/(high_power-low_power)*w;sy=lambda y:top+h-y/.6*h
    for value in [10**p for p in range(low_power,high_power+1)]:body+=line(sx(value),top,sx(value),top+h)+text(sx(value),top+h+30,f'{value:g}',21,'text-anchor="middle"')
    for value in [0,.25,.5]:body+=line(left,sy(value),left+w,sy(value))+text(left-10,sy(value)+7,f'{value:.0%}',21,'text-anchor="end"')
    for key,label,color,dy in [('base','基座',BLUE,-15),('sft','SFT',ORANGE,-15),('echo_baseline','原样输出',GREEN,-15)]:
        m=summaries[key]['existing_22_test'];x=sx(m['latency_ms_median']);y=sy(m['exact_match']);body+=f'<circle cx="{x}" cy="{y}" r="8" fill="{color}"/>'+text(x-12,y+dy,f'{label} {m["latency_ms_median"]:.4f}ms',21,'text-anchor="end"' if key=='base' else '')
    body+=text(30,505,'基座每例用满8字节，SFT通常3字节就停止；不能据此说SFT每步算子更快。',21)+text(30,551,'规则基线不运行神经网络；计时含张量创建和解码，不含训练与模型加载。',21)+text(30,597,'样本仅24例、一次进程；不外推服务P95、并发吞吐或CPU峰值性能。',22)
    save(folder,'quality-cost',640,'图27-7：真实固定任务EM与一次预热中位CPU延迟','实际基座SFT原样输出基线在既有测试上的质量和毫秒延迟对数点图，输出长度不同且只有单次小样本，不声称服务性能或每token提速。',body)

if __name__=='__main__':
    root=Path(__file__).resolve().parents[2];r=json.loads((root/'build/experiments/27-model-evaluation/results.json').read_text(encoding='utf-8'));write_figures(r,root/'docs/assets/27-model-evaluation')
