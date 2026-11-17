"""Pure SVG explanations and real DPO curves; no raster embedding."""
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
    source=f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{h}" viewBox="0 0 900 {h}" role="img" aria-labelledby="c24-{name}-title c24-{name}-desc"><title id="c24-{name}-title">{escape(title)}</title><desc id="c24-{name}-desc">{escape(desc)}</desc><defs><marker id="c24-{name}-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{BLUE}"/></marker></defs>'+rect(0,0,900,h,'#f7f9fc','rx="16"')+'<g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" fill="#172c44">'+body+'</g></svg>\n'
    (folder/f'{name}.svg').write_text(source,encoding='utf-8')

import math
def write_figures(report,folder):
    folder.mkdir(parents=True,exist_ok=True)
    body=text(30,42,'偏好数据相同，优化路线可以不同',28,'font-weight="700"')+text(30,82,'SFT给示范答案；偏好对给同一提示下两个回答的相对选择。',22)
    body+=box(30,130,250,90,'SFT策略 πref','复制为固定参考')+box(360,130,510,90,'比较数据 (x, yw, yl)','记录标签来源、标准与分组划分')+line(280,175,355,175,BLUE,3,'marker-end="url(#c24-pipeline-arrow)"')
    body+=box(30,295,840,115,'显式奖励 + RLHF 路线','拟合奖励差 → 用RL更新策略；期望奖励减βKL')
    body+=box(30,475,840,115,'本章实际执行：离线DPO','固定参考；直接用两个回答的对数概率比训练策略')
    body+=text(30,650,'本章标签来自首/末字母确定规则；没有人类标注、PPO或独立奖励网络。',21)+text(30,697,'优化标签偏好不会自动保证事实正确、格式正确或所有人的价值一致。',22)
    save(folder,'pipeline',735,'图24-1：SFT、显式奖励RLHF与离线DPO的关系','原创流程图，区分示范监督、偏好比较、先拟合奖励再用强化学习与直接偏好优化。本章只执行离线DPO，标签为确定字符串规则，不能称人类反馈实验。',body)

    body=text(30,42,'KL正则化的理想解：参考概率按奖励重新加权',27,'font-weight="700"')+text(30,82,'先用只有两个回答的完整有限空间，β=1，r=(1,0)，πref=(0.5,0.5)。',21)
    colors=[BLUE,ORANGE];vals=[[.5,.5],report['hand']['two_response_optimum']]
    for panel,(title,values) in enumerate(zip(['固定参考','理想最优策略'],vals)):
        left=80+panel*435;body+=text(left,135,title,25,'font-weight="700"')
        for i,value in enumerate(values):
            x=left+i*140;body+=rect(x,395-value*260,95,value*260,colors[i],'rx="5"')+text(x+47,430,['回答1','回答2'][i],22,'text-anchor="middle"')+text(x+47,380-value*260,f'{value:.4f}',23,'text-anchor="middle"')
        body+=line(left-10,395,left+270,395,INK,2)
    body+=text(30,498,'π*(y) = πref(y) exp(r(y)/β) / Z；Z使总概率恰好为1。',24)+text(30,547,'奖励更高的回答被增权；参考为0的回答不能靠这个比值获得支持。',22)+text(30,596,f'实测手算：KL={report["hand"]["two_response_kl"]:.6f}，E[r]−KL={report["hand"]["two_response_objective"]:.6f}。',23)+text(30,645,'本图是完整二回答玩具空间；实际LM实验没有精确计算完整回答空间KL。',21)
    save(folder,'kl-optimum',685,'图24-2：KL正则奖励目标的有限回答空间手算','原创两个回答概率柱图，参考各零点五，奖励一与零，beta一时最优概率零点七三一与零点二六九，KL零点一一一。强调归一化和支持条件。',body)

    hand=report['hand'];body=text(30,42,'DPO比较“相对参考的变化”，不是直接比较两个概率',26,'font-weight="700"')+text(30,82,'πθ(yw)=0.6、πθ(yl)=0.3；πref两者均0.4；剩余质量属于其他回答。',21)
    for i,(label,value) in enumerate([('偏好回答的概率比','0.6 / 0.4 = 1.5'),('非偏好回答的概率比','0.3 / 0.4 = 0.75')]):
        y=130+i*100;body+=rect(30,y,840,80,'#fff','rx="10" stroke="#b6c6d8"')+text(50,y+33,label,23,'font-weight="700"')+text(490,y+50,value,25)
    body+=text(30,378,'β=0.5：m=β[ln1.5−ln0.75]=0.5ln2=0.346574。',24)+text(30,426,f'偏好概率 σ(m)={hand["preference_probability"]:.6f}；损失−lnσ(m)={hand["loss"]:.6f}。',23)+text(30,474,'对logπθ(yw)的导数为−0.207107；对logπθ(yl)为+0.207107。',22)+text(30,522,'策略与参考完全相同时：两个比值均1，m=0，损失=ln2。',23)+text(30,570,'参数共享且还有其他回答：相对排序改善，不保证每条偏好回答绝对概率都上升。',21)
    save(folder,'dpo-margin',610,'图24-3：DPO对数概率比、间隔、概率与梯度手算','原创三回答合法概率例子，两个候选概率零点六与零点三，参考均零点四，beta零点五，margin零点三四六，损失零点五三五。展示独立对数概率偏导与参数耦合边界。',body)

    body=text(30,42,'两个候选各自做teacher forcing，只有回答字节计分',26,'font-weight="700"')+text(30,82,'示例提示“首字母 gh”；yw=g+换行，yl=h+换行；候选长度相同。',22)
    for i,(label,answer,color) in enumerate([('chosen','g',BLUE),('rejected','h',ORANGE)]):
        y=130+i*135;body+=box(30,y,440,90,'[指令] 首字母 gh [回答]','提示位置logits不计入回答分数')+box(550,y,320,90,f'{label}: {answer} + LF','恰好2个目标字节')+line(470,y+45,545,y+45,color,3,'marker-end="url(#c24-response-scoring-arrow)"')
    body+=text(30,455,'lnπ(y|x)=lnπ(首个回答字节|完整提示)+lnπ(LF|提示和回答字节)。',22)+text(30,505,'先按词表做log_softmax → gather目标 → mask → 沿回答位置求和。',22)+text(30,555,'参考与策略使用相同模板、分词、目标位置；本章不按回答长度求平均。',22)
    save(folder,'response-scoring',595,'图24-4：偏好回答对数概率的移位与计分范围','原创提示与两候选流程，每个回答为一个字母加换行，两目标字节概率求和，不计提示和补齐，参考与策略必须一致。',body)

    body=text(30,42,'三组真实训练曲线：翻转标签也可以把自己的损失降很低',25,'font-weight="700"')+text(30,82,'纵轴为各组实际优化标签下的平均DPO损失；横轴为更新步。',21)
    left,top,w,h=85,145,740,270;sx=lambda x:left+x/300*w;sy=lambda y:top+h-y/1.0*h
    for value in [0,.25,.5,.75,1.0]:body+=line(left,sy(value),left+w,sy(value))+text(left-10,sy(value)+7,f'{value:g}',21,'text-anchor="end"')
    for step in [0,100,200,300]:body+=line(sx(step),top,sx(step),top+h)+text(sx(step),top+h+30,step,21,'text-anchor="middle"')
    for result,color in zip(report['experiments'][1:],[BLUE,GREEN,ORANGE]):body+=path([(sx(r['step']),sy(r['train']['optimization_loss'])) for r in result['history']],color)
    for y,title,color in [(490,'β=1，真实合成标签',BLUE),(535,'β=0.1，真实合成标签',GREEN),(580,'β=1，故意翻转训练标签',ORANGE)]:body+=line(35,y-7,85,y-7,color,4)+text(100,y,title,22)
    body+=text(30,633,'低训练损失说明拟合了给定标签；标签标准错误时，优化不会自动修正它。',21)
    save(folder,'training-curves',675,'图24-5：真实DPO与翻转标签控制组训练损失','从本机结果JSON绘制三个配置每25步实际优化目标，翻转标签组同样快速降低其错误目标，不能用损失下降代替正确性评估。',body)

    body=text(30,42,'同一测试集：候选排序成绩，不等于自由生成正确答案',25,'font-weight="700"')+text(30,82,'12个未见组合提示；左列比较两个指定候选，右列实际贪心生成。',21)
    labels=['SFT参考','DPO β=1','DPO β=0.1','翻转训练标签'];headers=['配置','候选排序正确','相对变化正确','生成完全正确']
    for x,value in zip([35,305,500,690],headers):body+=text(x,145,value,22,'font-weight="700"')
    for i,(label,result) in enumerate(zip(labels,report['experiments'])):
        y=210+i*72;body+=rect(25,y-35,850,58,'#fff','rx="8"')
        vals=[label,f'{result["test"]["raw_candidate_pair_accuracy"]:.1%}',f'{result["test"]["relative_pair_accuracy"]:.1%}',f'{result["generation"]["exact_match"]:.1%}']
        for x,value in zip([35,335,530,730],vals):body+=text(x,y,value,23)
    body+=text(30,565,'排序指标与完整生成必须分开读取，不能互相替代。',23)+text(30,610,'表中数值来自本次运行；候选集合外也可能有概率质量。',22)+text(30,655,'不是人类偏好胜率；也不是完整策略KL或通用能力评估。',23)
    save(folder,'ranking-generation',695,'图24-6：真实候选排序、相对间隔与自由生成的分离','四个固定模型对同一十二测试提示，记录原始候选排序、扣除参考后的相对排序、实际贪心完整生成。所有指标读取本次结果JSON，不使用历史运行的硬编码数值。',body)

if __name__=='__main__':
    root=Path(__file__).resolve().parents[2];report=json.loads((root/'build/experiments/24-preference-alignment/results.json').read_text(encoding='utf-8'));write_figures(report,root/'docs/assets/24-preference-alignment')
