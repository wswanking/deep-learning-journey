"""Editable pure SVG capstone evidence diagrams and measured learning curves."""
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
    source=f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{h}" viewBox="0 0 900 {h}" role="img" aria-labelledby="c30-{name}-title c30-{name}-desc"><title id="c30-{name}-title">{escape(title)}</title><desc id="c30-{name}-desc">{escape(desc)}</desc><defs><marker id="c30-{name}-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{BLUE}"/></marker></defs>'+rect(0,0,900,h,'#f7f9fc','rx="16"')+'<g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" fill="#172c44">'+body+'</g></svg>\n'
    (folder/f'{name}.svg').write_text(source,encoding='utf-8',newline='\n')



import math
def write_figures(report,folder):
    folder.mkdir(parents=True,exist_ok=True)
    body=text(30,42,'一个小项目，两条独立的证据路线',28,'font-weight="700"')+text(30,82,'统一配置与交付记录；语言模型质量和规则检索质量分别计分。',22)
    for x,y,w,a,b in [(30,130,250,'原创文本','21章语料与分组'),(350,130,240,'随机初始化训练','35,328参 / 600步'),(660,130,210,'回答区间SFT','800步 / 验证选点'),(30,320,250,'原创设施文档','12篇 / 版本与位置'),(350,320,240,'TF-IDF索引','验证选切分 / 保存'),(660,320,210,'受限规则读者','抽取 / 引用 / 拒答')]:body+=box(x,y,w,100,a,b)
    marker='marker-end="url(#c30-project-flow-arrow)"'
    for y in [180,370]:body+=line(280,y,345,y,BLUE,3,marker)+line(590,y,655,y,BLUE,3,marker)
    body+=box(30,510,840,105,'统一评估与证据包','固定36条语言任务 + 16条资料问答；保留真实神经读者失败')
    body+=text(30,668,'不把规则答案当成LM生成；视觉分支可另行接入，并有自己的数据与指标。',21)
    save(folder,'project-flow',710,'图30-1：综合项目的文本训练与资料检索两条路线','原创项目结构，文本随机初始化预训练与SFT独立于资料索引和规则抽取，两者共享可追溯评估与交付，不把规则成功冒充语言模型生成。',body)

    body=text(30,42,'优化事件账本：预算相加，质量指标不能混加',27,'font-weight="700"')+text(30,82,'本章从零重建并独立验证一次237步恢复；SFT每例监督2字母+LF。',21)
    rows=[('主预训练',600,8,64,307200),('恢复核验额外路线',600,8,64,307200),('回答区间SFT',800,16,3,38400)]
    for x,v in zip([35,345,460,575,710],['阶段','更新步','批大小','目标/例','目标事件']):body+=text(x,145,v,23,'font-weight="700"')
    for i,row in enumerate(rows):
        y=215+i*85;body+=rect(25,y-38,850,65,'#fff','rx="8"')
        for x,v in zip([35,345,460,575,710],row):body+=text(x,y,f'{v:,}' if isinstance(v,int) else v,22)
    body+=text(30,525,'总计652,800个优化目标事件，包含大量重复，不是不同文本字节数。',23)+text(30,575,'训练的目标范围不同；该合计用于核算工作，不是可以平均的能力分数。',22)+text(30,625,'恢复额外路线237+363=600步；主模型与恢复最终参数实测完全相同。',22)
    save(folder,'budget-ledger',665,'图30-3：真实固定训练预算与独立恢复核验工作账本','六百步八乘六十四预训练三十万七千二百事件，恢复另算同样预算，八百步十六乘三回答SFT三万八千四百，总六十五万二千八百，重复与目标差异明确。',body)

    history=report['training']['pretraining_history'];sft=report['training']['sft'];body=text(30,42,'本章独立重训的真实曲线：终点与选点规则事先固定',26,'font-weight="700"')+text(30,82,'上：预训练按整份切片计分；下：只按SFT回答区间计分，不共用纵轴含义。',21)
    left,w=85,740
    def panel(top,h,maxx,maxy,title,series):
        sx=lambda x:left+x/maxx*w;sy=lambda y:top+h-y/maxy*h
        value=text(left,top-22,title,23,'font-weight="700"')
        for y in [0,maxy/2,maxy]:value+=line(left,sy(y),left+w,sy(y))+text(left-10,sy(y)+7,f'{y:g}',20,'text-anchor="end"')
        for x in [0,maxx/2,maxx]:value+=line(sx(x),top,sx(x),top+h)+text(sx(x),top+h+26,f'{x:g}',20,'text-anchor="middle"')
        for i,(pairs,color) in enumerate(series):
            curve=path([(sx(x),sy(y)) for x,y in pairs],color)
            value+=curve.replace('/>',' stroke-dasharray="9 5"/>') if i==1 else curve
        value+=text(left+w/2,top+h+51,'更新步',20,'text-anchor="middle"')
        return value
    body+=panel(145,150,600,6,'预训练CE（nat/byte）：训练实线 / 验证虚线；固定600步',[( [(r['step'],r['train']['cross_entropy_nat_per_byte']) for r in history],BLUE),([(r['step'],r['validation']['cross_entropy_nat_per_byte']) for r in history],ORANGE)])
    curves=sft['curves'];maxy=max(curves['validation_response_nll'])*1.1;rounded=math.ceil(maxy)
    body+=panel(410,150,800,rounded,f'SFT回答NLL（nat/byte）；验证最低点{ sft["best_update"] }步',[(list(zip(curves['update'],curves['validation_response_nll'])),GREEN)])
    body+=text(30,635,'不同任务损失不能直接比较；验证选点之后才报告固定测试，失败同样保留。',21)
    save(folder,'training-curves',675,'图30-4：本章独立训练的预训练与SFT验证曲线','读取本章真实重训记录，预训练固定六百步；SFT八百步过程中每五十步评价回答验证NLL并恢复最低检查点，两个纵轴计分范围不同。',body)

    body=text(30,42,'交付可复现，与应用质量达标分别验收',28,'font-weight="700"')+text(30,82,'门槛是本章回顾性项目要求；前章结果已知，不伪称新盲测预注册。',21)
    for i,gate in enumerate(report['quality_review']['gates']):
        y=145+i*85;body+=rect(25,y-25,850,65,'#fff','rx="8"')
        labels=['SFT既有任务','SFT新字母','神经证据问答','受限规则问答','受限规则拒答'];body+=text(40,y+15,labels[i],23,'font-weight="700"')+text(330,y+15,f'实测 {gate["value"]:.0%}',24)+text(510,y+15,f'门槛 {gate["minimum"]:.0%}',23)+text(735,y+15,'满足' if gate['passed'] else '未满足',24,f'fill="{GREEN if gate["passed"] else ORANGE}"')
    status=report['quality_review']['status'];conclusion='当前结果满足本章回顾性门槛' if report['quality_review']['ready_for_stated_application'] else '当前结果未满足全部应用门槛'
    body+=text(30,645,f'工程核对通过；{conclusion}，项目结论为{status}。',21)
    save(folder,'quality-review',685,'图30-6：工程交付与回顾性应用质量门槛分别记录','五个明确质量门槛，显示实际指标、门槛及逐项审核结果，数值不可混成总分；工程运行成功并不自动表示应用可用。',body)

    cases={r['id']:r for r in report['language']['cases']};bad=next((key,value) for key,value in report['language']['outputs']['sft'].items() if not value['exact_match'] and cases[key]['suite']=='existing_22_test');ood=next((key,value) for key,value in report['language']['outputs']['sft'].items() if cases[key]['suite']=='new_alphabet_diagnostic');probe=report['neural_evidence']['base']['pairs'][0]
    body=text(30,42,'失败是交付证据：保留输入、输出和阶段边界',27,'font-weight="700"')
    for i,(title,a,b) in enumerate([('既有任务：格式正确但答案错误',f'指令：{cases[bad[0]]["instruction"]}；目标：{cases[bad[0]]["response"]}',f'实际输出：{bad[1]["text"]}；已LF停止，严格EM=False'),('新字母：输出范围没有泛化',f'指令：{cases[ood[0]]["instruction"]}；目标：{cases[ood[0]]["response"]}',f'实际输出：{ood[1]["text"]}；格式合格但答案错'),('有真实证据：微型模型仍未正确回答',f'问题：{probe["question"]}；答案：{probe["gold"]}',f'输入含完整证据；实际字节前8项：{probe["with_evidence"]["new_byte_ids"][:8]}')]):
        yy=120+i*160;body+=rect(30,yy,840,140,'#fff','rx="10" stroke="#b6c6d8"')+text(50,yy+34,title,25,'font-weight="700"')+text(50,yy+80,a,22)+text(50,yy+119,b,21)
    body+=text(30,660,'所有案例均来自本章独立运行的JSON；不删非法字节，也不把规则输出替换进去。',21)
    save(folder,'failure-cases',700,'图30-5：独立项目运行中SFT和神经问答的真实失败案例','从实际全部输出选择一个既有错误和一个域外错误，另列有完整资料的真实基座问答失败，用原始字节避免掩盖非法编码，不替换为规则成功。',body)

    body=text(30,42,'证据包：源文件指纹与产物指纹是两张账本',27,'font-weight="700"')+text(30,82,'SHA256标识确切文件，不证明语义独立，也不保证换版本得到相同数值。',21)
    for x,y,w,title,sub in [(30,135,390,'sources','代码 / JSONL / 配置 / 依赖'),(480,135,390,'artifacts','权重 / 索引 / JSON / SVG'),(30,320,390,'report.md → HTML / PDF','同一Markdown + 纯SVG'),(480,320,390,'manifest + verification','逐文件复算；不包含自身循环摘要')]:body+=box(x,y,w,110,title,sub)
    body+=text(30,520,'检查点文件可以因元数据不同而SHA不同；另记state_sha256识别具体参数。',21)+text(30,570,'只读前章代码；本章训练与报告全部落入build/experiments/30-capstone。',21)+text(30,620,'重新推理与重新训练分别记录；replay不会冒充本次又完成了800步更新。',22)
    save(folder,'evidence-package',660,'图30-2：源与产物指纹、同源出版及重训重放的证据结构','原创证据包图，sources和artifacts记录不同对象，报告同Markdown生成HTMLPDF，manifest排除自身避免循环摘要，状态参数摘要与文件摘要语义区分。',body)

if __name__=='__main__':
    root=Path(__file__).resolve().parents[2];r=json.loads((root/'build/experiments/30-capstone/results.json').read_text(encoding='utf-8'));write_figures(r,root/'docs/assets/30-capstone')
