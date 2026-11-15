"""Editable pure SVG figures generated from chapter-21 actual experiment JSON."""
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
    source=f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{h}" viewBox="0 0 900 {h}" role="img" aria-labelledby="c21-{name}-title c21-{name}-desc"><title id="c21-{name}-title">{escape(title)}</title><desc id="c21-{name}-desc">{escape(desc)}</desc><defs><marker id="c21-{name}-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{BLUE}"/></marker></defs>'+rect(0,0,900,h,'#f7f9fc','rx="16"')+'<g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" fill="#172c44">'+body+'</g></svg>\n'
    (folder/f'{name}.svg').write_text(source,encoding='utf-8')
def curve_panel(left,top,w,h,title,series,max_y=6):
    sx=lambda s:left+s/600*w;sy=lambda value:top+h-value/max_y*h
    body=text(left,top-20,title,24,'font-weight="700"')
    for value in [0,2,4,6]:body+=line(left,sy(value),left+w,sy(value))+text(left-10,sy(value)+7,value,20,'text-anchor="end"')
    for step in [0,200,400,600]:body+=line(sx(step),top,sx(step),top+h)+text(sx(step),top+h+27,step,20,'text-anchor="middle"')
    for values,color in series:body+=path([(sx(s),sy(v)) for s,v in values],color)
    return body

def write_figures(report,folder):
    folder.mkdir(parents=True,exist_ok=True)
    body=text(30,42,'一次移位：同一文档给出输入和下一字节目标',28,'font-weight="700"')+text(30,82,'原创手算示意：文本“红A\\n”编码为5个UTF-8字节。',22)
    columns=[('位置','1','2','3','4'),('输入ID','231','186','162','65'),('目标ID','186','162','65','10')]
    for row,values in enumerate(columns):
        for col,value in enumerate(values):
            x=35+col*166;y=120+row*65
            body+=rect(x,y,160,58,'#e7f0fb' if row==0 else '#fff','rx="7" stroke="#b6c6d8"')+text(x+80,y+38,value,24,'text-anchor="middle"')
    body+=text(35,368,'红 = [231,186,162]，A = [65]，换行 = [10]。',23)+text(35,414,'每个位置只见当前位置及此前输入，预测右移一位的目标。',22)+text(35,460,'右侧补齐时：输入填0、mask=False、目标=-100；0本身仍是合法字节。',21)
    save(folder,'byte-shift',500,'图21-1：UTF-8字节移位目标与补齐语义','红A和换行共五字节，输入前四字节对应后四字节目标。一个中文字符占多个字节，无BOS、EOS或保留PAD。补齐使用独立mask与忽略标签。',body)

    marker='marker-end="url(#c21-tiny-decoder-arrow)"'
    body=text(30,42,'35,328个可训练参数：完整的微型decoder-only模型',27,'font-weight="700"')+text(30,82,'B=批大小，T≤64，d=32；4个头各8维；2个pre-LN残差块。',22)
    body+=box(30,130,250,90,'输入ID [B,T]','UTF-8字节 0..255')+box(350,130,520,90,'字节嵌入 + 可学习位置嵌入','[B,T,32]；8,192 + 2,048 参数')+line(280,175,345,175,BLUE,3,marker)
    body+=box(350,275,520,90,'LayerNorm → 因果多头注意力 → 残差','Q/K/V各32→32；只读真实前缀键')+line(610,220,610,270,BLUE,3,marker)
    body+=box(350,420,520,90,'LayerNorm → 32→64→32 GELU → 残差','以上两段构成1块；每块8,416参数')+line(610,365,610,415,BLUE,3,marker)
    body+=text(35,328,'两段重复2次',23,'font-weight="700"')+text(35,371,'不共享输入/输出权重',21)+text(35,414,'dropout=0',22)
    body+=box(350,565,520,90,'末尾LayerNorm → 线性词表头','[B,T,256] logits；64 + 8,192 参数')+line(610,510,610,560,BLUE,3,marker)
    body+=text(30,703,'注意力维度不含偏置；FFN含偏置；标准化各含缩放与平移。',22)+text(30,748,'10,240 + 2×8,416 + 64 + 8,192 = 35,328。',24)
    save(folder,'tiny-decoder',790,'图21-2：微型decoder结构与参数账本','输入字节和位置嵌入送入两层pre-LN因果自注意力及GELU前馈残差块，最终映射256字节logits。详细参数账本对应实际模型35328参数。',body)

    body=text(30,42,'恢复训练需要优化过程状态，而不只是推理权重',27,'font-weight="700"')+text(30,82,'相同CPU、版本、确定性设置；固定采样器与训练预算。',22)
    body+=box(30,130,840,95,'路线A：连续执行600步','最终参数、评估历史、下一批采样器状态')
    body+=box(30,275,370,95,'路线B：执行237步后保存','model / AdamW / next_step')+box(500,275,370,95,'读取状态后继续至600步','采样RNG / torch RNG / 协议')
    body+=line(400,320,495,320,BLUE,3,'marker-end="url(#c21-checkpoint-arrow)"')
    body+=text(30,430,f'实测：最终参数最大绝对差 = {report["resume"]["maximum_parameter_absolute_error"]:g}。',25,'font-weight="700"')+text(30,475,'两条评估历史逐项相同；最终采样器状态相同。',23)+text(30,520,'base-model.pt仅供推理/后续微调；step-237-checkpoint.pt才含恢复状态。',21)+text(30,565,'固定随机种子不等于跨硬件、跨PyTorch版本逐位可复现。',22)
    save(folder,'checkpoint',605,'图21-3：连续训练与保存恢复的实际相等核验','实测237步保存后恢复至600步，参数最大差零，评估历史与采样器状态相同。区分推理检查点与含优化器随机状态的训练检查点，限定同版本同硬件。',body)

    main=report['experiments'][0];small=report['experiments'][2]
    body=text(30,42,'真实学习曲线：每50步用完整划分做确定性评估',27,'font-weight="700"')+text(30,82,'纵轴为平均交叉熵，单位nat/byte；横轴为优化器更新步。',22)
    body+=curve_panel(80,140,745,150,'主模型：32篇训练文档',[( [(r['step'],r['train']['cross_entropy_nat_per_byte']) for r in main['history']],BLUE),([(r['step'],r['validation']['cross_entropy_nat_per_byte']) for r in main['history']],ORANGE)])
    body+=curve_panel(80,380,745,150,'同一模型：只给4篇训练文档',[( [(r['step'],r['train']['cross_entropy_nat_per_byte']) for r in small['history']],BLUE),([(r['step'],r['validation']['cross_entropy_nat_per_byte']) for r in small['history']],ORANGE)])
    body+=line(40,590,85,590,BLUE,4)+text(95,598,'训练集',22)+line(255,590,300,590,ORANGE,4)+text(310,598,'验证集',22)+text(30,643,'小数据组训练误差更低，但验证误差约1.60；不能只检查训练损失。',22)
    save(folder,'learning-curves',680,'图21-4：主模型与少数据组的真实训练验证曲线','由真实JSON绘制每50步交叉熵。主模型和少数据组都降训练损失，少数据组训练验证分离更大，学习曲线也非单调。',body)

    body=text(30,42,'相同步数与输入预算下：较深模型没有自动获胜',27,'font-weight="700"')+text(30,82,'三组均600步、B=8、T=64；累计307,200个预测目标，包含重复。',21)
    headers=['配置','参数量','训练CE','验证CE']
    for col,value in enumerate(headers):body+=text([35,395,555,715][col],145,value,24,'font-weight="700"')
    labels=['主模型：2层/32篇','消融：1层/32篇','少数据：2层/4篇']
    for i,(label,result) in enumerate(zip(labels,report['experiments'])):
        y=212+i*80;body+=rect(25,y-38,850,65,'#fff','rx="7"')
        final=result['history'][-1]
        for col,value in enumerate([label,f'{result["parameters"]:,}',f'{final["train"]["cross_entropy_nat_per_byte"]:.4f}',f'{final["validation"]["cross_entropy_nat_per_byte"]:.4f}']):body+=text([35,395,555,715][col],y,value,22)
    body+=text(30,500,'固定种子的一次小实验；参数量、优化难度和初始参数不同。',22)+text(30,547,'1层验证误差更低是本次配置事实，不能据此宣称1层普遍优于2层。',21)+text(30,594,'测试集只评估固定主模型一次：CE=1.002630，逐字节困惑度=2.725440。',21)
    save(folder,'ablation',635,'图21-5：层数与数据量配置对照及测试协议','真实最终步主模型两层35328参数，单层26912参数，少数据组同主模型参数量。单层本次验证误差更低；少数据训练误差低但验证更高。只测试主模型一次。',body)

    body=text(30,42,'采样把预测分布变成文本，但不会补足训练中缺少的任务',26,'font-weight="700"')+text(30,82,'下列均来自固定提示与完整预声明输出，不按好看程度挑选种子。',21)
    rows=[('The red / 贪心','star rests on the table. ... this green square.','末尾发生重复：quiarquiaroom is quia'),('红色 / 贪心与温度0.8','三角形放在桌上。记录写着：红色三角形。','房间很安静，窗边有一束光。'),('计算3+5= / 贪心','写着桌上。记录写着：红色方块。','房间很安静，窗边有一束光。')]
    rows[0]=('The red / 贪心','star rests on the table. ... this green square.','末尾发生重复：“quiarquiaroom is quia”')
    for i,(title,a,b) in enumerate(rows):
        yy=120+i*145;body+=rect(30,yy,840,125,'#fff','rx="10" stroke="#b6c6d8"')+text(50,yy+31,title,24,'font-weight="700"')+text(50,yy+70,a,21)+text(50,yy+105,b,21)
    body+=text(30,602,'示意节录处标“...” ；完整6条输出与字节ID保存在results.json。',21)+text(30,646,'英文颜色不一致、重复；算术提示未回答8；一种采样产生非法UTF-8。',21)+text(30,690,'困惑度低只说明本协议的下一字节预测好；不构成推理能力证据。',22)
    save(folder,'generation-boundaries',730,'图21-6：真实生成样例与语言模型能力边界','摘录预先固定的三种提示。英文生成变更颜色并重复，中文复现训练模板，算术提示转为无关句子，一种采样非法UTF8。完整所有输出保存在实验JSON，不能只挑成功文本。',body)

if __name__=='__main__':
    root=Path(__file__).resolve().parents[2];r=json.loads((root/'build/experiments/21-tiny-language-model/results.json').read_text(encoding='utf-8'));write_figures(r,root/'docs/assets/21-tiny-language-model')
