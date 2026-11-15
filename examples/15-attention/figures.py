"""Pure SVG diagrams and actual numerical charts for chapter 15."""
from pathlib import Path
import json
import math
from xml.sax.saxutils import escape
BLUE,ORANGE,GREEN='#246cc0','#bd611a','#39805b'
INK,GRID='#172c44','#d0dbe7'
def text(x,y,value,size=22,extra=''):return f'<text x="{x}" y="{y}" font-size="{size}" {extra}>{escape(str(value))}</text>'
def line(x1,y1,x2,y2,color=GRID,width=1.5,extra=''):return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" {extra}/>'
def rect(x,y,w,h,fill='#fff',extra=''):return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" {extra}/>'
def path(points,color,width=3,extra=''):
    data=' '.join(f'{"M" if i==0 else "L"} {x:.3f} {y:.3f}' for i,(x,y) in enumerate(points))
    return f'<path d="{data}" fill="none" stroke="{color}" stroke-width="{width}" {extra}/>'
def save(folder,name,height,title,desc,body):
    marker=f'<defs><marker id="c15-{name}-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{BLUE}"/></marker></defs>'
    source=f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" viewBox="0 0 900 {height}" role="img" aria-labelledby="c15-{name}-title c15-{name}-desc"><title id="c15-{name}-title">{escape(title)}</title><desc id="c15-{name}-desc">{escape(desc)}</desc>'+marker+rect(0,0,900,height,'#f7f9fc','rx="16"')+'<g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" fill="#172c44">'+body+'</g></svg>\n'
    (folder/f'{name}.svg').write_text(source,encoding='utf-8')

def write_figures(report,folder):
    folder.mkdir(parents=True,exist_ok=True)
    body=text(30,42,'单头注意力：先匹配，再按权重汇总内容',28,'font-weight="700"')+text(30,83,'Q的每一行提出一次查询；K与V的每一行是一对地址与内容。',22)
    boxes=[(30,135,180,92,'Q：n_q × d_k','查询向量'),(30,285,180,92,'K：n_k × d_k','匹配用的键'),(295,200,210,105,'QKᵀ / √d_k','n_q × n_k 分数'),(610,200,245,105,'按键轴softmax','每行权重和为1'),(30,440,235,90,'V：n_k × d_v','待汇总的内容'),(610,440,245,90,'O = A V','n_q × d_v 输出')]
    for x,y,w,h,title,sub in boxes:
        body+=rect(x,y,w,h,'#fff','rx="12" stroke="#8a9eb4" stroke-width="2"')+text(x+w/2,y+36,title,24,'text-anchor="middle" font-weight="700"')+text(x+w/2,y+73,sub,22,'text-anchor="middle"')
    marker='marker-end="url(#c15-qkv-flow-arrow)"'
    body+=path([(210,181),(250,181),(250,227),(290,227)],BLUE,3,marker)+path([(210,331),(250,331),(250,277),(290,277)],BLUE,3,marker)+line(505,252,605,252,BLUE,3,marker)
    body+=line(732,305,732,435,BLUE,3,marker)+line(265,485,605,485,BLUE,3,marker)+text(745,375,'A',25)
    body+=text(30,574,'mask在softmax之前应用；d_k与d_v可以不同。',23)
    save(folder,'qkv-flow',610,'图15-1：查询、键、值与输出的单头数据流','查询和键点积缩放生成每个查询对应所有键的分数，按最后键轴归一化得到权重，再与值矩阵相乘。注明所有形状与掩码位置。',body)

    hand=report['hand']; body=text(30,42,'手算权重热图：每一行独立归一化',28,'font-weight="700"')+text(30,83,'分数行为(ln2,0,0)与(0,ln2,0)；exp后分别为(2,1,1)、(1,2,1)。',21)
    left,top,cellw,cellh=160,155,175,95
    for j in range(3):body+=text(left+(j+.5)*cellw,137,f'键/值 {j+1}',23,'text-anchor="middle"')
    for i,row in enumerate(hand['weights']):
        body+=text(125,top+i*cellh+56,f'查询{i+1}',23,'text-anchor="end"')
        for j,value in enumerate(row):
            fill='#9dc2ef' if value==.5 else '#dbe9fb'
            body+=rect(left+j*cellw,top+i*cellh,cellw,cellh,fill,'stroke="#71869d"')+text(left+(j+.5)*cellw,top+i*cellh+57,f'{value:.2f}',29,'text-anchor="middle"')
        body+=text(735,top+i*cellh+56,'行和=1',23)
    body+=text(30,402,'V₁=(2,0)，V₂=(0,4)，V₃=(2,4)。颜色越深，权重越大。',23)
    body+=text(30,451,'O₁=0.5V₁+0.25V₂+0.25V₃=(1.5,2)。',24)
    body+=text(30,496,'O₂=0.25V₁+0.5V₂+0.25V₃=(1,3)。',24)
    body+=text(30,547,'这是人为构造、可逐项核对的输入；不是训练模型发现的语言关系。',22)
    save(folder,'hand-attention-heatmap',585,'图15-2：两查询三键的真实权重与手算输出','权重矩阵来自实际运行，第一行零点五零点二五零点二五，第二行零点二五零点五零点二五。按行加权同一组值，得到一点五二和一三两个输出。',body)

    body=text(30,42,'为什么除以√d_k：在明确随机假设下看数值尺度',27,'font-weight="700"')+text(30,83,'每个维度独立标准正态；每个维度配置512个查询、每查询8个键。',22)
    lefts=[85,515]; top=170;width=300;height=260
    rows=report['scaling']; dimensions=[r['key_dimension'] for r in rows]
    sx=lambda d,left:left+(math.log2(d)-1)/7*width
    for panel,left in enumerate(lefts):
        body+=text(left-35,137,'分数方差（对数轴）' if panel==0 else '平均权重熵（nat）',23,'font-weight="700"')
        sy=(lambda v:top+height-math.log10(v)/3*height) if panel==0 else (lambda v:top+height-v/2.1*height)
        ticks=[1,10,100,1000] if panel==0 else [0,1,2]
        for tick in ticks:body+=line(left,sy(tick),left+width,sy(tick))+text(left-10,sy(tick)+7,f'{tick:g}',20,'text-anchor="end"')
        for dimension in [2,8,32,128,256]:body+=text(sx(dimension,left),top+height+32,dimension,19,'text-anchor="middle"')
        for label,color,dash in [('raw',BLUE,''),('scaled',ORANGE,'stroke-dasharray="9 4"')]:
            key='score_variance' if panel==0 else 'mean_entropy_nat'
            pts=[(sx(r['key_dimension'],left),sy(r[label][key])) for r in rows]
            body+=path(pts,color,3,dash)
            for x,y in pts:body+=f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>'
    body+=text(30,504,'蓝实线：未缩放；橙虚线：除以√d_k。横轴d_k按log₂排布。',22)
    body+=text(30,549,'缩放后分数方差接近1，权重较少过度集中；不是“所有权重相等”。',21)
    body+=text(30,590,'只说明这个随机模型；训练后Q/K的相关性与尺度可能不同。',22)
    save(folder,'scaling-experiment',625,'图15-3：缩放前后的实际分数方差与注意力熵','固定随机种子，五个键维度各生成五百一十二组查询和八键，画出实际方差与平均熵。未缩放方差随维度上升，除以维度平方根后接近一，明确独立标准正态的前提。',body)

    body=text(30,42,'因果与padding掩码：只决定哪些键可以参与',28,'font-weight="700"')+text(30,83,'蓝色=允许，灰色×=禁止；行是查询位置，列是键位置，编号从1开始。',21)
    for left,title,matrix,padded in [(100,'长度4：因果mask',report['masks']['causal'],False),(535,'长度2：因果 AND 有效键',report['masks']['allow'][1],True)]:
        body+=text(left-40,132,title,24,'font-weight="700"')
        for j in range(4):body+=text(left+j*60+30,170,j+1,22,'text-anchor="middle"')
        for i,row in enumerate(matrix):
            body+=text(left-20,214+i*60,i+1,22,'text-anchor="end"')
            for j,allowed in enumerate(row):
                fill='#cce0f8' if allowed else '#e4e8ef'; body+=rect(left+j*60,180+i*60,60,60,fill,'stroke="#71869d"')+text(left+j*60+30,219+i*60,'1' if allowed else '×',25,'text-anchor="middle"')
        if padded:
            body+=rect(left,300,240,120,'none','stroke="#bd611a" stroke-width="3" stroke-dasharray="7 4"')+text(left+120,459,'橙框：padding查询行，另行忽略',20,'text-anchor="middle"')
    body+=text(30,511,'每个有效查询至少保留一个键；本实现拒绝整行全部禁止。',22)
    body+=text(30,554,'只屏蔽padding键不会自动把padding查询的输出设零。',22)
    body+=text(30,597,'实际干预：改变未来/填充值1000，前两位置输出差与相应梯度均为0。',21)
    save(folder,'causal-padding-masks',635,'图15-4：因果掩码与有效键掩码的交集','两个四乘四掩码矩阵分别表示长度四和填充后真实长度二的序列。允许位置在softmax前保留，禁止位置权重为零。无效查询行用橙框注明仍需要输出或损失的独立处理。',body)

    counter=report['attention_explanation_counterexample']; body=text(30,42,'不同注意力权重，可以得到完全相同的输出',28,'font-weight="700"')+text(30,83,'一个最小反例：两个值向量相同，V₁=V₂=(2,3)。',23)
    for y,title,weights in [(140,'方案A',counter['first_weights'][0]),(315,'方案B',counter['second_weights'][0])]:
        body+=text(30,y+33,title,25,'font-weight="700"')
        x=165
        for value,color in zip(weights,[BLUE,ORANGE]):
            width=420*value; body+=rect(x,y,width,65,color)+text(x+width/2,y+43,f'{value:.1f}',25,'text-anchor="middle" fill="#fff"'); x+=width
        body+=text(630,y+42,'O=(2,3)',27,'font-weight="700"')
        body+=text(165,y+106,f'{weights[0]:.1f}V₁+{weights[1]:.1f}V₂=(2,3)',24)
    body+=text(30,493,'热图反映此层的混合系数；输出还依赖V、后续层与残差等路径。',22)
    body+=text(30,538,'解释需要明确问题和干预；不能只凭颜色深浅断言输入的因果重要性。',21)
    save(folder,'attention-counterexample',575,'图15-5：不同权重相同输出的真实数值反例','两个相同值向量分别用零点九零点一和零点一零点九加权，输出完全相同。用可手算且已程序验证的例子说明注意力热图不能自动等同于完整因果解释。',body)

if __name__=='__main__':
    write_figures(json.loads(Path('build/experiments/15-attention/results.json').read_text(encoding='utf-8')),Path(__file__).resolve().parents[2]/'docs/assets/15-attention')
