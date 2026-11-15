"""Original vector-only teaching diagrams; numerical plots consume the real report."""
from pathlib import Path
import json,math
from xml.sax.saxutils import escape
ROOT=Path(__file__).resolve().parents[2]
BLUE,ORANGE,GREEN,INK,GRID='#246cc0','#ba5a18','#28815c','#193047','#cedae5'
def txt(x,y,s,size=21,extra=''):
    return f'<text x="{x}" y="{y}" font-size="{size}" {extra}>{escape(str(s))}</text>'
def line(x,y,x2,y2,c=GRID,w=1.5,extra=''):
    return f'<line x1="{x}" y1="{y}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="{w}" {extra}/>'
def rect(x,y,w,h,fill='white',extra=''):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" {extra}/>'
def path(points,c=BLUE,w=3,extra=''):
    return '<path d="'+' '.join(f'{"M" if i==0 else "L"}{x:.3f},{y:.3f}' for i,(x,y) in enumerate(points))+f'" fill="none" stroke="{c}" stroke-width="{w}" {extra}/>'
def dot(x,y,c=BLUE,r=4):return f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{r}" fill="{c}"/>'
def save(folder,name,height,title,description,body):
    ident='c17-'+name
    head=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 {height}" role="img" aria-labelledby="{ident}-title {ident}-desc"><title id="{ident}-title">{escape(title)}</title><desc id="{ident}-desc">{escape(description)}</desc>'
    defs=f'<defs><marker id="{ident}-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{BLUE}"/></marker></defs>'
    (folder/(name+'.svg')).write_text(head+defs+rect(0,0,1000,height,'#f6f9fc','rx="14"')+f'<g font-family="Microsoft YaHei,Noto Sans CJK SC,Arial,sans-serif" fill="{INK}">'+body+'</g></svg>\n',encoding='utf-8')
def axes(x,y,w,h,xmin,xmax,ymin,ymax,xlabel,ylabel,log=False):
    def point(a,b):
        yy=math.log10(max(b,1e-12)) if log else b
        return x+(a-xmin)/(xmax-xmin)*w,y+h-(yy-ymin)/(ymax-ymin)*h
    b=line(x,y,x,y+h,INK,2)+line(x,y+h,x+w,y+h,INK,2)
    for frac in [0,.5,1]:
        val=xmin+(xmax-xmin)*frac;px,py=point(val,10**ymin if log else ymin)
        b+=txt(px,y+h+26,f'{val:g}',16,'text-anchor="middle"')
    for frac in [0,.25,.5,.75,1]:
        v=ymin+(ymax-ymin)*frac;py=y+h-frac*h
        label=f'10^{v:g}' if log else f'{v:g}'
        b+=line(x,py,x+w,py)+txt(x-10,py+6,label,16,'text-anchor="end"')
    b+=txt(x+w/2,y+h+60,xlabel,20,'text-anchor="middle"')+txt(x,y-20,ylabel,19)
    return b,point

def write_figures(report,folder):
    folder.mkdir(parents=True,exist_ok=True)
    b=txt(30,44,'图17-1  编码、瓶颈和重建：目标来自输入自身',29,'font-weight="700"')
    boxes=[(30,175,150,'输入 x','d维'),(235,175,180,'编码器 fθ','d → r'),(465,155,100,'代码 z','r维'),(615,175,170,'解码器 gψ','r → d'),(835,175,135,'重建 x̂','d维')]
    for x,y,w,title,sub in boxes:
        b+=rect(x,y,w,100,'white','rx="10" stroke="#88a7c3" stroke-width="2"')+txt(x+w/2,y+38,title,24,'text-anchor="middle"')+txt(x+w/2,y+77,sub,21,'text-anchor="middle"')
    marker='marker-end="url(#c17-01-architecture-arrow)"'
    for a,c in [(180,230),(415,460),(565,610),(785,830)]:b+=line(a,220,c,220,BLUE,3,marker)
    b+=path([(105,275),(105,330),(900,330),(900,278)],BLUE,2,'stroke-dasharray="7 5"')+txt(500,360,'比较输入与重建：按全部样本、全部坐标平均平方误差',22,'text-anchor="middle"')
    b+=txt(30,415,'r < d 限制通过的坐标数；可压缩结构，也可能丢掉有用信息。',22)
    b+=txt(30,457,'代码坐标没有天然语义：可逆地改坐标并配套改解码器，重建仍可相同。',22)
    save(folder,'01-architecture',490,'自编码器数据流与重建目标','原创结构图，区分输入、代码、编码器、解码器和重建；没有外部标签。',b)
    b=txt(30,44,'图17-2  PCA手算：投影到一条直线，丢掉垂直分量',28,'font-weight="700"')
    b+=txt(30,82,'四点 (±2, ±0.5)；训练均值为0；保留方差较大的横轴。',22)
    a,p=axes(125,165,710,260,-2.5,2.5,-1,1,'第一坐标 x₁','第二坐标 x₂');b+=a
    x0,y0=p(-2.5,0);x1,y1=p(2.5,0);b+=line(x0,y0,x1,y1,BLUE,3)
    for x in [-2,2]:
        for y in [-.5,.5]:
            px,py=p(x,y);rx,ry=p(x,0)
            b+=line(px,py,rx,ry,ORANGE,2,'stroke-dasharray="5 4"')+dot(px,py,ORANGE,6)
        rx,ry=p(x,0);b+=rect(rx-6,ry-6,12,12,BLUE)
    b+=txt(30,535,'橙圆：原输入；蓝方：重建。四个点各丢掉 ±0.5 的纵向分量。',22)
    b+=txt(30,575,'每样本平方范数误差 = 0.25；每坐标MSE = 0.25 / 2 = 0.125。',22)
    b+=txt(30,615,'C = diag(4, 0.25)；保留横向方差4，丢弃方差0.25。',22)
    save(folder,'02-pca-projection',645,'PCA投影与精确误差手算','四个手算点及正交投影，实线为保留子空间，虚线为残差；不是实验散点。',b)
    b=txt(30,44,'图17-3  实际训练：验证选模；测试只在全部运行冻结后评价',27,'font-weight="700"')
    a,p=axes(115,140,750,295,0,report['protocol']['steps'],-5,0,'参数更新步','每坐标MSE（log₁₀）',True);b+=a
    r=next(r for r in report['runs'] if r['latent']==1 and r['seed']==11)
    b+=path([p(q['step'],q['train_mse']) for q in r['curves']],BLUE)
    b+=path([p(q['step'],q['validation_mse']) for q in r['curves']],ORANGE,3,'stroke-dasharray="8 5"')
    b+=line(125,535,185,535,BLUE,3)+txt(200,542,'训练：实线',21)+line(520,535,580,535,ORANGE,3,'stroke-dasharray="8 5"')+txt(595,542,'验证：虚线',21)
    b+=txt(30,590,'展示 r=1、种子11的一次完整轨迹；最佳权重来自验证记录点。',22)
    b+=txt(30,632,'三维度×三种子的完整指标见正文；不是从测试集中挑出的最好曲线。',22)
    save(folder,'03-learning-curves',660,'自编码器实际训练和验证曲线','来源是本机CPU float64日志；纵轴log10，实线与虚线区分数据划分。',b)
    b=txt(30,44,'图17-4  二维代码与重建：看见结构，不等于获得语义或生成分布',26,'font-weight="700"')
    r=next(r for r in report['runs'] if r['latent']==2 and r['seed']==11)
    z=r['test_latent'];lo=[min(row[j] for row in z) for j in [0,1]];hi=[max(row[j] for row in z) for j in [0,1]]
    pad=[max((hi[j]-lo[j])*.1,.1) for j in [0,1]]
    a,p=axes(100,150,325,285,lo[0]-pad[0],hi[0]+pad[0],lo[1]-pad[1],hi[1]+pad[1],'代码 z₁','代码 z₂');b+=a
    a,q=axes(625,150,280,285,-1.25,1.25,-1.25,1.25,'输入/重建 x₃','输入/重建 x₁');b+=a
    order=sorted(range(len(z)),key=lambda i:report['data']['test_t'][i])
    b+=path([p(*z[i]) for i in order],GRID,1.4)
    for i in range(len(z)):
        t=report['data']['test_t'][i];c=BLUE if t<0 else ORANGE
        px,py=p(*z[i]);b+=dot(px,py,c,3.6)
        x=report['data']['test'][i];rec=r['test_reconstruction'][i]
        ox,oy=q(x[2],x[0]);rx,ry=q(rec[2],rec[0]);b+=dot(ox,oy,INK,3)+line(ox,oy,rx,ry,GREEN,1)
        b+=rect(rx-2,ry-2,4,4,GREEN)
    b+=txt(30,535,'左：64条测试输入的二维代码；按真实t顺序用细线辅助观察。',22)
    b+=txt(30,576,'蓝/橙分别标记 t<0 / t≥0；t未参与训练或选模，不作为标签监督。',22)
    b+=txt(30,617,'右：原始三维数据在 (x₃,x₁) 平面的投影；黑圆输入、绿方重建。',22)
    b+=txt(30,658,'这是已见分布的新样本重建；没有证明任意采样代码都能产生合理数据。',22)
    save(folder,'04-latent-and-reconstruction',690,'二维潜在表示与三维输入投影上的重建','两面板均来自种子11、r2模型的最终测试；三维观测只显示二维投影，不声称完整三维图。',b)

    b=txt(30,44,'图17-5  三种输入与目标关系：普通重建、去噪、采样',28,'font-weight="700"')
    rows=[(140,'普通重建','输入 x → 编码 → 解码 → 重建','目标：同一个 x'),
          (280,'去噪任务','输入 x̃ → 编码 → 解码 → 重建','目标：扰动前 x'),
          (420,'生成使用','采样 z ~ p(z) → 解码 → 新输出','需定义代码采样分布')]
    for y,title,flow,sub in rows:
        b+=rect(30,y-42,940,105,'white','rx="12" stroke="#88a7c3" stroke-width="2"')
        b+=txt(50,y,title,24,'font-weight="700"')+txt(235,y,flow,24)+txt(235,y+38,sub,21)
    b+=txt(30,565,'重建目标没有自动规定 p(z)；任意远处的代码可能落在训练未约束区域。',22)
    b+=txt(30,610,'本章CPU实验属于第一行；第二、三行是关系说明，不冒充已训练结果。',22)
    save(folder,'05-reconstruction-denoising-generation',645,'普通重建、去噪和采样所需的信息','原创关系图，明确输入与目标的区别，以及生成使用需要额外代码分布。',b)

if __name__=='__main__':
    report=json.loads((ROOT/'build/experiments/17-representation-learning/results.json').read_text(encoding='utf-8'))
    write_figures(report,ROOT/'docs/assets/17-representation-learning')
