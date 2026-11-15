"""Vector diagrams and actual training curves for generalization experiments."""
import html
import math
from pathlib import Path

INK, BLUE, TEAL, ORANGE, GRID = "#16344f", "#235bc7", "#007d75", "#a95813", "#cbdbe7"

def txt(x,y,s,size=24,color=INK,anchor="start"):
    return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{color}" text-anchor="{anchor}">{html.escape(str(s))}</text>'

def line(x1,y1,x2,y2,color=GRID,width=2,dash=""):
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"' + (f' stroke-dasharray="{dash}"' if dash else '') + '/>'

def arr(x1,y1,x2,y2,color=INK):
    a=math.atan2(y2-y1,x2-x1)
    return line(x1,y1,x2,y2,color,3)+f'<path d="M {x2-11*math.cos(a-.5):.2f} {y2-11*math.sin(a-.5):.2f} L {x2:.2f} {y2:.2f} L {x2-11*math.cos(a+.5):.2f} {y2-11*math.sin(a+.5):.2f}" fill="none" stroke="{color}" stroke-width="3"/>'

def rect(x,y,w,h):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#eef5fa" stroke="{GRID}" stroke-width="2"/>'

def path(points,color,width=3,dash=""):
    d='M '+' L '.join(f'{x:.2f} {y:.2f}' for x,y in points)
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>'

def save(folder,name,title,desc,w,h,body):
    (folder/name).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title desc"><title id="title">{html.escape(title)}</title><desc id="desc">{html.escape(desc)}</desc><rect width="{w}" height="{h}" fill="white"/><g font-family="Microsoft YaHei, Segoe UI, sans-serif">{body}</g></svg>\n',encoding='utf-8')

def write_figures(report,folder:Path):
    folder.mkdir(parents=True,exist_ok=True)
    colors={'baseline':BLUE,'weight_decay':TEAL,'dropout':ORANGE}
    labels={'baseline':'基线','weight_decay':'权重衰减','dropout':'Dropout'}
    selected=[r for r in report['runs'] if r['seed']==11]
    # All plotted predictions are restored validation-selected models, not final-epoch models.
    body=txt(35,45,'图10-1  小样本回归：同一训练集、种子11、验证集选出的模型',27)
    left,top,w,h=100,100,690,380
    all_y=report['data']['train_y']+[v for r in selected for v in r['best_prediction']]+[-1,1]
    ymin=min(all_y)-.2;ymax=max(all_y)+.2
    sx=lambda x:left+(x+1)/2*w
    sy=lambda y:top+(ymax-y)/(ymax-ymin)*h
    for y in [-1,0,1]:body+=line(left,sy(y),left+w,sy(y))+txt(left-12,sy(y)+7,y,21,INK,'end')
    for x in [-1,-.5,0,.5,1]:body+=line(sx(x),top,sx(x),top+h)+txt(sx(x),top+h+30,x,21,INK,'middle')
    body+=line(left,top+h,left+w,top+h,INK)+line(left,top,left,top+h,INK)
    grid=selected[0]['grid']
    body+=path([(sx(x),sy(math.sin(3*x))) for x in grid],INK,3,'9 6')
    a,b=report['linear_reference']['coefficients']
    body+=path([(sx(x),sy(a*x+b)) for x in grid],'#657b8c',3,'3 4')
    for j,r in enumerate(selected):
        body+=path([(sx(x),sy(y)) for x,y in zip(grid,r['best_prediction'])],colors[r['method']],3,['','10 5','3 5'][j])
    for x,y in zip(report['data']['train_x'],report['data']['train_y']):body+=f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="5" fill="white" stroke="{INK}" stroke-width="2"/>'
    for j,label in enumerate(['真实无噪声函数','线性参照','基线','权重衰减','Dropout']):
        c=[INK,'#657b8c',BLUE,TEAL,ORANGE][j];dash=['9 6','3 4','','10 5','3 5'][j]
        body+=line(825,150+j*65,870,150+j*65,c,3,dash)+txt(885,158+j*65,label,22,c)
    body+=txt(560,555,'输入x（无量纲）',24)+txt(55,90,'目标/预测y',23)
    body+=txt(50,610,'空心点为20个含噪训练样本；不展示测试标签来调整配置。',23)
    save(folder,'10-01-data-capacity.svg','小样本回归与模型输出','实际固定数据与种子11的三种验证集选优模型，含真实函数和线性参照，测试数据不参与拟合或选择。',1160,650,body)

    body=txt(35,45,'图10-2  实际学习曲线：同一评估模式下比较纯MSE',28)
    for j,r in enumerate(selected):
        left=65+j*365;top=125;w=280;h=310
        max_epoch=max(1,r['stop_epoch']);vals=r['train_eval_mse']+r['validation_mse']
        lo=math.floor(math.log10(min(vals)));hi=math.ceil(math.log10(max(vals)))
        if hi==lo:hi+=1
        sx=lambda k:left+k/max_epoch*w
        sy=lambda v:top+(hi-math.log10(v))/(hi-lo)*h
        body+=txt(left+w/2,92,labels[r['method']],26,colors[r['method']],'middle')
        for power in range(lo,hi+1):
            body+=line(left,sy(10.**power),left+w,sy(10.**power))+txt(left-6,sy(10.**power)+6,'1e'+str(power),17,INK,'end')
        body+=line(left,top,left,top+h,INK)+line(left,top+h,left+w,top+h,INK)
        body+=path([(sx(k),sy(v)) for k,v in enumerate(r['train_eval_mse'])],BLUE,3)
        body+=path([(sx(k),sy(v)) for k,v in enumerate(r['validation_mse'])],ORANGE,3,'8 5')
        body+=line(sx(r['best_epoch']),top,sx(r['best_epoch']),top+h,TEAL,2,'3 4')
        body+=txt(left,top+h+32,0,20)+txt(left+w,top+h+32,max_epoch,20,INK,'end')
        body+=txt(left,top+h+75,'最佳轮：'+str(r['best_epoch']),22,TEAL)
    body+=line(90,555,140,555,BLUE,3)+txt(155,563,'训练MSE（实线）',23,BLUE)
    body+=line(490,555,540,555,ORANGE,3,'8 5')+txt(555,563,'验证MSE（虚线）',23,ORANGE)
    body+=txt(50,620,'各面板横轴到实际停止轮；纵轴为对数MSE，刻度分别标注。',23)
    save(folder,'10-02-learning-curves.svg','三种方法的实际训练验证曲线','同一初始化种子11，训练和验证指标都在eval模式计算，绿色竖线标验证集最佳轮；横轴到各自早停轮。',1160,660,body)

    body=txt(35,45,'图10-3  Inverted Dropout：保留项在训练时缩放',29)
    vectors=[('输入 h',[2,4,6,8]),('一次掩码 m',[1,0,1,0]),('训练输出 h′',[4,0,12,0]),('评估：直接取h',[2,4,6,8])]
    for j,(label,values) in enumerate(vectors):
        x=50+j*285;body+=txt(x+105,125,label,25,INK,'middle')
        for i,v in enumerate(values):
            body+=rect(x+55,165+i*70,100,55)+txt(x+105,203+i*70,v,27,INK,'middle')
        if j<2:body+=arr(x+165,295,x+255,295)
    body+=txt(55,530,'丢弃概率p=0.5；h′=m·h/(1−p)。单次输出不必等于原值。',25)
    body+=txt(55,590,'精确枚举16种等概率掩码可得 E[h′]=h；eval模式直接保留h。',24)
    save(folder,'10-03-dropout.svg','Dropout训练与评估','固定示意掩码1,0,1,0在丢弃概率一半时把2,4,6,8映射到4,0,12,0；期望保持，评估为恒等映射。',1200,640,body)

    body=txt(35,45,'图10-4  数据职责与初始化：在比较前固定协议',29)
    for x,label,sub in [(50,'训练集','仅它提供参数梯度'),(455,'验证集','选超参数与最佳轮'),(860,'测试集','配置冻结后最终评价')]:
        body+=rect(x,120,290,145)+txt(x+145,172,label,29,INK,'middle')+txt(x+145,225,sub,23,INK,'middle')
    body+=arr(345,195,440,195)+arr(750,195,845,195)
    body+=txt(60,325,'比较三种方法：同一数据划分、同一初始化、同一预算与早停规则。',25)
    body+=rect(60,380,1070,160)+txt(85,425,'ReLU隐藏层的尺度直觉（满足独立、对称等近似条件）：',25)
    body+=txt(85,472,'输入二阶矩 → d_in个加权项相加 → ReLU约保留一半二阶矩',24)
    body+=txt(85,515,'选择 Var(W)≈2/d_in，有助于控制逐层尺度；不是任意数据的精确保证。',23)
    save(folder,'10-04-protocol-initialization.svg','数据协议与初始化尺度','训练数据更新权重，验证数据选择方案，测试数据在冻结后评价；Kaiming初始化以输入连接数和ReLU二阶矩近似控制尺度。',1200,600,body)
