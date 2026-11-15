"""Editable pure SVG figures for chapter 12; numerical plots read actual JSON logs."""
from pathlib import Path
import json
import math
from xml.sax.saxutils import escape

BLUE,ORANGE,GREEN,PURPLE='#246cc0','#bd611a','#39805b','#8954a3'
INK,GRID='#172c44','#d0dbe7'
def text(x,y,value,size=22,extra=''): return f'<text x="{x}" y="{y}" font-size="{size}" {extra}>{escape(str(value))}</text>'
def line(x1,y1,x2,y2,color=GRID,width=1.5,extra=''): return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" {extra}/>'
def rect(x,y,w,h,fill='#fff',extra=''): return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" {extra}/>'
def path(points,color,width=3,extra=''):
    data=' '.join(f'{"M" if i==0 else "L"} {x:.3f} {y:.3f}' for i,(x,y) in enumerate(points))
    return f'<path d="{data}" fill="none" stroke="{color}" stroke-width="{width}" {extra}/>'
def save(folder,name,height,title,desc,body):
    definition=f'<defs><marker id="c12-{name}-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{BLUE}"/></marker></defs>'
    source=f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" viewBox="0 0 900 {height}" role="img" aria-labelledby="c12-{name}-title c12-{name}-desc"><title id="c12-{name}-title">{escape(title)}</title><desc id="c12-{name}-desc">{escape(desc)}</desc>'+definition+rect(0,0,900,height,'#f7f9fc','rx="16"')+'<g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" fill="#172c44">'+body+'</g></svg>\n'
    (folder/f'{name}.svg').write_text(source,encoding='utf-8')

def write_figures(report,folder):
    folder.mkdir(parents=True,exist_ok=True)
    body=text(30,42,'残差块：相加需要形状相同',28,'font-weight="700"')+text(30,82,'学习分支 F(x)；捷径把输入直接送到加法处。',22)
    for x,w,label,sub in [(35,140,'输入 x','形状 d'),(300,225,'残差分支 F','输出形状 d'),(705,160,'输出 y','x + F(x)')]:
        body+=rect(x,195,w,100,'#fff','rx="12" stroke="#8a9eb4" stroke-width="2"')+text(x+w/2,233,label,25,'text-anchor="middle" font-weight="700"')+text(x+w/2,274,sub,22,'text-anchor="middle"')
    marker='marker-end="url(#c12-residual-block-arrow)"'
    body+=line(175,245,295,245,BLUE,3,marker)+line(525,245,614,245,BLUE,3,marker)+line(665,245,700,245,BLUE,3,marker)
    body+='<circle cx="640" cy="245" r="25" fill="#e4effc" stroke="#246cc0" stroke-width="2"/>'+text(640,255,'+',33,'text-anchor="middle"')
    body+=path([(210,245),(210,135),(640,135),(640,216)],BLUE,3,marker)+text(415,123,'恒等捷径 I；尺寸变化时用投影 P',22,'text-anchor="middle"')
    body+=text(30,352,'局部雅可比：I + J_F；上游梯度可以沿两条路径传递。',23)+text(30,395,'若 F(x)=−x，两条导数恰好抵消；残差不保证梯度永不消失。',22)
    save(folder,'residual-block',430,'图12-1：残差分支、恒等捷径与形状要求','输入分到残差分支和恒等捷径，在加法处按相同位置相加。形状不同时必须投影或调整，局部导数为恒等矩阵加分支雅可比，不存在无条件的梯度下界。',body)

    body=text(30,42,'归一化轴：先问哪些元素共同计算均值与方差',27,'font-weight="700"')+text(30,82,'示例输入 X 为 B×C；每行一个样本，每列一个特征。',22)
    for left,title,kind in [(95,'BatchNorm1d：按列', 'bn'),(525,'LayerNorm(C)：按行','ln')]:
        body+=text(left-25,139,title,25,'font-weight="700"')
        for row in range(2):
            for col in range(3):
                fills=['#dceafe','#ffead6','#e1f0e7']; fill=fills[col] if kind=='bn' else fills[row]
                body+=rect(left+col*80,180+row*75,80,75,fill,'stroke="#71869d"')+text(left+col*80+40,229+row*75,row*3+col+1,27,'text-anchor="middle"')
        body+=text(left+120,365,'同色元素共享统计量',22,'text-anchor="middle"')
        body+=text(left+120,405,'跨样本，逐特征' if kind=='bn' else '单个样本，跨特征',22,'text-anchor="middle"')
    body+=text(30,460,'NCHW 图像：BatchNorm2d 固定 C，统计 B、H、W 三个轴。',23)
    body+=text(30,503,'LayerNorm 的轴由 normalized_shape 指定；不会自动识别“通道”。',22)
    body+=text(30,546,'BN 的 γ、β 通常每通道一对；LN 的可学习参数跟随指定的末尾形状。',21)
    save(folder,'normalization-axes',580,'图12-2：BatchNorm与LayerNorm的统计轴','二维批次矩阵的列对应特征、行对应样本。BatchNorm按列跨样本归一化，LayerNorm在每个样本内跨指定末尾特征归一化。另注明NCHW图像的BatchNorm统计轴和LayerNorm接口要求。',body)

    demo=report['normalization_and_residual_demonstrations']
    body=text(30,42,'相同输入，BatchNorm训练与评估可以不同',27,'font-weight="700"')+text(30,82,'X=[[1,2],[3,6]]，γ=1、β=0、ε=10⁻⁵；momentum=1。',22)
    columns=[(30,'量'),(340,'通道1'),(615,'通道2')]
    for x,label in columns: body+=text(x,135,label,24,'font-weight="700"')
    rows=[('本批均值',demo['mean']),('前向方差：除以2',demo['biased_variance']),('存储方差：除以1',demo['running_variance']),('首样本 train 输出',demo['train_output'][0]),('首样本 eval 输出',demo['eval_output'][0])]
    for i,(label,values) in enumerate(rows):
        y=185+i*62; body+=line(30,y+19,865,y+19)+text(30,y,label,23)
        for xx,value in zip([340,615],values): body+=text(xx,y,f'{value:.6f}',23)
    body+=text(30,525,'这是一批数据后的刻意演示；实际默认momentum=0.1。',22)
    body+=text(30,567,'冻结γ、β不会冻结running_mean/var；它们是缓冲区。',22)
    save(folder,'batchnorm-modes',605,'图12-3：真实BatchNorm手算与模式输出','本批平均和用于前向的有偏方差可手算。PyTorch存储的移动方差使用无偏估计，本例momentum设为一，训练归一化结果约正负一，评估约正负零点七零七。',body)

    body=text(30,42,'迁移学习：先定义来源，再明确更新哪些部分',27,'font-weight="700"')+text(30,82,'本例自行预训练；没有下载ImageNet模型，也不声称获得通用视觉表征。',21)
    marker='marker-end="url(#c12-transfer-workflow-arrow)"'
    for xx,ww,title,sub in [(30,235,'源任务：四象限','256点，4类'),(335,235,'预训练骨干','源模型492参数'),(640,230,'替换目标头','目标2类，32点')]:
        body+=rect(xx,125,ww,95,'#fff','rx="12" stroke="#8a9eb4" stroke-width="2"')+text(xx+ww/2,164,title,24,'text-anchor="middle" font-weight="700"')+text(xx+ww/2,200,sub,22,'text-anchor="middle"')
    body+=line(265,171,330,171,BLUE,3,marker)+line(570,171,635,171,BLUE,3,marker)
    items=[(30,'从零训练','随机骨干 + 相同新头','474参数可更新'),(335,'冻结特征','预训练骨干固定','仅18个头参数更新'),(640,'微调','预训练骨干继续更新','474参数可更新')]
    for xx,title,sub,end in items:
        body+=rect(xx,300,230,134,'#fff','rx="12" stroke="#8a9eb4" stroke-width="2"')+text(xx+115,339,title,25,'text-anchor="middle" font-weight="700"')+text(xx+115,379,sub,21,'text-anchor="middle"')+text(xx+115,416,end,21,'text-anchor="middle"')
    body+=text(30,484,'比较相同目标数据与100步预算；迁移另有200步预训练成本。',22)
    body+=text(30,528,'实测冻结和微调都未超过从零基线；失败结果同样用于学习。',22)
    save(folder,'transfer-workflow',565,'图12-4：源任务预训练与三种目标训练策略','先对四象限源任务预训练同形状骨干，替换成二分类头，对照随机初始化、冻结预训练特征、全部微调。三种方法使用相同目标训练集和步数，另外注明迁移的源任务成本及真实失败结果。',body)

    body=text(30,42,'普通与残差网络：三种子真实训练曲线',28,'font-weight="700"')+text(30,82,'相同492参数与逐种子初值；只切换三个块是否执行 x+F(x)。',22)
    left,top,width,height=105,135,690,305
    sx=lambda step:left+step/200*width
    sy=lambda loss:top+height-(math.log10(loss)+4)/4.5*height
    for tick in [1,.1,.01,.001,.0001]:
        body+=line(left,sy(tick),left+width,sy(tick))+text(left-12,sy(tick)+7,f'{tick:g}',21,'text-anchor="end"')
    for tick in [0,100,200]:body+=text(sx(tick),top+height+32,tick,22,'text-anchor="middle"')
    for residual,color,dash in [(False,BLUE,''),(True,ORANGE,'stroke-dasharray="9 4"')]:
        histories=[r['history'] for r in report['source_comparison'] if r['residual']==residual]
        for hist in histories: body+=path([(sx(r['step']),sy(r['train']['loss'])) for r in hist],color,1.5,'opacity="0.28" '+dash)
        average=[sum(h[i]['train']['loss'] for h in histories)/len(histories) for i in range(201)]
        body+=path([(sx(i),sy(v)) for i,v in enumerate(average)],color,3,dash)
    body+=text(30,120,'交叉熵（对数轴）',21)+text(805,505,'更新次数',22,'text-anchor="end"')
    body+=text(30,547,'蓝实线：普通；橙虚线：残差。粗线算术平均，细线各个种子。',22)
    body+=text(30,590,'六组最终训练准确率均100%；验证效果不保证残差逐种子更好。',22)
    save(folder,'residual-training-curves',625,'图12-5：三个种子的普通与残差网络真实训练曲线','交叉熵对数坐标展示相同初始化下普通网络与残差网络的真实训练轨迹。每种方法三条细线和一条算术平均粗线。最后都可拟合训练集，残差的优化路径与泛化并无无条件优势。',body)

    body=text(30,42,'目标任务：迁移没有保证优于从零训练',28,'font-weight="700"')+text(30,82,'32个训练点、256个验证点；三个模型种子，不按验证集选最佳步。',22)
    left,top,width,height=100,135,695,295
    sx=lambda step:left+step/100*width
    all_values=[r['validation']['loss'] for result in report['transfer'] for r in result['history']]
    ymax=math.ceil(max(all_values)*2)/2
    sy=lambda loss:top+height-(math.log10(loss)+2)/3*height
    for val in [10,1,.1,.01]:
        body+=line(left,sy(val),left+width,sy(val))+text(left-12,sy(val)+7,f'{val:g}',21,'text-anchor="end"')
    for tick in [0,50,100]:body+=text(sx(tick),top+height+32,tick,22,'text-anchor="middle"')
    for mode,color,dash in [('scratch',BLUE,''),('frozen',GREEN,'stroke-dasharray="9 4"'),('finetune',ORANGE,'stroke-dasharray="3 3"')]:
        histories=[r['history'] for r in report['transfer'] if r['mode']==mode]
        for hist in histories: body+=path([(sx(r['step']),sy(r['validation']['loss'])) for r in hist],color,1.5,'opacity="0.2" '+dash)
        average=[sum(h[i]['validation']['loss'] for h in histories)/3 for i in range(101)]
        body+=path([(sx(i),sy(v)) for i,v in enumerate(average)],color,3,dash)
    body+=text(30,120,'验证交叉熵（对数轴）',21)+text(810,495,'目标更新次数',22,'text-anchor="end"')
    for xx,label,color in [(30,'从零：蓝实线',BLUE),(320,'冻结：绿虚线',GREEN),(610,'微调：橙点线',ORANGE)]:body+=text(xx,540,label,22,f'fill="{color}"')
    body+=text(30,583,'粗线为三种子均值，细线为各次运行；不把训练损失更低等同泛化更好。',21)
    save(folder,'transfer-validation-curves',620,'图12-6：三种目标训练策略的真实验证曲线','真实目标验证损失显示从零、冻结特征和微调的差异。三条细线表示各个种子，粗线表示平均，保留迁移未超过基线的证据。训练预算固定一百步，不按验证集选择最优中间状态。',body)

if __name__=='__main__':
    write_figures(json.loads(Path('build/experiments/12-residual-and-normalization/results.json').read_text(encoding='utf-8')),Path(__file__).resolve().parents[2]/'docs/assets/12-residual-and-normalization')
