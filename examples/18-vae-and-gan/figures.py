"""Pure SVG diagrams and actual numerical charts for chapter 18."""
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
    marker=f'<defs><marker id="c18-{name}-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{BLUE}"/></marker></defs>'
    source=f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" viewBox="0 0 900 {height}" role="img" aria-labelledby="c18-{name}-title c18-{name}-desc"><title id="c18-{name}-title">{escape(title)}</title><desc id="c18-{name}-desc">{escape(desc)}</desc>'+marker+rect(0,0,900,height,'#f7f9fc','rx="16"')+'<g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" fill="#172c44">'+body+'</g></svg>\n'
    (folder/f'{name}.svg').write_text(source,encoding='utf-8')

def scatter_panel(left,top,width,height,title,series,circles=False):
    sx=lambda x:left+(x+3.8)/7.6*width
    sy=lambda y:top+height-(y+1.2)/2.4*height
    body=text(left,top-22,title,23,'font-weight="700"')
    for val in [-2,0,2]:body+=line(sx(val),top,sx(val),top+height)+text(sx(val),top+height+26,val,20,'text-anchor="middle"')
    for val in [-1,0,1]:body+=line(left,sy(val),left+width,sy(val))+text(left-10,sy(val)+7,val,20,'text-anchor="end"')
    if circles:
        for center in [-2,2]:body+=f'<ellipse cx="{sx(center)}" cy="{sy(0)}" rx="{.75/7.6*width}" ry="{.75/2.4*height}" fill="none" stroke="#8a9eb4" stroke-width="1.5" stroke-dasharray="5 3"/>'
    for points,color,cross in series:
        for x,y in points:
            if cross:body+=line(sx(x)-3,sy(y)-3,sx(x)+3,sy(y)+3,color,1.6)+line(sx(x)-3,sy(y)+3,sx(x)+3,sy(y)-3,color,1.6)
            else:body+=f'<circle cx="{sx(x):.3f}" cy="{sy(y):.3f}" r="2.7" fill="{color}" opacity="0.6"/>'
    return body


def write_figures(report,folder):
    folder.mkdir(parents=True,exist_ok=True)
    body=text(30,42,'VAE：生成模型与近似推断是不同方向',28,'font-weight="700"')+text(30,82,'生成从先验z开始；训练用观测x构造近似后验q(z|x)。',22)
    marker='marker-end="url(#c18-vae-paths-arrow)"'
    for x,y,w,title,sub in [(30,140,190,'观测 x','二维数据'),(315,140,245,'编码器 qφ(z|x)','输出 μ、logσ²'),(650,140,220,'重参数化','z=μ+σε'),(650,350,220,'解码器 pθ(x|z)','均值 fθ(z)、固定噪声'),(315,350,245,'新样本 x̃','均值 + 观测噪声'),(30,350,190,'先验 p(z)','标准正态')]:
        body+=rect(x,y,w,100,'#fff','rx="12" stroke="#8a9eb4" stroke-width="2"')+text(x+w/2,y+38,title,24,'text-anchor="middle" font-weight="700"')+text(x+w/2,y+77,sub,21,'text-anchor="middle"')
    body+=line(220,190,310,190,BLUE,3,marker)+line(560,190,645,190,BLUE,3,marker)+line(760,240,760,345,BLUE,3,marker)+line(650,400,565,400,BLUE,3,marker)
    body+=path([(125,350),(125,295),(605,295),(605,370),(645,370)],BLUE,3,marker)+text(345,282,'生成时直接从先验采样z',22,'text-anchor="middle"')
    body+=text(30,504,'训练：重建对数密度的期望 − KL(qφ(z|x) || p(z))。',23)
    body+=text(30,548,'ε与参数无关；固定一次ε后，μ、σ到z的路径可求导。',22)
    save(folder,'vae-paths',585,'图18-2：VAE概率模型与可微计算路径','区分从观测到近似后验再重参数化的训练路径，和从标准正态先验到解码条件分布采样的生成路径。解码器输出均值并另加观测噪声，不能把均值曲线当完整采样分布。',body)

    hand=report['hand'];body=text(30,42,'ELBO是下界：先用两个离散潜状态精确核对',28,'font-weight="700"')+text(30,82,'p(z)=(0.5,0.5)，p(x|z)=(0.8,0.2)，q(z|x)=(0.5,0.5)。',22)
    top=145;left=145;width=620;height=205;sx=lambda value:left+(value+1.1)/1.1*width
    for tick in [-1,-.5,0]:body+=line(sx(tick),top,sx(tick),top+height)+text(sx(tick),top+height+32,f'{tick:g}',22,'text-anchor="middle"')
    for y,value,label,color in [(200,hand['elbo'],'ELBO = −0.916291',BLUE),(300,hand['log_evidence'],'ln p(x) = −0.693147',ORANGE)]:
        body+=line(left,y,sx(value),y,color,5)+f'<circle cx="{sx(value)}" cy="{y}" r="7" fill="{color}"/>'+text(sx(value)+15,y+7,label,24)
    body+=text(30,432,'p(x)=0.5，真实后验=(0.8,0.2)。',24)
    body+=text(30,477,'间隙 KL(q || 真实后验)=0.223144，恰好补足两个对数值的差。',22)
    body+=text(30,522,'这个q与先验相同，所以“到先验的KL”为0，但后验近似仍不精确。',21)
    save(folder,'elbo-gap',560,'图18-1：两个潜状态下ELBO与真实证据的精确间隙','用离散潜变量手算证据零点五，ELBO负零点九一六，真实对数证据负零点六九三，两者间隙为到真实后验的KL零点二二三。强调到先验和到真实后验是不同KL。',body)

    body=text(30,42,'GAN交替训练：两步更新不同参数',28,'font-weight="700"')+text(30,82,'D输出真实来源的概率；代码使用未sigmoid的logit与稳定BCE。',22)
    for yy,title,sub in [(125,'① 更新D：真样本标1，假样本标0','G(z).detach()切断到G的求导，更新D参数'),(320,'② 更新G：希望假样本被D判真','冻结D参数；梯度仍穿过D的输入路径回到G')]:
        body+=rect(30,yy,840,145,'#fff','rx="12" stroke="#8a9eb4" stroke-width="2"')+text(55,yy+37,title,25,'font-weight="700"')+text(55,yy+80,sub,22)
        body+=text(55,yy+121,'最小化 L_D=−E lnD(x)−E ln(1−D(G(z)))' if yy==125 else '最小化 L_G=−E lnD(G(z))（非饱和目标）',23)
    body+=text(30,517,'生成器步不对D前向使用no_grad；那会连通向G的导数一起切断。',22)
    body+=text(30,562,'本实验每轮1次D更新+1次G更新；两条损失不是同一固定函数。',22)
    save(folder,'gan-alternation',600,'图18-3：GAN判别器与生成器交替更新及梯度边界','两步分别最小化判别二元交叉熵与生成器非饱和损失。判别器步骤分离假样本，生成器步骤冻结判别器参数但保留输入导数，不能对判别前向关闭求导。',body)

    normal=next(r for r in report['vae'] if r['seed']==11 and r['beta']==1);collapsed=next(r for r in report['vae'] if r['seed']==11 and r['beta']==100)
    body=text(30,42,'VAE真实点图：重建与先验采样分别观察',28,'font-weight="700"')+text(30,82,'固定第一种子11；x₁横轴、x₂纵轴；采样图各256点，坐标范围相同。',21)
    body+=scatter_panel(85,145,315,180,'输入与后验均值重建',[(normal['reconstruction_inputs'],BLUE,False),(normal['posterior_mean_reconstructions'],ORANGE,True)])
    body+=scatter_panel(515,145,315,180,'β=1：完整新样本',[(normal['generated_samples'],BLUE,False)])
    body+=scatter_panel(85,435,315,180,'β=1：解码器均值',[(normal['generated_means'],GREEN,False)])
    body+=scatter_panel(515,435,315,180,'β=100：后验坍塌压力组',[(collapsed['generated_samples'],ORANGE,False)])
    body+=text(30,687,'左上蓝点为输入、橙叉为重建；左下未加观测噪声，不是完整模型样本。',21)
    body+=text(30,730,'压力组改变KL权重，样本挤在两模之间；它不是标准β=1的目标。',22)
    save(folder,'vae-reconstruction-samples',770,'图18-4：VAE重建、解码均值、完整采样与压力组失败','真实CPU种子十一点图。重建使用编码均值，生成从先验抽样，完整样本另加零点二五观测噪声。KL权重一百压力组忽略潜变量，生成点落在真实双模中间，明确目标已改变。',body)

    normal_g=next(r for r in report['gan'] if r['seed']==11 and not r['constant_generator']);constant_g=next(r for r in report['gan'] if r['seed']==11 and r['constant_generator'])
    body=text(30,42,'GAN真实点图：接近模式与覆盖模式需要分别检查',27,'font-weight="700"')+text(30,82,'固定第一种子11；每组256个生成样本；灰圈为模式中心0.75距离范围。',21)
    body+=scatter_panel(85,155,315,250,'普通G：82个参数',[(normal_g['generated_samples'],BLUE,False)],True)
    body+=scatter_panel(515,155,315,250,'常量G：仅2个参数',[(constant_g['generated_samples'],ORANGE,False)],True)
    body+=text(30,476,'普通组：109左、123右、24个不在两圈内；覆盖仍不完美。',23)
    body+=text(30,522,'常量组：256点完全重合在右模附近，虽全部“接近真模式”，多样性为0。',21)
    body+=text(30,567,'这是故意移除噪声作用的容量反例，不能称为普通GAN自发训练坍塌。',21)
    save(folder,'gan-samples-collapse',610,'图18-5：GAN生成样本与受限常量生成器的模式坍塌反例','普通GAN真实样本分布覆盖两个模式但仍有桥接或偏差。退化常量生成器经真实对抗训练后输出全相同，全部点靠近右模，说明只检查接近数据无法衡量覆盖与多样性。',body)

    body=text(30,42,'真实训练日志：分别读VAE目标与GAN动态',28,'font-weight="700"')+text(30,82,'固定种子11；VAE画标准β=1的验证单样本负ELBO，GAN画逐批损失。',21)
    for left,width in [(85,315),(515,315)]:
        for frac in [0,.5,1]:body+=line(left,155+265*frac,left+width,155+265*frac)
    normal_hist=normal['history']; vmax=max(row['negative_elbo_mc'] for row in normal_hist)*1.05
    sx=lambda step:85+step/600*315; sy=lambda value:420-value/vmax*265
    for val in [0,vmax/2,vmax]:body+=text(72,sy(val)+7,f'{val:.1f}',20,'text-anchor="end"')
    body+=path([(sx(row['step']),sy(row['negative_elbo_mc'])) for row in normal_hist],BLUE,3)
    for step in [0,300,600]:body+=text(sx(step),455,step,22,'text-anchor="middle"')
    body+=text(85,130,'验证负ELBO（越低越好）',22)
    gh=normal_g['history'];gmax=max(max(row['generator_loss'],row['discriminator_loss']) for row in gh)*1.05
    sx=lambda step:515+step/1000*315;sy=lambda value:420-value/gmax*265
    for val in [0,gmax/2,gmax]:body+=text(502,sy(val)+7,f'{val:.1f}',20,'text-anchor="end"')
    for key,color,dash in [('discriminator_loss',BLUE,''),('generator_loss',ORANGE,'stroke-dasharray="9 4"')]:body+=path([(sx(row['step']),sy(row[key])) for row in gh],color,3,dash)
    for step in [0,500,1000]:body+=text(sx(step),455,step,22,'text-anchor="middle"')
    body+=text(515,130,'D蓝实线；G橙虚线',22)
    body+=text(30,516,'横轴均为各自更新轮数；两类目标的数值不能横向当作同一质量分数。',21)
    body+=text(30,560,'GAN对手持续变化，损失不必单调；同时看点图、覆盖与模式内方差。',22)
    save(folder,'training-curves',600,'图18-6：实际VAE验证负ELBO与GAN训练损失曲线','固定第一模型种子的真实数值曲线。VAE使用固定验证噪声的一次蒙特卡洛估计，GAN两条曲线来自不同动态目标，不能比较数值大小来给生成质量排名。',body)

if __name__=='__main__':
    write_figures(json.loads(Path('build/experiments/18-vae-and-gan/results.json').read_text(encoding='utf-8')),Path(__file__).resolve().parents[2]/'docs/assets/18-vae-and-gan')
