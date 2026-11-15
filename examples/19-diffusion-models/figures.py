"""Original vector DDPM diagrams and plots from actual CPU experiment records."""
import html
import math
from pathlib import Path
INK,BLUE,TEAL,ORANGE,GRID='#16344f','#235bc7','#007d75','#a95813','#cbdbe7'
def txt(x,y,s,size=24,color=INK,anchor='start'):
 return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{color}" text-anchor="{anchor}">{html.escape(str(s))}</text>'
def line(x1,y1,x2,y2,color=GRID,width=2,dash=''):
 return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>'
def arrow(x1,y1,x2,y2,color=INK):
 a=math.atan2(y2-y1,x2-x1)
 return line(x1,y1,x2,y2,color,3)+f'<path d="M {x2-10*math.cos(a-.5):.2f} {y2-10*math.sin(a-.5):.2f} L {x2:.2f} {y2:.2f} L {x2-10*math.cos(a+.5):.2f} {y2-10*math.sin(a+.5):.2f}" stroke="{color}" fill="none" stroke-width="3"/>'
def box(x,y,w,h,label,sub=''):
 s=f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#eef5fa" stroke="{GRID}" stroke-width="2"/>'+txt(x+w/2,y+h/2+(0 if sub else 8),label,25,INK,'middle')
 return s+(txt(x+w/2,y+h/2+30,sub,20,INK,'middle') if sub else '')
def path(points,color,width=3,dash=''):
 d='M '+' L '.join(f'{x:.2f} {y:.2f}' for x,y in points)
 return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>'
def save(folder,name,title,desc,w,h,body):
 (folder/name).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title desc"><title id="title">{html.escape(title)}</title><desc id="desc">{html.escape(desc)}</desc><rect width="{w}" height="{h}" fill="white"/><g font-family="Microsoft YaHei, Segoe UI, sans-serif">{body}</g></svg>\n',encoding='utf-8')
def scatter(left,top,size,points,lim,color=BLUE,ticks=True):
 sx=lambda x:left+(x+lim)/(2*lim)*size;sy=lambda y:top+(lim-y)/(2*lim)*size
 b=f'<rect x="{left}" y="{top}" width="{size}" height="{size}" fill="white" stroke="{GRID}" stroke-width="2"/>'+line(left,sy(0),left+size,sy(0))+line(sx(0),top,sx(0),top+size)
 for x,y in points:b+=f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="2.1" fill="{color}" fill-opacity=".7"/>'
 if ticks:
  for val in [-lim,0,lim]:b+=txt(sx(val),top+size+23,f'{val:g}',17,INK,'middle')+txt(left-7,sy(val)+6,f'{val:g}',17,INK,'end')
 return b

def write_figures(report,folder:Path):
 folder.mkdir(parents=True,exist_ok=True);forward=report['forward_snapshots'];reverse=report['reverse_snapshots'];lookup=lambda d,t:d.get(str(t),d.get(t))
 all_points=[p for d in [forward,reverse] for ps in d.values() for p in ps];lim=max(4,math.ceil(max(abs(c) for p in all_points for c in p)))
 body=txt(35,45,'图19-1  实际二维样本：固定前向加噪链与学习后的反向生成链',29)
 body+=txt(50,100,'前向：独立高斯增量，逐步破坏四个簇',25,BLUE)
 for j,t in enumerate([0,16,32,48,64]):
  left=65+j*250;body+=txt(left+85,135,f't={t}',24,INK,'middle')+scatter(left,155,170,lookup(forward,t),lim,BLUE)
  if j<4:body+=arrow(left+180,235,left+230,235,BLUE)
 body+=txt(50,410,'反向：从新高斯样本开始，重复调用同一个时间条件网络',25,TEAL)
 for j,t in enumerate([64,48,32,16,0]):
  left=65+j*250;body+=txt(left+85,445,f't={t}',24,INK,'middle')+scatter(left,465,170,lookup(reverse,t),lim,TEAL)
  if j<4:body+=arrow(left+180,550,left+230,550,TEAL)
 body+=txt(50,715,f'各面板同一坐标范围[−{lim},{lim}]；横轴x₁、纵轴x₂，均无量纲，每图160点。',23)
 body+=txt(50,765,'上下两行不逐点配对；生成过程没有取回某条训练样本的原始加噪路径。',23)
 save(folder,'19-01-forward-reverse.svg','前向加噪与反向生成','实际前向Markov链使用每步独立高斯噪声，反向链从新标准高斯样本起步并用训练网络迭代64步，两个方向不是逐点逆变换。',1300,810,body)

 body=txt(35,45,'图19-2  时间条件噪声网络：参数共享，噪声等级作为输入',29)
 body+=box(40,135,205,90,'加噪点 xₜ','B×2')+box(40,365,205,90,'时间步 t','B个整数，1…64')
 body+=box(350,365,310,100,'u=t/64','u、sin(πu)、cos(πu)')+arrow(250,410,340,410)
 body+=box(750,240,225,100,'拼接输入','B×5')+arrow(250,180,740,275)+arrow(665,410,740,305)
 body+=box(1040,240,225,100,'MLP → ε̂','5→48→48→2')+arrow(980,290,1030,290)
 body+=txt(50,555,'两个隐藏层用tanh；同一θ处理所有时间步，不是每个t训练一个网络。',24)
 body+=txt(50,615,'输出与xₜ同形为B×2；训练标签是实际加入的ε，采样时该标签未知。',24)
 save(folder,'19-02-time-network.svg','时间条件噪声预测网络','二维加噪点与三维固定时间特征拼接为五维输入，经过两个48宽tanh隐藏层预测二维噪声，2738参数在所有时间步共享。',1320,660,body)

 body=txt(35,45,'图19-3  训练与采样：一次训练更新不需要遍历整条反向链',28)
 body+=txt(50,110,'训练一次更新：有真实x₀与已知噪声标签',25,BLUE)
 for x,w,label,sub in [(40,210,'取训练x₀','抽样t与ε'),(335,250,'闭式构造xₜ','无需先走t次加噪'),(670,245,'网络预测ε̂','输入xₜ与t'),(1000,245,'平均平方误差','反传并更新θ')]:body+=box(x,160,w,105,label,sub)
 for a,b in [(255,325),(590,660),(920,990)]:body+=arrow(a,212,b,212,BLUE)
 body+=txt(50,370,'采样完整链：没有真实x₀，也没有真实ε标签',25,TEAL)
 for x,w,label,sub in [(40,210,'从x₆₄开始','标准高斯'),(335,250,'调用εθ(xₜ,t)','构造反向均值'),(670,245,'采样xₜ₋₁','t>1时加入新噪声'),(1000,245,'返回生成x₀','共64次网络调用')]:body+=box(x,420,w,105,label,sub)
 for a,b in [(255,325),(590,660),(920,990)]:body+=arrow(a,472,b,472,TEAL)
 body+=line(790,530,790,605,TEAL,3)+line(790,605,460,605,TEAL,3)+arrow(460,605,460,535,TEAL)+txt(500,590,'t递减并重复',23,TEAL)
 body+=txt(50,690,'扩散时间t、训练更新次数、采样随机种子是三种不同的控制量。',24)
 save(folder,'19-03-training-sampling.svg','训练与采样流程区别','训练随机选择一个噪声等级直接构造样本并回归噪声，采样则从噪声逐步反向迭代，时间步与优化更新步不能混淆。',1300,740,body)

 body=txt(35,45,'图19-4  实测固定验证噪声MSE：不要求把随机噪声逐点猜到零误差',28)
 left,top,w,h=105,120,795,360;curves=report['curves'];high=max(curves['validation_noise_mse']+[report['zero_validation_noise_mse']])*1.1;low=.2
 sx=lambda k:left+k/report['protocol']['updates']*w;sy=lambda v:top+(high-v)/(high-low)*h
 for val in [.25,.5,.75,1.]:body+=line(left,sy(val),left+w,sy(val))+txt(left-8,sy(val)+7,f'{val:g}',21,INK,'end')
 for val in [0,500,1000,1500,2000,2500]:
  if val<=report['protocol']['updates']:body+=txt(sx(val),top+h+30,val,21,INK,'middle')
 body+=line(left,top,left,top+h,INK)+line(left,top+h,left+w,top+h,INK)+path([(sx(k),sy(v)) for k,v in zip(curves['updates'],curves['validation_noise_mse'])],BLUE,3)
 for value,color,dash,label in [(report['zero_validation_noise_mse'],ORANGE,'8 5','零噪声预测'),(report['oracle_validation_noise_mse'],TEAL,'3 5','已知分布的解析参照')]:body+=line(left,sy(value),left+w,sy(value),color,3,dash)
 for j,(label,color,dash) in enumerate([('学习网络',BLUE,''),('零噪声预测',ORANGE,'8 5'),('解析参照',TEAL,'3 5')]):body+=line(940,175+j*80,985,175+j*80,color,3,dash)+txt(1000,183+j*80,label,22,color)
 body+=line(sx(report['best_update']),top,sx(report['best_update']),top+h,INK,2,'4 5')+txt(650,550,'参数更新次数',23)
 body+=txt(50,620,'t与ε在验证中固定；解析参照使用真实混合分布，不是训练网络可见的额外标签。',23)
 save(folder,'19-04-noise-learning.svg','噪声预测实测学习曲线','固定验证数据和噪声等级的MSE曲线，零预测与知道真实分布的条件均值预测作为参照，选模只根据验证值。',1300,670,body)

 body=txt(35,45,'图19-5  实测分布对照：覆盖四个象限不等于学到了四个簇',29)
 sets=[('独立测试数据',report['test_points'],INK),('直接标准高斯',report['standard_normal_points'],ORANGE),('训练后反向生成',report['generated_points'],TEAL)]
 lim=max(4,math.ceil(max(abs(c) for _,points,_ in sets for p in points for c in p)))
 for j,(title,points,color) in enumerate(sets):
  left=85+j*425;body+=txt(left+150,115,title,25,color,'middle')+scatter(left,150,300,points,lim,color)
  body+=txt(left+150,505,'x₁（无量纲）',22,INK,'middle')
 body+=txt(45,605,'每图160个实际点，坐标范围相同；完整1024点指标见正文，不凭单张散点图判定质量。',23)
 save(folder,'19-05-distribution-comparison.svg','二维真实数据高斯与生成分布','同一坐标尺度展示独立测试点、初始标准高斯和训练网络的反向生成点，点来自真实记录，分布质量另报告完整样本指标。',1360,650,body)
