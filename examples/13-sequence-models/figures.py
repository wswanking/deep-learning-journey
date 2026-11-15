"""Original SVG teaching diagrams plus real measured sequence curves."""
import html
import math
from pathlib import Path
INK,BLUE,TEAL,ORANGE,GRID='#16344f','#235bc7','#007d75','#a95813','#cbdbe7'
def text(x,y,s,size=24,color=INK,anchor='start'):
 return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{color}" text-anchor="{anchor}">{html.escape(str(s))}</text>'
def line(x1,y1,x2,y2,color=GRID,width=2,dash=''):
 return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>'
def arrow(x1,y1,x2,y2,color=INK,dash=''):
 a=math.atan2(y2-y1,x2-x1)
 return line(x1,y1,x2,y2,color,3,dash)+f'<path d="M {x2-11*math.cos(a-.5):.2f} {y2-11*math.sin(a-.5):.2f} L {x2:.2f} {y2:.2f} L {x2-11*math.cos(a+.5):.2f} {y2-11*math.sin(a+.5):.2f}" stroke="{color}" fill="none" stroke-width="3"/>'
def box(x,y,w,h,s,sub='',color=INK):
 out=f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#eef5fa" stroke="{GRID}" stroke-width="2"/>'
 out+=text(x+w/2,y+h/2+(0 if sub else 8),s,26,color,'middle')
 if sub:out+=text(x+w/2,y+h/2+31,sub,21,color,'middle')
 return out
def path(points,color,width=3,dash=''):
 d='M '+' L '.join(f'{x:.2f} {y:.2f}' for x,y in points)
 return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>'
def save(folder,name,title,desc,w,h,body):
 (folder/name).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title desc"><title id="title">{html.escape(title)}</title><desc id="desc">{html.escape(desc)}</desc><rect width="{w}" height="{h}" fill="white"/><g font-family="Microsoft YaHei, Segoe UI, sans-serif">{body}</g></svg>\n',encoding='utf-8')
def write_figures(report,folder:Path):
 folder.mkdir(parents=True,exist_ok=True)
 body=text(35,45,'图13-1  RNN沿时间展开：四次调用，同一组参数',29)
 body+=text(35,278,'h₀',27)
 for j in range(4):
  x=170+j*255
  body+=box(x-65,225,130,80,f'h{j+1}','m维状态')+box(x-65,90,130,65,f'ŷ{j+1}')+box(x-65,400,130,65,f'x{j+1}','')
  body+=arrow(x,225,x,160)+arrow(x,395,x,310)
  body+=arrow(68 if j==0 else x-185,265,x-75,265)
  if j>0:body+=arrow(x-75,345,x-185,345,ORANGE,'7 5')
 body+=text(230,200,'V、bᵧ共享',23)+text(180,390,'Wₓ、b共享',23)
 body+=text(115,520,'实线：前向依赖；橙虚线：损失对较早状态的梯度向后传播。',23)
 body+=text(115,565,'时间增加展开深度，参数数量不随序列长度T成倍增加。',23)
 save(folder,'13-01-unrolled-rnn.svg','RNN时间展开','四个时间步共享输入、循环和输出参数，状态连接向前，反向梯度沿依赖回传。',1200,610,body)

 body=text(35,45,'图13-2  隔离路径的梯度乘积：数值条件决定衰减或放大',28)
 left,top,w,h=100,100,680,380
 sx=lambda t:left+t/40*w;sy=lambda g:top+(8-math.log10(g))/21*h
 for power in [-12,-8,-4,0,4,8]:body+=line(left,sy(10.**power),left+w,sy(10.**power))+text(left-8,sy(10.**power)+7,'1e'+str(power),20,INK,'end')
 for t in [0,10,20,30,40]:body+=line(sx(t),top,sx(t),top+h)+text(sx(t),top+h+30,t,21,INK,'middle')
 for j,(base,c,dash,label) in enumerate([(.5,BLUE,'','RNN w=0.5'),(1.,INK,'8 5','RNN w=1.0'),(1.5,ORANGE,'3 5','RNN w=1.5'),(.95,TEAL,'12 4','LSTM固定f=0.95')]):
  body+=path([(sx(t),sy(base**t)) for t in range(41)],c,3,dash)+line(825,140+j*65,875,140+j*65,c,3,dash)+text(890,148+j*65,label,21,c)
 body+=line(left,top,left,top+h,INK)+line(left,top+h,left+w,top+h,INK)
 body+=text(50,90,'梯度绝对值（对数轴）',23)+text(580,555,'相隔时间步数',23)
 body+=text(50,610,'RNN为零输入、零状态的tanh轨迹；LSTM只画冻结门的直接记忆路径。',23)
 body+=text(50,655,'本图是可核对的机制算例，不是训练模型的总梯度测量。',23)
 save(folder,'13-02-gradient-products.svg','梯度路径乘积','普通RNN零轨迹的精确导数为w的时间步次方，LSTM曲线仅表示固定忘记门的直接路径乘积，不能当作一般总梯度。',1200,700,body)

 body=text(35,45,'图13-3  LSTM：旧记忆与新候选相加，再控制读出',29)
 body+=box(35,150,170,85,'cₜ₋₁','旧记忆')+box(310,150,145,85,'× fₜ','保留比例')+box(600,150,115,85,'＋')+box(875,150,185,85,'cₜ','新记忆')
 for x1,x2 in [(210,300),(460,590),(720,865)]:body+=arrow(x1,192,x2,192)
 body+=box(280,345,175,85,'gₜ','新候选')+box(600,345,115,85,'× iₜ')+arrow(460,387,590,387)+arrow(657,340,657,245)
 body+=box(875,345,185,85,'tanh(cₜ)')+arrow(967,240,967,335)
 body+=box(875,510,185,85,'× oₜ → hₜ')+arrow(967,435,967,500)
 body+=box(45,515,680,90,'门 fₜ、iₜ、oₜ 与候选 gₜ','由 xₜ、hₜ₋₁ 的仿射变换及sigmoid/tanh计算')
 body+=text(45,675,'横向加法路径保留旧记忆；门本身也依赖状态，因此总梯度还包含其他路径。',23)
 save(folder,'13-03-lstm-memory.svg','LSTM门控记忆','旧记忆乘忘记门、新候选乘输入门后相加得到细胞状态，状态通过tanh和输出门得到隐藏状态。门由当前输入及之前隐藏状态计算。',1200,720,body)

 body=text(35,45,'图13-4  GRU：保留旧状态与写入候选；采用PyTorch候选公式',27)
 body+=box(45,120,205,85,'hₜ₋₁','旧状态')+box(435,120,165,85,'× zₜ','保留部分')+box(850,120,160,85,'＋ → hₜ')
 body+=arrow(255,162,425,162)+arrow(605,162,840,162)
 body+=box(45,315,290,105,'Wₕₙhₜ₋₁ + bₕₙ','先做循环仿射')+box(435,315,165,105,'× rₜ','再做重置')+arrow(340,367,425,367)
 body+=box(755,315,355,105,'nₜ = tanh(输入项 + 重置项)')+arrow(605,367,745,367)
 body+=box(755,495,355,80,'× (1−zₜ)','候选写入部分')+arrow(932,425,932,485)
 body+=arrow(1118,535,1160,535)+arrow(1160,535,1160,162)+arrow(1160,162,1020,162)
 body+=text(45,535,'rₜ、zₜ 均由 xₜ、hₜ₋₁ 计算。',23)
 body+=text(45,640,'D2L将重置放在循环矩阵乘法之前；向量情况下两种公式一般不相等。',23)
 save(folder,'13-04-gru-gates.svg','GRU保留与候选写入','更新门保留旧状态，重置门作用于循环仿射后再进入候选，另一部分候选按1减更新门写入；明确采用PyTorch形式。',1200,690,body)

 colors=[BLUE,TEAL,ORANGE];dashes=['','9 5','3 5'];runs=report['runs']
 body=text(35,45,'图13-5  实际四步延迟回忆：验证曲线与冻结模型的测试样例',28)
 # Left: validation learning curves, all same scale.
 left,top,w,h=105,135,430,320
 sx=lambda v:left+v/report['protocol']['epochs']*w
 allv=[v for r in runs for v in r['curves']['validation']];lo=math.floor(math.log10(min(allv)));hi=math.ceil(math.log10(max(allv)))
 sy=lambda v:top+(hi-math.log10(v))/(hi-lo)*h
 for power in range(lo,hi+1):body+=line(left,sy(10.**power),left+w,sy(10.**power))+text(left-8,sy(10.**power)+6,'1e'+str(power),19,INK,'end')
 for epoch in sorted({0,report['protocol']['epochs']//2,report['protocol']['epochs']}):body+=text(sx(epoch),top+h+30,epoch,21,INK,'middle')
 body+=line(left,top,left,top+h,INK)+line(left,top+h,left+w,top+h,INK)+text(left,105,'验证MSE（对数轴）',23)+text(395,520,'训练轮',22)
 for r,c,dash in zip(runs,colors,dashes):body+=path([(sx(e),sy(v)) for e,v in zip(r['curves']['epoch'],r['curves']['validation'])],c,3,dash)
 # Right: first test sequence, only the scored positions.
 left,top,w,h=690,135,430,320;sx=lambda t:left+(t-5)/15*w;sy=lambda y:top+(1.1-y)/2.2*h
 for y in [-1,0,1]:body+=line(left,sy(y),left+w,sy(y))+text(left-8,sy(y)+7,y,21,INK,'end')
 for t in [5,10,15,20]:body+=text(sx(t),top+h+30,t,21,INK,'middle')
 body+=line(left,top,left,top+h,INK)+line(left,top+h,left+w,top+h,INK)+text(left,105,'测试序列#1：目标与预测',23)+text(970,520,'时间步t',22)
 body+=path([(sx(t),sy(y)) for t,y in enumerate(report['data']['sample_target'],5)],INK,6,'7 6')
 for r,c,dash in zip(runs,colors,dashes):body+=path([(sx(t),sy(y)) for t,y in enumerate(r['sample_prediction'][4:],5)],c,2.7,dash)
 for j,(r,c,dash) in enumerate(zip(runs,colors,dashes)):
  x=90+j*255;body+=line(x,575,x+45,575,c,3,dash)+text(x+60,583,r['kind'],23,c)
 body+=line(900,575,945,575,INK,5,'7 6')+text(960,583,'目标',23)
 body+=text(50,640,f"3种网络均以相同隐藏宽度12训练{report['protocol']['epochs']}轮；参数量不同，不能据此证明门控的因果优势。",23)
 save(folder,'13-05-sequence-results.svg','序列学习实测结果','三个CPU网络在同一训练验证划分上的真实验证曲线，及完成选模后第一条测试序列的四步延迟预测；不是长期记忆能力排名。',1200,690,body)
