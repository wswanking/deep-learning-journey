"""Vector diagrams for the Transformer chapter; attention grids use measured weights."""
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
 return line(x1,y1,x2,y2,color,3)+f'<path d="M {x2-11*math.cos(a-.5):.2f} {y2-11*math.sin(a-.5):.2f} L {x2:.2f} {y2:.2f} L {x2-11*math.cos(a+.5):.2f} {y2-11*math.sin(a+.5):.2f}" stroke="{color}" fill="none" stroke-width="3"/>'
def box(x,y,w,h,label,sub=''):
 s=f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#eef5fa" stroke="{GRID}" stroke-width="2"/>'+txt(x+w/2,y+h/2+(0 if sub else 8),label,25,INK,'middle')
 return s+(txt(x+w/2,y+h/2+29,sub,20,INK,'middle') if sub else '')
def save(folder,name,title,desc,w,h,body):
 (folder/name).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title desc"><title id="title">{html.escape(title)}</title><desc id="desc">{html.escape(desc)}</desc><rect width="{w}" height="{h}" fill="white"/><g font-family="Microsoft YaHei, Segoe UI, sans-serif">{body}</g></svg>\n',encoding='utf-8')
def write_figures(report,folder:Path):
 folder.mkdir(parents=True,exist_ok=True)
 body=txt(35,45,'图16-1  原始encoder–decoder结构：先看数据流，再拆解子层',28)
 body+=txt(85,95,'原始Post-LN：每个子层后做残差相加与LayerNorm；本章实验采用Pre-LN。',23)
 # Upward flow. A block has shared design, separate parameters across depth.
 for x,title in [(120,'编码器 Encoder'),(730,'解码器 Decoder')]:body+=txt(x+170,155,title,28,INK,'middle')
 left=[(890,'源序列嵌入＋位置','Tₛ × d'),(735,'多头自注意力','Q、K、V均来自源表示'),(645,'Add & Norm','残差相加后归一化'),(555,'逐位置FFN','相同MLP作用于每个位置'),(465,'Add & Norm','一层的输出仍为 Tₛ × d'),(320,'编码器最终表示','供每层交叉注意力读取')]
 for y,label,sub in left:body+=box(120,y,340,75,label,sub)
 for y1,y2 in [(890,810),(735,720),(645,630),(555,540),(465,395)]:body+=arrow(290,y1,290,y2)
 body+=line(80,450,80,825,TEAL,3)+txt(45,625,'N层',24,TEAL,'middle')
 right=[(890,'右移目标嵌入＋位置','Tᵧ × d'),(785,'因果多头自注意力','只读当前及更早输入位置'),(695,'Add & Norm',''),(605,'交叉注意力','Q来自目标；K、V来自源'),(515,'Add & Norm',''),(425,'逐位置FFN',''),(335,'Add & Norm','一层的输出仍为 Tᵧ × d'),(220,'线性读出＋softmax','每位置输出词表概率')]
 for y,label,sub in right:body+=box(730,y,340,75,label,sub)
 for lower,upper in zip([890,785,695,605,515,425,335],[785,695,605,515,425,335,220]):body+=arrow(900,lower,900,upper+75)
 body+=line(1110,325,1110,870,TEAL,3)+txt(1150,600,'N层',24,TEAL,'middle')
 body+=arrow(465,357,590,357)+line(590,357,590,642,INK,3)+arrow(590,642,720,642)+txt(475,320,'源表示作为K、V',22)
 body+=txt(80,1030,'N层表示重复同一种结构；不同深度拥有各自的参数，并非时间上的权重共享。',23)
 save(folder,'16-01-encoder-decoder.svg','原始Transformer数据流','依据Vaswani等2017图1与第3节重构，编码器自注意力和FFN，解码器因果自注意力、交叉注意力和FFN，各子层采用原始Post-LN。',1200,1080,body)

 body=txt(35,45,'图16-2  两个头的形状：投影、分头、独立权重、拼接与读出',28)
 body+=box(35,240,190,105,'X','B×T×8')+box(300,240,220,105,'Q、K、V投影','各 B×T×8')+arrow(230,292,290,292)
 body+=box(625,115,230,115,'头1：A₁V₁','Q₁/K₁/V₁: B×T×4')+box(625,365,230,115,'头2：A₂V₂','Q₂/K₂/V₂: B×T×4')
 body+=arrow(525,270,615,172)+arrow(525,320,615,422)
 body+=box(935,240,230,105,'拼接＋输出投影','B×T×8')+arrow(860,172,925,270)+arrow(860,422,925,320)
 body+=txt(50,560,'每头 A 的形状为 B×T×T，按键位置归一化；头之间不先取平均。',24)
 body+=txt(50,615,'代码实际分头形状为 B×2×T×4；所有投影权重按输出维×输入维存储。',23)
 save(folder,'16-02-head-shapes.svg','多头注意力与形状','宽度8分为两个宽度4的头，各头自己计算注意力权重，然后拼接并投影回宽度8。',1200,660,body)

 body=txt(35,45,'图16-4  本章微型Pre-LN block：两条残差路径保持宽度',29)
 for x,w,label in [(40,180,'输入X'),(300,220,'LN₁→MHA'),(620,180,'残差相加U'),(900,240,'LN₂→FFN')]:
  body+=box(x,220,w,90,label,'B×T×8')
 body+=arrow(225,265,290,265)+arrow(525,265,610,265)+arrow(805,265,890,265)
 body+=line(130,215,130,125,TEAL,3)+line(130,125,710,125,TEAL,3)+arrow(710,125,710,210,TEAL)
 body+=box(900,455,240,90,'残差相加Y','B×T×8')+arrow(1020,315,1020,445)
 body+=line(710,315,710,500,TEAL,3)+arrow(710,500,890,500,TEAL)
 body+=txt(245,110,'U = X + MHA(LN₁(X))',24,TEAL)+txt(55,565,'Y = U + FFN(LN₂(U))',25,TEAL)
 body+=txt(55,625,'LN按每个位置的8个特征归一化；FFN只变换特征，不混合不同位置。',23)
 save(folder,'16-04-prenorm-block.svg','Pre-LN残差块','注意力前做第一层归一化并加回输入，再在第二次归一化后做逐位置FFN并加回中间状态；两条残差均保持宽度。',1200,670,body)

 body=txt(35,45,'图16-5  真实因果权重：行是查询位置，列是可读取的键位置',28)
 for head,matrix in enumerate(report['attention_first_sample']):
  left=160+head*550;top=145;cell=70
  body+=txt(left+175,100,'头'+str(head+1),27,INK,'middle')
  for j in range(5):body+=txt(left+j*cell+35,130,j+1,22,INK,'middle')
  for i,row in enumerate(matrix):
   body+=txt(left-25,top+i*cell+43,i+1,22,INK,'middle')
   for j,value in enumerate(row):
    if j>i:fill='#edf0f2';label='×'
    else:
     # Numbers remain legible; shape and explicit x mark forbidden cells.
     level=int(245-100*value);fill=f'rgb({level},{min(255,level+8)},255)';label=f'{value:.2f}'
    body+=f'<rect x="{left+j*cell}" y="{top+i*cell}" width="{cell}" height="{cell}" fill="{fill}" stroke="white" stroke-width="2"/>'+txt(left+j*cell+35,top+i*cell+43,label,22,INK,'middle')
 body+=txt(50,555,'灰色×为被禁止的未来位置，权重精确为0；其余标注保留两位小数。',24)
 body+=txt(50,610,'原始权重每行和为1；来自种子1601、未训练block的第1条样本。',23)
 save(folder,'16-05-measured-attention.svg','微型block实测注意力','两个头的五乘五因果注意力矩阵，每行按允许的键归一化，未来位置为零；颜色与数字同时给出权重。',1200,650,body)

 body=txt(35,45,'图16-6  三类架构与训练目标：可见范围和监督位置需要一起规定',27)
 for j,(title,lines) in enumerate([
  ('Encoder-only / BERT',['双向自注意力','输入含被选择的遮蔽/替换','仅选中位置预测原词','原始BERT另有NSP目标']),
  ('Decoder-only / GPT',['因果自注意力','输入右移，预测下一个词','训练各有效位置并行计算','生成按新词逐步进行']),
  ('Encoder–decoder',['源侧双向编码','目标侧因果自注意力','交叉注意力读取源表示','预测目标下一词，条件是源'])]):
  x=35+j*395;body+=box(x,110,370,75,title)
  for k,item in enumerate(lines):body+=txt(x+20,245+k*65,item,23)
 body+=line(425,195,425,500)+line(820,195,820,500)
 body+=txt(50,560,'“遮住输入词”与“遮住注意力边”不是同一种mask；结构名称也不能代替损失定义。',23)
 save(folder,'16-06-model-families.svg','三类Transformer架构','BERT编码器以双向上下文重建被选中的词并原始加入句间目标，GPT因果预测下一词，编码器解码器以源表示为条件预测目标。',1230,610,body)

 body=txt(35,45,'图16-3  位置编码的不同频率：以宽度8、零基位置p为例',28)
 left,top,w,h=100,115,760,340
 sx=lambda p:left+p/20*w;sy=lambda v:top+(1.1-v)/2.2*h
 for y in [-1,0,1]:body+=line(left,sy(y),left+w,sy(y))+txt(left-10,sy(y)+7,y,22,INK,'end')
 for p in [0,5,10,15,20]:body+=line(sx(p),top,sx(p),top+h)+txt(sx(p),top+h+30,p,22,INK,'middle')
 for j,(fn,label,color,dash) in enumerate([(lambda p:math.sin(p),'sin(p)',BLUE,''),(lambda p:math.cos(p),'cos(p)',INK,'8 5'),(lambda p:math.sin(p/10),'sin(p/10)',TEAL,'12 4'),(lambda p:math.cos(p/10),'cos(p/10)',ORANGE,'3 5')]):
  points=[(sx(i/10),sy(fn(i/10))) for i in range(201)];d='M '+' L '.join(f'{x:.2f} {y:.2f}' for x,y in points)
  body+=f'<path d="{d}" stroke="{color}" stroke-width="3" fill="none"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>'
  body+=line(900,160+j*70,950,160+j*70,color,3,dash)+txt(965,168+j*70,label,23,color)
 body+=line(left,top,left,top+h,INK)+line(left,top+h,left+w,top+h,INK)+txt(50,100,'编码分量值',23)+txt(650,525,'零基位置p',23)
 body+=txt(50,590,'曲线说明频率不同；网络仍需学习怎样使用位置，未保证任意长度外推。',23)
 save(folder,'16-03-position-encoding.svg','正弦位置编码频率','宽度8时前四个通道分别为sin位置、cos位置、sin位置除10、cos位置除10；按公式绘制的连续曲线，实际编码只在整数位置采样。',1200,640,body)
