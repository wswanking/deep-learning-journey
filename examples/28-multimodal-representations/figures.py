"""Original editable vector drawings and plots, including rectangles for actual synthetic pixels."""
from pathlib import Path
from html import escape
import math
INK='#163451';BLUE='#2459bf';TEAL='#008b86';ORANGE='#cf6b12';GRAY='#d6e2ef'
def t(x,y,s,size=24,color=INK,anchor='start'):return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}">{escape(str(s))}</text>'
def line(x1,y1,x2,y2,color=GRAY,width=2,dash=''):return f'<path d="M{x1},{y1}L{x2},{y2}" fill="none" stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>'
def box(x,y,w,h,label,color=BLUE):return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#f5f9ff" stroke="{color}" stroke-width="2"/>'+t(x+w/2,y+h/2+8,label,24,color,'middle')
def arrow(x1,y1,x2,y2,color=BLUE):return line(x1,y1,x2,y2,color,3)+f'<path d="M{x2-10},{y2-7}L{x2},{y2}L{x2-10},{y2+7}" fill="none" stroke="{color}" stroke-width="3"/>'
def save(folder,name,title,description,w,h,body):
 folder.mkdir(parents=True,exist_ok=True);(folder/name).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img"><title>{escape(title)}</title><desc>{escape(description)}</desc><rect width="100%" height="100%" fill="white"/><g font-family="Microsoft YaHei,Arial,sans-serif">{body}</g></svg>\n',encoding='utf-8')
def write_figures(report,folder):
 body=t(35,45,'图28-1 从RGB图像到视觉词元：patch是空间小块，不是离散文字编号',28)
 body+=f'<rect x="50" y="110" width="240" height="240" fill="#10151a"/><circle cx="170" cy="230" r="78" fill="#d52c22"/>'
 for j in range(5):body+=line(50+60*j,110,50+60*j,350,'#a7bbd4')+line(50,110+60*j,290,110+60*j,'#a7bbd4')
 body+=t(50,390,'16×16×3 → 16块4×4×3',23)+arrow(300,225,355,225)+box(355,180,280,90,'48维 → 32维投影')+arrow(635,225,700,225)
 for j in range(16):body+=f'<rect x="{710+(j%8)*57}" y="{150+(j//8)*90}" width="43" height="55" fill="#dce9ff" stroke="{BLUE}"/>'+t(731+(j%8)*57,185+(j//8)*90,str(j+1),18,INK,'middle')
 body+=t(710,390,'16个连续向量 + 位置嵌入',24)+t(40,465,'编码器允许每个patch读取整幅图的其他patch，再池化为图像级表示。',24)
 body+=t(40,515,'本章保留forward_tokens输出[B,16,32]；用于检索的投影归一化表示是[B,16]。',23)
 save(folder,'28-01-visual-tokens.svg','图像patch与视觉词元','原创红圆形分为16个图像patch，展开48维并投影32维，加位置后形成连续视觉词元。',1230,560,body)
 body=t(35,45,'图28-2 图文双塔：分别编码，在同一嵌入空间比较',28)
 for y,a,b in [(110,'图像：RGB [B,3,16,16]','视觉编码器 → 图像向量'),(295,'文字：颜色/形状 [B,2]','属性编码器 → 文字向量')]:
  body+=box(40,y,320,80,a)+arrow(360,y+40,415,y+40)+box(415,y,330,80,b,TEAL)+arrow(745,y+40,800,y+40)+box(800,y,240,80,'投影 + L2归一化',ORANGE)
 body+=box(1080,205,110,90,'点积',ORANGE)+line(920,190,920,220,ORANGE,3)+line(920,220,1060,220,ORANGE,3)+line(1060,220,1060,232,ORANGE,3)+arrow(1060,232,1080,232,ORANGE)+line(1040,335,1060,335,ORANGE,3)+line(1060,335,1060,273,ORANGE,3)+arrow(1060,273,1080,273,ORANGE)
 body+=t(40,465,'图文不需逐坐标相同；匹配对的点积应高于不匹配候选。',24)+t(40,520,'相似度矩阵S = 图像嵌入 × 文字嵌入ᵀ / τ；双向交叉熵沿行/列分别归一化。',23)
 save(folder,'28-02-dual-towers.svg','图文双塔与相似度','独立图像编码器和文字属性编码器输出同维归一化向量，通过矩阵点积学习匹配。',1210,570,body)
 p=math.exp(1)/(math.exp(1)+1);body=t(35,45,'图28-3 两对样例手算：行找文字，列找图像',28)
 for j,(title,values) in enumerate([('logits S',[[1,0],[0,1]]),('沿行softmax',[[p,1-p],[1-p,p]]),('沿列softmax',[[p,1-p],[1-p,p]])]):
  x=90+j*365;body+=t(x,115,title,24)
  for r in range(2):
   for c in range(2):
    value=values[r][c];body+=f'<rect x="{x+c*113}" y="{155+r*90}" width="110" height="87" fill="{ "#dce9ff" if r==c else "#faf4ec"}" stroke="white"/>'+t(x+c*113+55,210+r*90,f'{value:.6f}' if j else str(value),22,INK,'middle')
 body+=t(40,405,'每个正确项概率 e/(e+1) ≈ 0.731059；每项负对数 ≈ 0.313262。',24)+t(40,460,'两方向各对2项取平均，再各乘1/2：最终L = log(1+e⁻¹)，不再除以矩阵4项。',23)
 save(folder,'28-03-contrastive-hand.svg','双向对比损失手算','两个正交图像向量和文字向量，温度1时对角logit1，非对角0，展示行和列归一化以及最终平均。',1180,510,body)
 body=t(35,45,'图28-4 假负例：两个不同记录可能具有相同语义',28)
 body+=box(55,120,280,75,'图像1：红色圆形')+box(55,280,280,75,'图像2：另一红圆形')+arrow(335,157,470,157)+arrow(335,317,470,317)
 body+=box(470,120,300,75,'文字1：红色圆形',TEAL)+box(470,280,300,75,'文字2：红色圆形',TEAL)
 body+=line(335,157,470,317,ORANGE,3,'7 5')+line(335,317,470,157,ORANGE,3,'7 5')
 body+=t(830,185,'实线：指定正例',23)+t(830,250,'虚线：同样相关',23,ORANGE)+t(830,315,'不能当真负例',23,ORANGE)
 body+=t(40,440,'若两条文字向量相同，任一行两个logit相同；单正例目标仍强制只有对角正确。',23)+t(40,490,'两候选时正确概率最多1/2，单正例损失至少ln2；语义未错，配对假设已不合适。',23)
 body+=t(40,540,'训练每批只放每种概念1张图；检索评价把同一概念的多张图都计为相关。',23)
 save(folder,'28-04-false-negatives.svg','假负例与多正例相关性','两幅同语义图像和两条相同文字构成多重相关，单正例对角损失存在ln2下界。',1200,590,body)
 body=t(35,45,'图28-5 实际测试样例：新渲染种子，仍是训练中见过的九种概念',28)
 names=report['data']['class_captions']
 for j,pixels in enumerate(report['representative_test_pixels']):
  x=70+(j%3)*385;y=110+(j//3)*240;scale=10
  for yy in range(16):
   for xx in range(16):
    color='#'+''.join(f'{round(pixels[k][yy][xx]*255):02x}' for k in range(3));body+=f'<rect x="{x+xx*scale}" y="{y+yy*scale}" width="{scale}" height="{scale}" fill="{color}"/>'
  body+=t(x+185,y+65,f'{j+1}. '+names[j],24)+t(x+185,y+105,'16×16 RGB',20)
 body+=t(40,835,'每个实际像素画为可编辑矢量矩形；无嵌入位图。任务不包含自然照片或自由语言。',23)
 save(folder,'28-05-synthetic-samples.svg','真实合成图形测试样例','九种颜色形状组合各取测试集第一张图，使用实际像素对应的矢量矩形显示。',1260,875,body)
 body=t(35,45,'图28-6 实测：验证目标下降，与测试图文匹配矩阵',28)
 curves=report['curves'];left=75;top=140;w=440;h=370;vals=[c['validation_class_nll'] for c in curves];low=math.log10(min(vals)*.8);high=math.log10(max(vals)*1.2);sx=lambda x:left+x/report['protocol']['updates']*w;sy=lambda y:top+(high-math.log10(y))/(high-low)*h
 body+=t(left,105,'验证图→文平均NLL（对数轴）',22)+line(left,top,left,top+h,INK)+line(left,top+h,left+w,top+h,INK)
 for power in range(math.ceil(low),math.floor(high)+1):
  y=10**power;body+=line(left,sy(y),left+w,sy(y))+t(left-10,sy(y)+6,f'{y:g}',18,INK,'end')
 path=' '.join(('M' if j==0 else 'L')+f'{sx(c["update"]):.2f},{sy(c["validation_class_nll"]):.2f}' for j,c in enumerate(curves));body+=f'<path d="{path}" fill="none" stroke="{BLUE}" stroke-width="3"/>'
 for step in [0,400,800,1200]:body+=t(sx(step),top+h+32,str(step),20,INK,'middle')
 body+=t(left+130,top+h+70,'参数更新步',22)
 matrix=report['representative_test_cosine'];left=680;top=140;cell=45;body+=t(left,105,'测试代表图→九条文字的余弦',22)
 for r in range(9):
  body+=t(left-15,top+r*cell+30,str(r+1),20,INK,'end')
  for c in range(9):
   v=matrix[r][c];intensity=min(1,abs(v));rgb=(round(255-180*intensity),round(255-135*intensity),255) if v>=0 else (255,round(255-140*intensity),round(255-150*intensity));fill='#'+''.join(f'{a:02x}' for a in rgb);body+=f'<rect x="{left+c*cell}" y="{top+r*cell}" width="{cell}" height="{cell}" fill="{fill}" stroke="white"/>'+t(left+c*cell+cell/2,top+r*cell+28,f'{v:.2f}',13,INK,'middle')
 for c in range(9):body+=t(left+c*cell+cell/2,top-13,str(c+1),20,INK,'middle')
 body+=t(650,590,'行：图像概念编号；列：文字概念编号',21)
 for j,name in enumerate(names):body+=t(60+(j%3)*390,655+(j//3)*43,f'{j+1} '+name,22)
 body+=t(40,815,'矩阵展示每类第一张测试图，指标使用全部216张；颜色及数字均表示余弦相似度。',22)
 save(folder,'28-06-contrastive-results.svg','真实训练曲线与图文相似度','验证集图文分类NLL随更新变化，以及各类别第一张测试图对九条文字的真实余弦矩阵。',1260,860,body)
