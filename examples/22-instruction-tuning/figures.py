"""Original vector diagrams and actual SFT comparison curves."""
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
 return line(x1,y1,x2,y2,color,3)+f'<path d="M {x2-11*math.cos(a-.5):.2f} {y2-11*math.sin(a-.5):.2f} L {x2:.2f} {y2:.2f} L {x2-11*math.cos(a+.5):.2f} {y2-11*math.sin(a+.5):.2f}" fill="none" stroke="{color}" stroke-width="3"/>'
def box(x,y,w,h,label,sub='',color=INK):
 s=f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#eef5fa" stroke="{GRID}" stroke-width="2"/>'+txt(x+w/2,y+h/2+(0 if sub else 8),label,25,color,'middle')
 return s+(txt(x+w/2,y+h/2+29,sub,20,color,'middle') if sub else '')
def path(points,color,width=3,dash=''):
 d='M '+' L '.join(f'{x:.2f} {y:.2f}' for x,y in points)
 return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>'
def save(folder,name,title,desc,w,h,body):
 (folder/name).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title desc"><title id="title">{html.escape(title)}</title><desc id="desc">{html.escape(desc)}</desc><rect width="{w}" height="{h}" fill="white"/><g font-family="Microsoft YaHei, Segoe UI, sans-serif">{body}</g></svg>\n',encoding='utf-8')
def write_figures(report,folder:Path):
 folder.mkdir(parents=True,exist_ok=True);runs=report['runs'];names={'base':'未微调基座','response_only':'仅监督回答','all_tokens':'整段监督'}
 body=txt(35,45,'图22-1  从预训练到SFT：同一模型结构，改变训练数据与监督目标',28)
 body+=box(40,160,230,100,'普通短句语料','第21章预训练')+box(370,160,245,100,'35,328参数基座','冻结为三组共同起点')+arrow(275,210,360,210)
 body+=box(775,70,385,95,'不更新基座','只做最终评价')+box(775,245,385,95,'回答监督SFT','只计回答及结束换行')+box(775,420,385,95,'整段监督对照','计提示和回答的下词损失')
 body+=arrow(620,180,765,118)+arrow(620,215,765,292)+arrow(620,245,765,467)
 body+=txt(45,615,'两种SFT使用同一权重、批次顺序与800更新；验证选模后才比较测试生成。',24)
 save(folder,'22-01-pretraining-sft.svg','预训练与SFT对照','同一个第21章微型语言模型基座复制为不更新、回答监督和整段监督三组，只有训练监督目标改变。',1240,660,body)

 body=txt(35,45,'图22-2  本章文本模板：角色标记也是UTF-8字节，不是新增特殊词元',27)
 body+=box(40,170,710,140,'[指令] ↵ 复制 af ↵ [回答] ↵','提示：28字节，位置0…27',BLUE)
 body+=box(810,170,150,140,'af','2字节',TEAL)+box(1020,170,150,140,'↵','1字节',TEAL)
 body+=arrow(755,240,800,240)+arrow(965,240,1010,240)
 body+=txt(45,390,'全串：提示＋回答＋LF；此短提示共31字节，长措辞模板共37字节。',24)
 body+=txt(45,455,'训练输入去掉最后一个字节；标签从第二字节开始，模型一直预测下一个字节。',23)
 body+=txt(45,520,'推理只提供提示，等待模型续写；LF字节10是本任务终止符，不是全局EOS。',23)
 save(folder,'22-02-chat-template.svg','单轮指令文本模板','明确本章指令和回答标记的文本序列，提示占28字节，两个回答字节与一个换行参与监督；模板使用固定字节词表。',1240,580,body)

 body=txt(35,45,'图22-3  右移后的监督位置：提示最后位置要预测回答第一个字节',27)
 body+=txt(45,110,'手算简化词表：Q=0、:=1、A=2、LF=3。完整序列为 Q : A LF。',23)
 labels=['输入位置r','输入字节','下词标签','回答loss mask'];columns=[['0','Q',':','0：忽略'],['1',':','A','1：监督'],['2','A','LF','1：监督'],['3','PAD','忽略','0：忽略']]
 for row,label in enumerate(labels):body+=txt(45,195+row*85,label,24)
 for col,items in enumerate(columns):
  x=275+col*225
  for row,item in enumerate(items):body+=box(x,145+row*85,200,66,item,color=TEAL if col in [1,2] and row>0 else INK)
 body+=arrow(350,520,735,520,BLUE)+txt(410,560,'提示仍作为可见上下文影响回答',23,BLUE)
 body+=txt(45,635,'被忽略位置的输出logits局部梯度为0；提示嵌入仍可通过回答损失得到梯度。',23)
 save(folder,'22-03-shifted-supervision.svg','标签右移与回答监督位置','提示Q冒号后预测A换行，mask对齐下一个目标字节而不是当前输入；padding不监督。提示仍可为后续回答提供上下文。',1240,690,body)

 body=txt(35,45,'图22-4  按底层字母对分组：同一问题的变体一起进入一个划分',28)
 body+=box(40,175,330,145,'一组 pair:ab','ab / ba 两个顺序')+box(455,120,330,100,'两项任务','复制 / 反转')+box(455,315,330,100,'各两种措辞','共8条关联记录')
 body+=arrow(375,230,445,170)+arrow(375,285,445,365)
 body+=box(890,175,300,145,'整体分组后划分','不把8条拆到不同集合')+arrow(790,170,880,225)+arrow(790,365,880,280)
 body+=txt(50,515,'15组按固定种子分成9 / 3 / 3组，对应训练72、验证24、测试24条。',24)
 body+=txt(50,580,'逐行随机划分可能让同一字母对的近重复变体跨集，夸大未见组合的表现。',23)
 save(folder,'22-04-grouped-data.svg','关联指令分组划分','一对不同字母的两个顺序、两个任务、两个措辞共有八条关联记录，按无序字母对分组整体划分，避免近重复跨集合。',1240,630,body)

 body=txt(35,45,'图22-5  实测：验证回答NLL与训练/测试整答准确率',29)
 left,top,w,h=110,140,475,330;lo=-.2;hi=1.1
 sx=lambda k:left+k/report['protocol']['updates']*w;sy=lambda v:top+(hi-math.log10(v))/(hi-lo)*h
 for val in [1,2,5,10]:body+=line(left,sy(val),left+w,sy(val))+txt(left-10,sy(val)+7,val,22,INK,'end')
 for k in [0,200,400,600,800]:body+=txt(sx(k),top+h+30,k,21,INK,'middle')
 body+=line(left,top,left,top+h,INK)+line(left,top+h,left+w,top+h,INK)+txt(left,105,'验证回答NLL（对数轴）',23)
 for run,color,dash in zip(runs[1:],[BLUE,ORANGE],['','8 5']):
  body+=path([(sx(k),sy(v)) for k,v in zip(run['curves']['update'],run['curves']['validation_response_nll'])],color,3,dash)
  body+=line(sx(run['best_update']),top,sx(run['best_update']),top+h,color,2,'3 5')
 baseline=runs[0]['best_validation_response_nll'];body+=line(left,sy(baseline),left+w,sy(baseline),INK,2,'3 5')
 # Right bars are actual independent test metrics, no smoothing or decorative values.
 left,top,w,h=770,140,410,330;sy=lambda v:top+(1-v)*h
 for val in [0,.5,1]:body+=line(left,sy(val),left+w,sy(val))+txt(left-10,sy(val)+7,f'{val:.0%}',21,INK,'end')
 body+=txt(left,105,'完整回答严格匹配',23)+line(left,top,left,top+h,INK)+line(left,top+h,left+w,top+h,INK)
 for j,run in enumerate(runs):
  center=left+65+j*145
  for offset,key,color in [(-28,'train_generation',BLUE),(28,'test_generation',TEAL)]:
   value=run[key]['exact_match'];x=center+offset-20;y=sy(value);body+=f'<rect x="{x}" y="{y}" width="40" height="{top+h-y}" fill="{color}"/>'+txt(center+offset,y-9,(f'{value:.0%}' if abs(value*100-round(value*100))<1e-8 else f'{value:.1%}'),19,color,'middle')
  body+=txt(center,top+h+36,['基座','回答','整段'][j],22,INK,'middle')
 body+=line(70,570,115,570,BLUE,3)+txt(130,578,'回答监督曲线',22,BLUE)+line(335,570,380,570,ORANGE,3,'8 5')+txt(395,578,'整段监督曲线',22,ORANGE)
 body+=f'<rect x="775" y="555" width="25" height="25" fill="{BLUE}"/>'+txt(815,578,'训练',22,BLUE)+f'<rect x="970" y="555" width="25" height="25" fill="{TEAL}"/>'+txt(1010,578,'测试',22,TEAL)
 body+=txt(50,655,'两个更新目标的曲线都以同一回答NLL评价；训练准确率接近100%，未见组合仍很弱。',23)
 save(folder,'22-05-sft-results.svg','SFT实测曲线与泛化差距','相同基座和训练批次的两种监督目标，统一用验证回答NLL选择模型，再显示真实训练与测试的完整回答匹配率，保留弱泛化结果。',1280,705,body)
