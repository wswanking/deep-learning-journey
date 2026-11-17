"""Editable true-SVG diagrams and exact experiment plots for chapter 25."""
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
def write_figures(result,folder):
 body=t(35,45,'图25-1 预填充先产生首个输出分布；随后每步只读一个新词元',28)
 body+=box(40,100,250,80,'提示：P个已知词元')+arrow(290,140,350,140)+box(350,100,265,80,'prefill：并行读提示')+arrow(615,140,680,140)+box(680,100,200,80,'分布 → 词元1',TEAL)
 body+=box(350,260,265,80,'decode：读词元1',ORANGE)+arrow(615,300,680,300,ORANGE)+box(680,260,200,80,'分布 → 词元2',TEAL)
 body+=line(780,180,780,220,TEAL,3)+line(480,220,780,220,TEAL,3)+line(480,220,480,260,TEAL,3)
 body+=t(40,405,'TTFT ≈ 预填充 + 首次选择；总输出G个词元需 G−1 次单词元解码。',24)
 body+=t(40,458,'这里只展示模型计算；真实服务还包括排队、分词、调度、传输及输出处理。',23)
 save(folder,'25-01-inference-stages.svg','预填充和解码时间线','提示预填充产生第一个输出分布，喂入新词元后依次产生后续分布；首输出与后续延迟分开。',1000,510,body)
 body=t(35,45,'图25-2 单层单头KV缓存：旧K/V保留，新查询读取全部可见历史',28)
 body+=box(40,115,220,80,'新输入 → 新隐藏')+arrow(260,155,330,155)+box(330,115,210,80,'新q、新k、新v')
 body+=box(330,300,210,80,'历史K/V缓存',TEAL)+arrow(540,340,610,340,TEAL)+box(610,300,280,80,'追加新k/v：长度 S+1',TEAL)
 body+=line(435,195,435,235,BLUE,3)+line(435,235,750,235,BLUE,3)+line(750,235,750,300,BLUE,3)
 body+=box(610,115,280,80,'q读全部K/V → 当前输出',ORANGE)+arrow(540,155,610,155)+t(565,140,'q',21)
 body+=line(890,155,960,155,ORANGE,3)+line(960,155,960,340,ORANGE,3)+line(890,340,960,340,ORANGE,3)+f'<path d="M900,148L890,155L900,162" fill="none" stroke="{ORANGE}" stroke-width="3"/>'+t(970,255,'K/V',20)+t(460,228,'k/v',21)
 body+=t(40,455,'每层分别缓存；仅旧K/V可复用，当前q和当前输出仍须计算。',24)
 body+=t(40,505,'位置从S继续；新分块中第r行只看键位置 ≤ S+r。',24)
 body+=t(40,555,'因果性保证旧隐藏不受未来影响；更换前缀、权重或位置方案后不能直接沿用。',22)
 save(folder,'25-02-kv-cache.svg','KV缓存数据流','当前q读取历史和新追加的键值，逐层保留缓存；新位置必须接续历史位置。',1040,600,body)
 s=result['checks']['sampling'];body=t(35,45,'图25-3 手算分布：先调整温度，再筛选，最后重新归一化',28)
 series=[('原始',s['base']),('温度0.5',s['temperature_half']),('top-k=2',s['top_k_2']),('top-p=0.8',s['top_p_08'])]
 for j,(label,probs) in enumerate(series):
  left=65+j*300;top=130;h=290;w=240
  body+=t(left,100,label,24)+line(left,top+h,left+w,top+h,INK)+line(left,top,left,top+h,INK)
  for i,p in enumerate(probs):
   x=left+12+i*56;y=top+(1-p)*h;body+=f'<rect x="{x}" y="{y}" width="35" height="{p*h}" fill="{BLUE if j<2 else TEAL}"/>'+t(x+17.5,y-8,f'{p:.3f}',17,INK,'middle')+t(x+17.5,top+h+30,['A','B','C','D'][i],21,INK,'middle')
 body+=t(40,515,'原分布：[0.60, 0.25, 0.10, 0.05]；top-p必须保留使累计概率首次达到阈值的词元。',23)
 body+=t(40,563,'提高温度不等于增加知识；本例top-k与top-p同分布，仅在这组数字上相同。',23)
 save(folder,'25-03-sampling.svg','温度与截断采样的手算','四种分布对照，纵轴为概率零到一，横轴为候选词元A至D，显示真实手算结果。',1280,610,body)
 q=result['quantization'];body=t(35,45,'图25-4 逐行对称int8量化：保存整数与比例尺，近似重建权重',28)
 body+=t(40,110,'原权重',24)+t(340,110,'q = round(w/s)',24)+t(730,110,'重建 ŵ = s q',24)
 for i,(original,code,recon) in enumerate(zip([-1,-.2,0,.7,1],q['hand_q'][0],q['hand_reconstructed'][0])):
  y=155+i*63;body+=t(80,y,f'{original:g}',24)+t(410,y,str(code),24)+t(780,y,f'{recon:.6f}',24)+arrow(160,y-9,310,y-9)+arrow(490,y-9,690,y-9)
 body+=t(40,505,'s = max|w|/127 = 1/127；比例尺也占空间；浮点重建仍有舍入误差。',23)
 body+=t(40,555,'本实验存int8，计算时还原float32；未实现整数矩阵内核，不能据此声称推理加速。',22)
 save(folder,'25-04-quantization.svg','量化和反量化手算','五个权重逐项映射int8和重建值，区分存储压缩与计算内核加速。',1130,600,body)
 rows=result['timings'];body=t(35,45,'图25-5 实测：同一前缀与15次解码，区分预填充和后续解码',28)
 maxv=max(r['total_p90_ms'] for r in rows)*1.08;left=270;right=1160;sy=lambda v:left+v/maxv*(right-left)
 for j,r in enumerate(rows):
  y=112+j*57;body+=t(30,y+22,f"B{r['batch']} P{r['prompt_bytes']} "+('缓存' if r['cache'] else '完整重算'),21)
  a=r['prefill_median_ms'];b=r['decode15_median_ms'];body+=f'<rect x="{left}" y="{y}" width="{sy(a)-left}" height="27" fill="{BLUE}"/><rect x="{sy(a)}" y="{y}" width="{sy(a+b)-sy(a)}" height="27" fill="{TEAL}"/>'
  body+=t(sy(a+b)+8,y+22,f"{r['total_median_ms']:.2f}",18)
 body+=line(left,600,right,600,INK)
 for k in range(5):
  v=maxv*k/4;body+=line(sy(v),595,sy(v),608,INK)+t(sy(v),638,f'{v:.1f}',20,INK,'middle')
 body+=t(600,680,'每个固定工作负载耗时（毫秒）',22)+f'<rect x="40" y="710" width="25" height="25" fill="{BLUE}"/>'+t(80,731,'预填充中位数',22)+f'<rect x="350" y="710" width="25" height="25" fill="{TEAL}"/>'+t(390,731,'15次解码中位数',22)
 body+=t(40,778,'31次交替测量；条长为分项中位数之和，末尾数为总耗时中位数，两者可能略有差别。',22)
 save(folder,'25-05-inference-results.svg','缓存效率实测','batch和提示长度固定，分别显示预填充和十五次解码中位数；所有值来自本机真实CPU实验。',1280,825,body)
