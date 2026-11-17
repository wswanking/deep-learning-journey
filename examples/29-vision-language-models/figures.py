"""Original native-SVG diagrams and plots from the actual chapter-29 report."""
from pathlib import Path
import html
import math
import numpy as np
from data import make_data

INK="#24374c"; BLUE="#1475b8"; ORANGE="#bc570e"; GREEN="#327858"
def text(x,y,s,size=18,color=INK):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}">{html.escape(str(s))}</text>'
def box(x,y,w,h,color="#eaf4fc"):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{color}" stroke="#afc4d6"/>'
def arrow(x,y,xx,yy):
    return f'<path d="M{x},{y}L{xx},{yy}" fill="none" stroke="{BLUE}" stroke-width="2" marker-end="url(#a)"/>'
def save(out,name,title,body,height=620):
    out.mkdir(parents=True,exist_ok=True)
    (out/name).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="{height}" viewBox="0 0 1080 {height}" role="img"><title>{html.escape(title)}</title><desc>原创结构与小数值示意，依据LLaVA2023第4节、BLIP2第3节和本章真实报告。图像样例以256个矩形直接表达实际RGB数组，无嵌入位图。</desc><defs><marker id="a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0L10,5L0,10Z" fill="{BLUE}"/></marker></defs><rect width="1080" height="{height}" fill="white"/><g font-family="Microsoft YaHei,Noto Sans CJK SC,sans-serif">{body}</g></svg>',encoding="utf8")

def principles(out):
    b=text(25,35,"图29-1  图像经视觉编码、连接器，成为语言模型可读取的连续前缀",22)
    b+=box(35,100,160,130,"#edf0f4")
    b+='<circle cx="115" cy="148" r="30" fill="#2143ca"/>'
    b+=text(55,210,"图像 [3,16,16]",17)+arrow(205,165,245,165)
    b+=box(255,100,220,130,"#edf0f4")+text(277,137,"冻结视觉编码器",21)+text(277,177,"4×4补丁 → 16位置",18)+text(277,210,"每位置32维",18)
    b+=arrow(485,165,525,165)+box(535,100,225,130)+text(555,137,"连接器32→64→32",20)+text(555,177,"共享MLP逐位置映射",17)+text(555,210,"4192可训练参数",18)
    b+=arrow(770,165,810,165)+box(820,100,225,130)+text(840,137,"视觉前缀 [16,32]",20)+text(840,177,"不是16个词表ID",18)+text(840,210,"再接问题/答案历史",18)
    b+='<path d="M935,240V282H200V325" fill="none" stroke="#1475b8" stroke-width="2"/>'
    b+=box(65,335,285,100)+text(85,373,"16视觉 + T文字位置",21)+text(85,410,"总长度≤64，显式因果mask",17)
    b+=arrow(360,385,425,385)+box(435,335,270,100)+text(455,373,"因果微型语言模型",21)+text(455,410,"逐字节预测分布 [256]",18)
    b+=arrow(715,385,780,385)+box(790,335,255,100,"#fff2e6")+text(810,373,"describe: → blue circle",18)+text(810,410,"生成答案字节，遇LF停",17)
    b+=text(50,500,"阶段1：视觉/语言都冻结，只训连接器；阶段2：视觉仍冻结，连接器+语言模型联合更新。",18)
    b+=text(50,550,"图标是概念示意；实际数据是原创16×16 RGB噪声图，不是自然照片或大型VLM。",18,ORANGE)
    save(out,"01-prefix-architecture.svg","连续视觉前缀的完整形状与两阶段训练",b,600)

    b=text(25,35,"图29-2  “同维度”只是接口条件；连接器还需学会被语言模型使用",22)
    b+=box(50,90,450,135,"#edf0f4")+text(75,128,"离散文字",22)+text(75,171,"byte ID 65 → embedding查表 → 32维",18)+text(75,205,"输入是整数类别；输出是连续向量",18)
    b+=box(550,90,480,135)+text(575,128,"连续视觉",22)+text(575,171,"图像特征32维 → 学习连接器 → 32维",18)+text(575,205,"没有先强行挑一个“最相近的单词ID”",18)
    b+=text(50,285,"线性手算：Z = F Wᵀ + b；2个位置、视觉2维，输出语言3维。",20)
    b+=box(50,325,270,145)+text(75,365,"F = [1 2; 3 4]",22)+text(75,410,"每一行是一个位置",18)
    b+=arrow(330,395,385,395)
    b+=box(395,325,320,145)+text(415,365,"W = [1 0; 0 1; 1 −1]",20)+text(415,410,"b = [0.5, −0.5, 1]",20)
    b+=arrow(725,395,775,395)
    b+=box(785,325,245,145,"#fff2e6")+text(805,365,"Z = [1.5 1.5 0;",20)+text(845,410,"3.5 3.5 0]",20)
    b+=text(50,530,"仅有Z的形状正确不等于语义已对齐；回答损失经冻结语言模型把梯度传回连接器。",18)
    b+=text(50,575,"本机使用两层MLP；上面的线性例子只为手算与NumPy/PyTorch逐项核对。",18,ORANGE)
    save(out,"02-continuous-connector.svg","离散文字与连续视觉表示、连接器的原创手算",b)

    b=text(25,35,"图29-3  视觉前缀、问题与回答：只有右移后的答案目标计入损失",22)
    names=["v₀","v₁","Q",":","A"];targets=["忽略","忽略","忽略","A","LF"]
    b+=text(45,100,"输入位置",18)+text(45,180,"监督目标",18)
    for i in range(5):
        x=165+165*i
        b+=box(x,65,130,55,"#edf0f4" if i<2 else "#eaf4fc")+text(x+50,101,names[i],23)
        b+=box(x,145,130,55,"#fff2e6" if i>=3 else "#edf0f4")+text(x+40,181,targets[i],20)
        b+=text(x+56,235,str(i),16)
    b+=text(45,285,"P=2，问题Q:有2字节；首回答A由位置P+2−1=3（冒号）预测。",19)
    b+=text(45,330,"视觉编码器已让补丁相互看见；下面是语言解码器中的因果可见矩阵：",18)
    for i in range(5):
        for j in range(5):
            b+=f'<rect x="{620+j*47}" y="{355+i*43}" width="42" height="38" fill="{"#c9e3f5" if j<=i else "#edf0f4"}"/>'
            b+=text(634+j*47,381+i*43,"✓" if j<=i else "×",16,BLUE if j<=i else "#8c98a4")
        b+=text(594,381+i*43,names[i],16)
        b+=text(631+i*47,346,names[i],16)
    b+=text(50,390,"答案真实概率0.8、0.5",20)+text(50,435,"NLL = −(log0.8+log0.5)/2",19)+text(50,480,"≈ 0.458145 nats/回答byte",19,BLUE)
    b+=text(50,550,"不监督视觉前缀≠不需要视觉梯度；它仍通过上下文影响答案。",17,ORANGE)
    save(out,"03-prefix-supervision.svg","连续视觉前缀的因果mask和回答监督精确对齐",b)

def write_figures(report,out):
    out=Path(out);principles(out)
    stages=report["stages"]
    b=text(25,35,"图29-4  真实验证选择：先训练连接器，再联合微调/恒定空图对照",22)
    for panel in range(2):
        left=90+panel*520
        relevant=[stages[0]] if panel==0 else stages[1:]
        top=math.ceil(max(p["validation_mean_nll"] for r in relevant for p in r["history"]))
        xmax=max(r["updates"] for r in relevant)
        b+=text(left,90,"阶段1：只看describe" if panel==0 else "阶段2：三种问题",20)
        for tick in range(5):
            val=top*tick/4;y=430-290*val/top
            b+=f'<path d="M{left},{y}H{left+390}" stroke="#e0e7ef"/>'
            b+=text(left-42,y+5,f"{val:g}",14)
            step=xmax*tick/4
            b+=text(left+390*tick/4-10,460,f"{step:g}",14)
        for i,r in enumerate(relevant):
            col=BLUE if panel==0 or i==0 else ORANGE
            pts=" ".join(f'{left+390*p["step"]/xmax},{430-290*p["validation_mean_nll"]/top}' for p in r["history"])
            dash = ' stroke-dasharray="8 6"' if panel == 1 and i == 1 else ""
            b+=f'<polyline points="{pts}" stroke="{col}" fill="none" stroke-width="2.7"{dash}/>'
        b+=text(left+125,500,"该阶段更新步数",18)
    b+=text(65,550,"蓝实线：真实图像；橙虚线：恒定黑图；纵轴=验证回答平均NLL（自然对数）。",18)
    b+=text(65,590,"两个阶段目标集合不同，曲线不直接作同一任务的优劣比较；测试不参与选择。",17,ORANGE)
    save(out,"04-training-curves.svg","两阶段视觉语言实践的真实验证轨迹",b,640)

    b=text(25,35,"图29-5  生成时是否依据图像：完整回答率与换图/遮图诊断",22)
    rows=[("只训连接器",stages[0]["test_generation"]["exact_match"]),
          ("联合微调",stages[1]["test_generation"]["exact_match"]),
          ("恒定黑图训练",stages[2]["test_generation"]["exact_match"]),
          ("联合模型：换图仍按原答案评分",report["interventions"]["shuffled"]["exact_match"]),
          ("联合模型：换图按新图答案评分",report["interventions"]["shuffled"]["counterfactual_image_accuracy"]),
          ("联合模型：用黑图遮挡",report["interventions"]["blank_image"]["exact_match"])]
    for i,(name,value) in enumerate(rows):
        y=105+65*i
        b+=text(45,y+20,name,18)
        b+=f'<rect x="490" y="{y}" width="{440*value}" height="28" fill="{BLUE if i<3 else ORANGE}"/>'
        b+=text(500+440*value,y+22,f"{value*100:.2f}%",18)
    b+=text(45,540,"同9类、新渲染seed；测试108图×3问题=324条。换图规则让颜色和形状都变。",18)
    b+=text(45,585,"遮挡是分布外诊断，不能将低分自动解释为统一“幻觉率”；这里只测封闭属性。",17,ORANGE)
    save(out,"05-grounding-results.svg","真实完整回答指标与图像干预结果",b,640)

    # Exact 16x16 data are expressed as native vector cells, not a bitmap payload.
    data,_=make_data();records=data["test"]["records"]
    predictions=stages[1]["test_generation"]["predictions"]
    chosen=next((p for p in predictions if p["task"]=="describe" and not p["exact_match"]),
                next(p for p in predictions if p["task"]=="describe"))
    image_index=chosen["image_index"];views=report["data"]["splits"]["test"]["views_per_class"]
    counter_index=((image_index//views+4)%9)*views+image_index%views
    shuffled={p["id"]:p for p in report["interventions"]["shuffled"]["predictions"]}[chosen["id"]]
    blanked={p["id"]:p for p in report["interventions"]["blank_image"]["predictions"]}[chosen["id"]]
    b=text(25,35,"图29-6  同一个describe问题，观察原图、换图与黑图时的实际生成",22)
    panels=[("原图",data["test"]["images"][image_index],chosen,chosen["expected"]),
            ("换图",data["test"]["images"][counter_index],shuffled,shuffled["counterfactual_expected"]),
            ("全黑图",np.zeros((3,16,16),dtype=np.float32),blanked,"没有封闭类别真值")]
    for i,(label,array,pred,truth) in enumerate(panels):
        left=45+350*i
        b+=text(left,90,label,21)
        rgb=np.asarray(array).transpose(1,2,0)
        for row in range(16):
            for col in range(16):
                color="#"+"".join(f"{round(float(v)*255):02x}" for v in rgb[row,col])
                b+=f'<rect x="{left+col*13}" y="{115+row*13}" width="13" height="13" fill="{color}"/>'
        b+=text(left,365,"图中真值："+truth,17)
        current_correct = pred["stopped"] and pred["generated"] == truth if i < 2 else None
        b+=text(left,410,"生成："+pred["generated"],18,BLUE if current_correct else ORANGE)
        b+=text(left,450,"按原答案正确："+str(pred["exact_match"]),16)
        b+=text(left,485,"按当前图正确："+str(current_correct) if i < 2 else "黑图无封闭类别真值",16,BLUE if current_correct else INK)
    b+=text(45,525,"每张16×16输入用256个SVG矩形逐格表达实测RGB数组，不是位图截图。",18)
    b+=text(45,570,"缺图不提供新类别真值；换图后要分别按原问题样本与实际替换图解释结果。",18,ORANGE)
    save(out,"06-image-interventions.svg","实际逐格矢量输入与同问题生成对照",b)
