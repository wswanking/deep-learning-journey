"""Original vector diagrams for the fixed-weight MLP forward experiment."""
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
    body=txt(35,48,'图07-1  两个带参数层，逐步追踪一次前向计算',29)
    blocks=[(35,'输入x','2×1','输入维度2'),(260,'仿射层1','W[1]：2×2','b[1]：2×1'),(485,'ReLU','h[1]：2×1','逐元素取max(0,z)'),(710,'仿射层2','W[2]：1×2','b[2]：1×1'),(935,'输出','z[2]：1×1','p = sigmoid(z[2])')]
    for x,title,line1,line2 in blocks:
        body+=rect(x,140,185,180)+txt(x+92,185,title,25,INK,'middle')+txt(x+92,238,line1,22,BLUE,'middle')+txt(x+92,285,line2,18,INK,'middle')
        if x<935: body+=arr(x+189,230,x+217,230)
    body+=txt(350,365,'第一层参数：4+2=6',24,TEAL,'middle')+txt(800,365,'第二层参数：2+1=3',24,TEAL,'middle')
    body+=rect(60,425,1030,125)+txt(85,470,'手算输入 x=(1,1)ᵀ：z[1]=(2,1)ᵀ → h[1]=(2,1)ᵀ → z[2]=−0.5',26)
    body+=txt(85,518,'再转为二分类概率 p≈0.37754；总参数9，批次大小不计入参数量。',24)
    save(folder,'07-01-forward-shapes.svg','MLP逐层形状与参数','输入二维，两维隐藏层先仿射再ReLU，最后一个输出分数再经sigmoid；两个参数层共9个参数。',1160,600,body)

    body=txt(35,48,'图07-2  三种激活函数：对每个坐标分别运算',29)
    items=[('ReLU',lambda z:max(0,z),-.4,4.3,TEAL),('sigmoid',lambda z:1/(1+math.exp(-z)),-.15,1.15,BLUE),('tanh',math.tanh,-1.2,1.2,ORANGE)]
    for j,(name,fn,ymin,ymax,color) in enumerate(items):
        x0=50+j*365;y0=130;w=290;h=300
        sx=lambda z:x0+(z+4)/8*w
        sy=lambda y:y0+(ymax-y)/(ymax-ymin)*h
        body+=txt(x0+w/2,98,name,27,color,'middle')
        body+=line(x0,sy(0),x0+w,sy(0),INK)+line(sx(0),y0,sx(0),y0+h,INK)
        for v in [-4,-2,2,4]: body+=line(sx(v),sy(0)-4,sx(v),sy(0)+4,INK)+txt(sx(v),y0+h+29,v,19,INK,'middle')
        ticks=[0,2,4] if j==0 else ([0,.5,1] if j==1 else [-1,0,1])
        for v in ticks: body+=line(x0,sy(v),x0+w,sy(v),GRID,1)+txt(x0-8,sy(v)+7,v,19,INK,'end')
        body+=path([(sx(-4+i*.02),sy(fn(-4+i*.02))) for i in range(401)],color,4)
        body+=txt(x0+w-2,y0+h+67,'z（无量纲）',22,INK,'end')
    body+=txt(50,555,'三图各自标明纵轴范围；曲线由解析定义生成，不是训练结果。',24)
    save(folder,'07-02-activations.svg','ReLU sigmoid tanh曲线','ReLU截断负值，sigmoid映射到0和1之间，tanh映射到负1和1之间，各面板独立标示轴范围。',1160,600,body)

    body=txt(35,45,'图07-3  XOR经过非线性表示后，可由一条直线分开',28)
    panels=[(80,145,390,320,-.2,1.2,-.2,1.2,'原输入空间 x₁、x₂'),(675,145,390,320,-.3,2.3,-.3,1.3,'隐藏空间 h₁、h₂')]
    for j,(left,top,w,h,xmin,xmax,ymin,ymax,title) in enumerate(panels):
        sx=lambda x:left+(x-xmin)/(xmax-xmin)*w
        sy=lambda y:top+(ymax-y)/(ymax-ymin)*h
        body+=txt(left+w/2,105,title,25,INK,'middle')+arr(left,sy(0),left+w,sy(0))+arr(sx(0),top+h,sx(0),top)
        for v in ([0,1] if j==0 else [0,1,2]): body+=txt(sx(v),top+h+28,v,21,INK,'middle')
        for v in [0,1]: body+=txt(left-12,sy(v)+7,v,21,INK,'end')
        if j==1:
            body+=path([(sx(0),sy(-.25)),(sx(2.3),sy(.9))],BLUE,3,'8 6')
            points=[(0,0,0,'00'),(1,0,1,'01、10'),(2,1,0,'11')]
        else: points=[(float(row[0]),float(row[1]),int(label),''.join(str(int(v)) for v in row)) for row,label in zip(report['inputs'],report['targets'])]
        for x,y,label,name in points:
            px,py=sx(x),sy(y)
            if label: body+=f'<circle cx="{px:.2f}" cy="{py:.2f}" r="12" fill="white" stroke="{TEAL}" stroke-width="4"/>'
            else: body+=f'<path d="M {px:.2f} {py-13:.2f} L {px+13:.2f} {py:.2f} L {px:.2f} {py+13:.2f} L {px-13:.2f} {py:.2f} Z" fill="{ORANGE}"/>'
            body+=txt(px+18,py-15,name,23)
    body+=arr(505,300,620,300,TEAL)+txt(563,265,'ReLU表示',22,TEAL,'middle')
    body+=txt(80,535,'空心圆：标签1；菱形：标签0。',24)+txt(675,535,'虚线：h₁−2h₂−0.5=0',24,BLUE)
    body+=txt(80,590,'01与10在隐藏空间重合，因其目标相同；表示并不需要保留全部输入细节。',23)
    save(folder,'07-03-xor-representation.svg','XOR输入空间与隐藏表示','四个XOR输入点经两单元ReLU映射为三个位置，两个正类重合，随后可用h1减2h2减0.5的阈值分开；标记形状同时区分类别。',1160,640,body)
