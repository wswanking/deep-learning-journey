import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const out=path.join(root,'docs/assets/08-backpropagation');fs.mkdirSync(out,{recursive:true});
const C={ink:'#183044',blue:'#176aa0',orange:'#ad5015',green:'#087c63',muted:'#4d6678'};
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const text=(x,y,s,size=22,color=C.ink,anchor='start')=>'<text x="'+x+'" y="'+y+'" font-size="'+size+'" fill="'+color+'" text-anchor="'+anchor+'">'+esc(s)+'</text>';
const box=(x,y,w,h,fill='#eff5fa')=>'<rect x="'+x+'" y="'+y+'" width="'+w+'" height="'+h+'" rx="12" fill="'+fill+'" stroke="#a8bdcd" stroke-width="2"/>';
const arrow=(x1,y1,x2,y2,color=C.blue)=>'<path d="M'+x1+' '+y1+' L'+x2+' '+y2+'" fill="none" stroke="'+color+'" stroke-width="3" marker-end="url(#arrow)"/>';
function save(name,height,title,body){
fs.writeFileSync(path.join(out,name),'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 '+height+'" role="img" aria-labelledby="title desc"><title id="title">'+esc(title)+'</title><desc id="desc">依据链式法则和本章原创数值例子绘制，蓝箭头为前向依赖，橙字为最终损失的反向梯度。</desc><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="context-stroke"/></marker></defs><rect width="1200" height="'+height+'" fill="white"/><g font-family="Microsoft YaHei,Noto Sans CJK SC,Arial,sans-serif">'+body+'</g></svg>\n');console.log(name);
}
let b=text(40,44,'图08-1  同一参数有两条影响路径：反向贡献必须相加',27);
for(const [x,y,label,value,grad] of [[60,125,'输入 x','2','∂L/∂x = 3'],[60,365,'参数 w','1','∂L/∂w = 9'],[400,125,'a = wx','2','∂L/∂a = 3'],[750,245,'s = a + w','3','∂L/∂s = 3'],[990,245,'L = s²/2','4.5','∂L/∂L = 1']]){
b+=box(x,y,180,142)+text(x+90,y+36,label,22,C.blue,'middle')+text(x+90,y+81,value,29,C.ink,'middle')+text(x+90,y+119,grad,19,C.orange,'middle');
}
b+=arrow(250,188,388,188)+arrow(245,390,388,240)+arrow(592,201,738,280)+arrow(245,435,737,360)+arrow(937,310,978,310);
b+=text(280,340,'局部：∂a/∂w=x=2',19,C.muted);
b+=text(410,452,'直接分支：∂s/∂w=1',20,C.muted);
b+=text(650,202,'局部：∂s/∂a=1',19,C.muted);
b+=box(40,550,1120,145,'#fff4e8');
b+=text(65,590,'w的间接贡献 = (∂L/∂s)(∂s/∂a)(∂a/∂w) = 3 × 1 × 2 = 6',23,C.orange);
b+=text(65,633,'w的直接贡献 = (∂L/∂s)(∂s/∂w) = 3 × 1 = 3',23,C.orange);
b+=text(65,675,'总梯度 = 6 + 3 = 9。局部导数只看当前算子，反向梯度包含之后的全部路径。',22);
save('01-shared-computation-graph.svg',735,'共享参数计算图与梯度相加',b);

b=text(40,44,'图08-2  一次前向、反向与同步更新：2 → 2 → 1 ReLU网络',27);
const blocks=[
[40,'输入 x','[1, 2]ᵀ','(2,)'],
[270,'z¹ = W¹x + b¹','[-1, 1.5]ᵀ','(2,)'],
[500,'h¹ = ReLU(z¹)','[0, 1.5]ᵀ','(2,)'],
[730,'输出 o','-1','标量'],
[960,'L = (o−1)²/2','2','标量']
];
for(const [x,title,value,shape] of blocks){b+=box(x,105,200,155)+text(x+100,144,title,21,C.blue,'middle')+text(x+100,195,value,26,C.ink,'middle')+text(x+100,236,shape,20,C.muted,'middle');}
for(const x of [245,475,705,935])b+=arrow(x,185,x+20,185);
b+=text(40,312,'W¹=[[1,−1],[0.5,0.5]]，b¹=[0,0]；W²=[2,−1]，b²=0.5；目标y=1。',22);
b+=arrow(1080,360,925,360,C.orange)+text(1000,399,'δ² = o−y = −2',22,C.orange,'middle');
b+=arrow(910,360,710,360,C.orange)+text(800,399,'对h梯度：[−4,2]',21,C.orange,'middle');
b+=arrow(695,360,455,360,C.orange)+text(575,399,'ReLU门：[0,1]',21,C.orange,'middle');
b+=arrow(440,360,205,360,C.orange)+text(320,399,'δ¹ = [0,2]ᵀ',21,C.orange,'middle');
b+=box(40,445,540,145,'#fff4e8')+text(65,486,'第一层梯度',24,C.orange);
b+=text(65,531,'∇W¹ = [[0,0],[2,4]]；∇b¹ = [0,2]',23);
b+=text(65,570,'负预激活分量的梯度被ReLU门挡住。',21,C.muted);
b+=box(620,445,540,145,'#edf7f0')+text(645,486,'第二层梯度',24,C.green);
b+=text(645,531,'∇W² = [0,−3]；∇b² = −2',23);
b+=text(645,570,'所有梯度计算完毕后再一起改参数。',21,C.muted);
b+=text(40,646,'η=0.1同步更新：新h=[0,0.3]ᵀ，输出0.49，损失0.13005（原损失2）。',24);
b+=text(40,690,'图中δ是对预激活的损失梯度；不是误差值本身，也不是参数的更新量。',22);
save('02-network-forward-backward.svg',735,'带数值的两层网络前向反向更新',b);

b=text(40,44,'图08-3  一步训练的生命周期：图、梯度缓冲区和参数各有职责',27);
for(const [x,title,line1,line2] of [[40,'① 前向建图','当前参数 → 新输出/损失','保存反向需要的中间值'],[335,'② backward','按链式法则求梯度','叶参数的.grad累加'],[630,'③ 更新参数','no_grad环境下更新','反向完成后再修改'],[925,'④ 下一步','清空旧.grad，新前向','建立下一次执行的图']]){
b+=box(x,115,235,158)+text(x+117,155,title,24,C.blue,'middle')+text(x+117,204,line1,18,C.ink,'middle')+text(x+117,244,line2,18,C.muted,'middle');
}
for(const x of [280,575,870])b+=arrow(x,195,x+45,195);
b+=box(40,330,1120,125,'#fff4e8')+text(65,371,'图的反向保存值',24,C.orange);
b+=text(65,416,'默认backward后会释放不再需要的保存值；需要这些值的同一旧图再反向通常报错。',22);
b+=box(40,485,1120,125,'#edf7f0')+text(65,526,'梯度缓冲区 .grad',24,C.green);
b+=text(65,571,'与图是否保留是两件事：新建两次前向也会累计.grad；每次独立更新前要清空。',22);
b+=text(40,668,'retain_graph保留反向所需数据；create_graph用于求导数的导数；二者都不是常规清零操作。',21);
b+=text(40,711,'detach切断依赖，但可能共享存储；no_grad控制记录，不等于模型训练/评估模式开关。',21);
save('03-autograd-lifecycle.svg',755,'计算图生命周期与梯度缓冲区',b);
