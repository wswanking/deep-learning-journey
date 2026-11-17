import {fileURLToPath} from 'node:url';
import fs from 'node:fs';
import path from 'node:path';
const out=path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../docs/assets/11-convolutional-networks');
fs.mkdirSync(out,{recursive:true});
const C={ink:'#24374c',blue:'#1475b8',light:'#eaf4fc',orange:'#bc570e',warm:'#fff2e6',line:'#afc4d6'};
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;');
const t=(x,y,s,size=18,color=C.ink,anchor='start')=>'<text x="'+x+'" y="'+y+'" font-size="'+size+'" fill="'+color+'" text-anchor="'+anchor+'">'+esc(s)+'</text>';
const rect=(x,y,w,h,fill='white',stroke=C.line,rx=4)=>'<rect x="'+x+'" y="'+y+'" width="'+w+'" height="'+h+'" rx="'+rx+'" fill="'+fill+'" stroke="'+stroke+'"/>';
const line=(x1,y1,x2,y2,color=C.blue,arrow=true)=>'<path d="M'+x1+','+y1+' L'+x2+','+y2+'" fill="none" stroke="'+color+'" stroke-width="1.8"'+(arrow?' marker-end="url(#arrow)"':'')+'/>';
function grid(x,y,values,cell=52,highlight=[]) {
 let body='';
 for(let i=0;i<values.length;i++)for(let j=0;j<values[i].length;j++) {
  const hot=highlight.some(([a,b])=>a===i&&b===j);
  body+=rect(x+j*cell,y+i*cell,cell,cell,hot?C.light:'white',C.line,0);
  body+=t(x+(j+.5)*cell,y+(i+.5)*cell+6,values[i][j],18,hot?C.blue:C.ink,'middle');
 }
 return body;
}
function write(name,title,body,h=570) {
 const svg='<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="'+h+'" viewBox="0 0 1080 '+h+'" role="img"><title>'+title+'</title><desc>原创解析示意；定义依据花书第9章、D2L第7章及PyTorch2.15。</desc><defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0L10,5L0,10Z" fill="'+C.blue+'"/></marker></defs><rect width="1080" height="'+h+'" fill="white"/><g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif">'+body+'</g></svg>';
 fs.writeFileSync(path.join(out,name),svg,'utf8');
}
let b=t(28,34,'图11-1  一个核在所有空间位置重复使用：互相关手算',23);
b+=t(55,85,'输入 X：3 × 3')+grid(55,105,[[1,2,3],[4,5,6],[7,8,9]],58,[[0,0],[0,1],[1,0],[1,1]]);
b+=t(335,85,'核 K：2 × 2')+grid(335,105,[[1,2],[0,-1]],58,[[0,0],[0,1],[1,0],[1,1]]);
b+=t(680,85,'输出 Y：2 × 2')+grid(680,105,[[0,2],[6,8]],58,[[0,0]]);
b+=line(245,165,315,165)+line(470,165,655,165)+t(490,145,'逐元素乘后求和',16);
b+=rect(55,310,960,78,C.light)+t(75,340,'左上窗口：1 × 1 + 2 × 2 + 4 × 0 + 5 × (−1) = 0',20);
b+=t(75,370,'向右移动1格：2 × 1 + 3 × 2 + 5 × 0 + 6 × (−1) = 2',18);
b+=rect(55,410,960,92,C.warm)+t(75,441,'四个窗口共享同一组4个权重；有偏置时再加1个共享标量。',20,C.orange);
b+=t(75,476,'数学卷积会将核翻转180°：本例翻转后输出 [12,14;18,20]，不能混用。',18,C.orange);
b+=t(55,540,'窗口不是矩阵乘法；这里核不翻转，偏置为0，步幅1，无填充。',16);
write('01-window-and-sharing.svg','二维互相关逐格计算与参数共享',b);
b=t(28,34,'图11-2  输入通道求和，输出通道保留；尺寸由窗口能放几次决定',22);
b+=t(50,85,'两输入通道小例子：第二通道是2X，第二片核是0.5K',18);
b+=rect(50,110,230,76,C.light)+t(70,140,'X 与 K → R',20,C.blue)+t(70,170,'R = [0,2;6,8]',17);
b+=rect(50,215,230,76,C.light)+t(70,245,'2X 与 0.5K → R',20,C.blue)+t(70,275,'按输入通道分别计算',17);
b+=line(285,148,360,195)+line(285,253,360,205)+t(370,210,'＋',35,C.blue);
b+=line(410,200,470,200)+grid(490,145,[[0,4],[12,16]],52);
b+=t(650,165,'相加得到1张输出图',19)+t(650,198,'换另一组完整核 → 另1张图',17)+t(650,231,'偏置：每个输出通道1个',17);
b+=rect(45,335,990,75,C.warm)+t(65,365,'核形状 (Cout, Cin, kh, kw)：例如 (4,3,3,3)',20);
b+=t(65,395,'参数数 = 4 × (3 × 3 × 3 + 1) = 112；不乘输出图的高与宽。',18);
b+=t(50,455,'单轴例：H=8，k=3，p=1（两侧各1），s=2，d=1',19);
for(let i=0;i<10;i++)b+=rect(65+83*i,480,77,24,i===0||i===9?'#eef0f3':C.light,C.line,0);
for(let j=0;j<4;j++){b+=line(65+166*j,522+20*j,65+166*j+3*83-6,522+20*j,C.blue,false);b+=t(65+166*j+100,538+20*j,String(j),15,C.blue);}
b+=t(50,630,'起点0、2、4、6：每次跨度3，均放在填充后的长度10内，共4个输出。',17);
write('02-channels-and-shape.svg','多通道聚合、参数量与步幅尺寸',b,660);
b=t(28,34,'图11-3  多层局部窗口累积成更大的感受野；池化减少空间分辨率',22);
b+=t(40,78,'左：一维截面，忽略边界；最后一个单元覆盖8个输入。',17);
const xs=Array.from({length:8},(_,i)=>190+42*i);
for(let i=0;i<6;i++)for(let k=0;k<3;k++)b+=line(xs[i+k],118,232+42*i,198,C.line);
for(let i=0;i<3;i++)for(let k=0;k<2;k++)b+=line(232+42*(2*i+k),238,253+84*i,310,C.line);
for(let i=0;i<3;i++)b+=line(253+84*i,350,337,418,C.line);
for(let i=0;i<8;i++)b+=rect(xs[i]-17,95,34,24,C.light)+t(xs[i],113,String(i),14,C.blue,'middle');
for(let i=0;i<6;i++)b+=rect(232+42*i-17,200,34,36,C.light);
for(let i=0;i<3;i++)b+=rect(253+84*i-24,312,48,36,C.warm);
b+=rect(309.5,420,55,38,C.light);
b+=t(20,112,'输入',15)+t(20,136,'r=1，j=1',15)+t(20,218,'卷积3 / 步1',15)+t(20,242,'r=3，j=1',15)+t(20,330,'池化2 / 步2',15)+t(20,354,'r=4，j=2',15);
b+=t(20,430,'卷积3 / 步1',15,C.blue)+t(20,454,'r=8，j=2',15,C.blue);
b+=t(600,80,'右：最大池化，2 × 2窗口，步幅1',19);
b+=grid(600,105,[[1,4,2],[3,0,5],[2,6,1]],47,[[0,1]]);
b+=line(760,175,815,175)+grid(845,127,[[4,5],[6,6]],47);
b+=t(600,295,'左上最大值4；右上最大值5。',18);
b+=t(600,330,'数值6被两个窗口选中：',18)+t(600,362,'若输出梯度都为1，它的梯度为2。',18);
b+=rect(580,405,450,90,C.warm)+t(600,435,'r：理论覆盖的输入跨度',18)+t(600,468,'j：相邻单元在原输入中的间距',18);
b+=t(45,550,'池化没有可训练核；最大值有并列时不可微，框架选择一种反向约定。',17);
write('03-receptive-field-and-pooling.svg','感受野逐层扩大与最大池化数值梯度',b,580);
console.log(JSON.stringify({generated:3,output:out}));
