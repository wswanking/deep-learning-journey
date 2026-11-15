import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const output=path.join(root,'docs/assets/typesetting');
fs.mkdirSync(output,{recursive:true});
const xml=value=>String(value).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;');
function text(x,y,value,size=20,color='#17243b',anchor='middle'){
  return '<text x="'+x+'" y="'+y+'" text-anchor="'+anchor+'" font-size="'+size+'" fill="'+color+'">'+xml(value)+'</text>';
}
function svg(title,body,height){
 return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 '+height+'" role="img">' +
 '<title>'+xml(title)+'</title><g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif">' + body + '</g></svg>\n';
}
function save(name,body){fs.writeFileSync(path.join(output,name+'.svg'),body);console.log('SVG='+name);}
let flow='<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#2456a6"/></marker></defs>';
const labels=[['输入向量','x：2 个分量'],['仿射变换','z = W x + b'],['输出向量','z：3 个分量']];
for(let i=0;i<3;i++){
 const x=30+i*305;
 flow+='<rect x="'+x+'" y="50" width="230" height="110" rx="10" fill="'+(i===1?'#eaf1fc':'#f4f7fb')+'" stroke="#a9bdd8" stroke-width="2"/>';
 flow+=text(x+115,92,labels[i][0],24)+text(x+115,131,labels[i][1],19,'#526277');
 if(i<2)flow+='<path d="M'+(x+242)+' 105H'+(x+290)+'" stroke="#2456a6" stroke-width="3" marker-end="url(#arrow)"/>';
}
flow+=text(450,208,'维度：W 为 3 × 2，b 为 3 × 1，列向量约定',18,'#526277');
save('linear-flow',svg('仿射变换的数据与形状',flow,245));

const left=90,top=35,width=720,height=230;
let curve='';
for(let i=0;i<=4;i++){
 const px=left+width*i/4,py=top+height-height*i/4;
 curve+='<path d="M'+px+' '+top+'V'+(top+height)+'M'+left+' '+py+'H'+(left+width)+'" stroke="#e1e8f0"/>';
 curve+=text(px,top+height+29,i,17,'#526277')+text(left-20,py+6,i,17,'#526277','end');
}
curve+='<path d="M'+left+' '+top+'V'+(top+height)+'H'+(left+width)+'" fill="none" stroke="#526277" stroke-width="2"/>';
let d='';
for(let i=0;i<=100;i++){
 const w=4*i/100,y=(w-2)**2;
 d+=(i?' L':'M')+(left+width*w/4).toFixed(2)+' '+(top+height-height*y/4).toFixed(2);
}
curve+='<path d="'+d+'" fill="none" stroke="#2456a6" stroke-width="4"/>';
curve+='<circle cx="450" cy="265" r="6" fill="#d78225"/>';
curve+=text(450,326,'参数 w',19)+text(91,22,'损失 L',18,'#526277','start')+text(660,65,'解析函数 L = (w − 2)²',20,'#2456a6');
save('loss-curve',svg('解析损失曲线，并非训练实测',curve,350));

const values=[[0.70,0.20,0.10],[0.15,0.65,0.20],[0.10,0.25,0.65]];
let heat=text(340,31,'归一化权重矩阵',22);
for(let j=0;j<3;j++)heat+=text(190+j*140,64,'键 '+(j+1),18,'#526277');
for(let i=0;i<3;i++){
 heat+=text(100,117+i*70,'查询 '+(i+1),18,'#526277','end');
 for(let j=0;j<3;j++){
  const value=values[i][j],x=125+j*140,y=78+i*70;
  const colors=['#edf3fc','#dee9f9','#c9dcf4','#a2c3ec','#719ed7','#4575b6','#2456a6'];
  const color=colors[Math.min(6,Math.floor(value*9))];
  heat+='<rect x="'+x+'" y="'+y+'" width="130" height="60" rx="5" fill="'+color+'" stroke="#fff" stroke-width="2"/>';
  heat+=text(x+65,y+38,value.toFixed(2),22,value>.5?'#fff':'#17243b');
 }
}
heat+=text(680,110,'每行权重之和为 1',20)+text(680,149,'矩形与数字均为矢量',18,'#526277');
heat+=text(450,314,'示意数据：仅检验热图排版，不解释为模型实测注意力。',17,'#526277');
save('attention-heatmap',svg('带数字标注的矢量热图',heat,340));
