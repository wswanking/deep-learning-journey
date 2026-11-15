/** Original vector figures for chapter 03; no raster assets or external libraries. */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const output = path.join(root, 'docs', 'assets', '03-probability');
fs.mkdirSync(output, { recursive: true });
const ink = '#172c44', muted = '#425b74', blue = '#246cc0', orange = '#bd611a';
const text = (x,y,value,size=22,attributes='') => `<text x="${x}" y="${y}" font-size="${size}" ${attributes}>${value}</text>`;
const line = (x1,y1,x2,y2,color='#a5b4c5',width=2,attributes='') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}" ${attributes}/>`;
const rect = (x,y,w,h,fill,attributes='') => `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${fill}" ${attributes}/>`;
const polyline = (points,color,width=3,attributes='') => `<path d="${points.map((p,i)=>`${i?'L':'M'} ${p[0].toFixed(2)} ${p[1].toFixed(2)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="${width}" ${attributes}/>`;
function save(name,height,title,description,body) {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="900" height="${height}" viewBox="0 0 900 ${height}" role="img" aria-labelledby="c03-${name}-title c03-${name}-desc"><title id="c03-${name}-title">${title}</title><desc id="c03-${name}-desc">${description}</desc>${rect(0,0,900,height,'#f7f9fc','rx="16"')}<g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" fill="${ink}">${body}</g></svg>\n`;
  fs.writeFileSync(path.join(output, `${name}.svg`),svg,'utf8');
}

// A joint PMF with row sums P(X) and column sums P(Y); highlight conditioning column.
{
  const columns=[30,160,285,410,535], rows=[105,170,240,310,380];
  let body=text(30,42,'条件概率：保留 Y=1 的情形，再归一化',28,'font-weight="700"');
  body+=text(30,77,'联合表中的四格相加为1；蓝色列的总概率为0.60。',21,`fill="${muted}"`);
  body+=rect(30,105,505,275,'#fff');
  body+=rect(285,105,125,275,'#e0edff');
  body+=rect(30,105,505,65,'#e8eef6');
  body+=rect(285,105,125,65,'#d0e2fb');
  body+=rect(410,170,125,210,'#edf1f6');
  body+=rect(30,310,380,70,'#edf1f6');
  body+=rect(285,310,125,70,'#d0e2fb');
  for(const x of columns) body+=line(x,105,x,380);
  for(const y of rows) body+=line(30,y,535,y);
  const cells=[['X / Y','Y=0','Y=1','P(X)'],['X=0','0.10','0.20','0.30'],['X=1','0.30','0.40','0.70'],['P(Y)','0.40','0.60','1.00']];
  const centers=[95,222.5,347.5,472.5], baselines=[145,214,284,354];
  cells.forEach((row,r)=>row.forEach((cell,c)=>body+=text(centers[c],baselines[r],cell,23,'text-anchor="middle"')));
  body+=rect(590,105,280,275,'#fff','rx="12" stroke="#246cc0" stroke-width="2"');
  body+=text(610,143,'固定 Y=1',24,'font-weight="700"');
  body+=text(610,187,'P(X=0 | Y=1)',22);
  body+=text(610,219,'= 0.20 / 0.60 = 1/3',21,`fill="${blue}"`);
  body+=text(610,274,'P(X=1 | Y=1)',22);
  body+=text(610,306,'= 0.40 / 0.60 = 2/3',21,`fill="${blue}"`);
  body+=text(610,352,'条件列：1/3 + 2/3 = 1',21);
  body+=text(30,427,'联合概率问“同时发生”；条件概率问“已知条件下的比例”。',22);
  body+=text(30,468,'P(X=1, Y=1) = 0.40',23);
  body+=text(30,510,'P(X=1 | Y=1) = 2/3；P(Y=1 | X=1) = 4/7',22,`fill="${muted}"`);
  save('joint-conditioning',545,'图03-1：联合概率表与条件概率归一化','联合分布行表示X，列表示Y。条件Y等于1时，保留概率为0.20和0.40的一列，除以0.60，得到三分之一和三分之二。',body);
}

// Discrete distributions with the same mean and different variances.
{
  let body=text(30,42,'期望相同，波动程度仍可以不同',28,'font-weight="700"');
  body+=text(30,77,'柱高为概率；虚线标出均值 μ=1，横轴是随机变量的取值。',21,`fill="${muted}"`);
  const panels=[{left:75,p:[.25,.5,.25],color:blue,label:'分布A：均值1，方差0.5'}, {left:505,p:[.5,0,.5],color:orange,label:'分布B：均值1，方差1'}];
  for(const {left,p,color,label} of panels){
    const base=400, height=240, centers=[left+45,left+150,left+255];
    body+=text(left-25,122,label,23,'font-weight="700"');
    for(const tick of [0,.25,.5]){
      const y=base-tick/.5*height;
      body+=line(left-5,y,left+300,y,'#d1dbe6',1);
      body+=text(left-15,y+7,tick.toString(),20,'text-anchor="end"');
    }
    body+=line(left-5,150,left-5,400,ink,2);
    body+=line(left-5,400,left+300,400,ink,2);
    body+=line(centers[1],147,centers[1],400,color,2,'stroke-dasharray="6 6"');
    p.forEach((prob,c)=>{
      const h=prob/.5*height;
      if(h)body+=rect(centers[c]-30,base-h,60,h,color,'fill-opacity="0.82"');
      body+=text(centers[c],base-h-12,prob.toFixed(2),21,'text-anchor="middle"');
      body+=text(centers[c],433,String(c),22,'text-anchor="middle"');
    });
    body+=text(left+293,463,'x',23,'text-anchor="end"');
  }
  body+=text(30,505,'A：0×0.25 + 1×0.50 + 2×0.25 = 1',22);
  body+=text(30,544,'B：0×0.50 + 1×0.00 + 2×0.50 = 1',22);
  body+=text(30,584,'方差按距均值的平方加权；B把更多概率放在两侧。',22,`fill="${muted}"`);
  save('distribution-moments',615,'图03-2：离散分布的期望与方差','两个分布都在0、1、2取值，均值为1。A的概率为0.25、0.50、0.25，方差0.5。B的概率为0.50、0、0.50，方差1。',body);
}

// Fixed-parameter probabilities and fixed-data likelihood, with different vertical scales.
{
  let body=text(30,42,'概率与似然：先说清楚什么固定、什么在变化',28,'font-weight="700"');
  body+=text(45,97,'概率：固定 θ=0.75',24,'font-weight="700"');
  body+=text(45,130,'比较下一次结果 X=0 与 X=1',21,`fill="${muted}"`);
  body+=text(475,97,'似然：固定 D=(1,0,1,1)',24,'font-weight="700"');
  body+=text(475,130,'比较不同 θ：L(θ)=θ³(1−θ)',21,`fill="${muted}"`);
  const lx=85,ly=420,lw=285,lh=245;
  for(const tick of [0,.25,.5,.75,1]){
    const y=ly-tick*lh;
    body+=line(lx,y,lx+lw,y,'#d1dbe6',1);
    body+=text(lx-10,y+6,tick.toFixed(2),20,'text-anchor="end"');
  }
  body+=line(lx,165,lx,ly,ink);body+=line(lx,ly,lx+lw,ly,ink);
  for(const [x,p,label] of [[155,.25,'X=0'],[300,.75,'X=1']]){
    body+=rect(x-34,ly-p*lh,68,p*lh,blue,'fill-opacity="0.82"');
    body+=text(x,ly-p*lh-12,p.toFixed(2),21,'text-anchor="middle"');
    body+=text(x,452,label,22,'text-anchor="middle"');
  }
  const rx=525,ry=420,rw=315,rh=245,max=.12;
  for(const tick of [0,.04,.08,.12]){
    const y=ry-tick/max*rh;
    body+=line(rx,y,rx+rw,y,'#d1dbe6',1);
    body+=text(rx-10,y+6,tick.toFixed(2),20,'text-anchor="end"');
  }
  body+=line(rx,165,rx,ry,ink);body+=line(rx,ry,rx+rw,ry,ink);
  const points=Array.from({length:201},(_,i)=>{const theta=i/200;return [rx+theta*rw,ry-theta**3*(1-theta)/max*rh];});
  body+=polyline(points,orange,4);
  for(const tick of [0,.25,.5,.75,1])body+=text(rx+tick*rw,452,tick.toString(),20,'text-anchor="middle"');
  body+=text(rx+rw,486,'θ',23,'text-anchor="end"');
  const px=rx+.75*rw,py=ry-.10546875/max*rh;
  body+=`<circle cx="${px}" cy="${py}" r="5" fill="${orange}"/>`;
  body+=text(px-15,py-16,'最大值在 θ=0.75',21,'text-anchor="end"');
  body+=text(30,530,'左图概率相加为1；右图是固定数据对参数的评分函数。',22);
  body+=text(30,571,'右图未对 θ 归一化，不能直接当成参数的后验分布。',22,`fill="${muted}"`);
  save('probability-likelihood',605,'图03-3：概率与似然的自变量区别','左图在Bernoulli参数0.75固定时比较结果0与1的概率。右图固定有序观测1、0、1、1，横轴变化参数theta，似然theta立方乘一减theta在0.75处最大。两图纵轴尺度不同并分别标注。',body);
}

// Analytic CE and KL curves, not a sampled or trained model result.
{
  const p0=.25,p1=.75,H=-p0*Math.log(p0)-p1*Math.log(p1);
  const left=90,base=430,width=720,height=310,ymax=3.6;
  const x=r=>left+r*width,y=v=>base-v/ymax*height;
  let body=text(30,42,'固定目标分布 p：预测分布改变，交叉熵怎样变化',28,'font-weight="700"');
  body+=text(30,78,'状态顺序为(0,1)：p=(0.25,0.75)，q=(1−r,r)，自然对数 / nat',21,`fill="${muted}"`);
  for(const tick of [0,1,2,3]){
    body+=line(left,y(tick),left+width,y(tick),'#d1dbe6',1);
    body+=text(left-15,y(tick)+7,String(tick),21,'text-anchor="end"');
  }
  body+=line(left,110,left,base,ink);body+=line(left,base,left+width,base,ink);
  for(const tick of [0,.25,.5,.75,1]){
    body+=line(x(tick),base,x(tick),base+6,ink);
    body+=text(x(tick),base+33,tick.toString(),21,'text-anchor="middle"');
  }
  body+=text(35,108,'nat',21);body+=text(left+width,490,'r = q₁',22,'text-anchor="end"');
  const rs=Array.from({length:197},(_,i)=>.01+i*.005);
  const ce=r=>-p0*Math.log1p(-r)-p1*Math.log(r);
  body+=line(left,y(H),left+width,y(H),'#537c5d',2,'stroke-dasharray="7 6"');
  body+=polyline(rs.map(r=>[x(r),y(ce(r))]),blue,4);
  body+=polyline(rs.map(r=>[x(r),y(ce(r)-H)]),orange,3,'stroke-dasharray="10 5"');
  body+=line(x(.75),120,x(.75),base,'#8b9fb5',2,'stroke-dasharray="5 5"');
  body+=`<circle cx="${x(.75)}" cy="${y(H)}" r="5" fill="${blue}"/><circle cx="${x(.75)}" cy="${base}" r="5" fill="${orange}"/>`;
  body+=text(x(.75)-15,y(H)-16,'q=p：H(p,q)=0.5623',21,'text-anchor="end"');
  body+=line(35,525,80,525,blue,4)+text(95,533,'实线：交叉熵 H(p,q)',22);
  body+=line(445,525,490,525,orange,3,'stroke-dasharray="10 5"')+text(505,533,'虚线：KL(p ∥ q)',22);
  body+=text(30,574,'绿色水平线：H(p)=0.5623；交叉熵比它高出的部分等于 KL。',22);
  body+=text(30,616,'曲线仅绘 0.01≤r≤0.99；r=0 或 1 时，这组 p 的交叉熵为 +∞。',21,`fill="${muted}"`);
  save('cross-entropy-kl',650,'图03-4：交叉熵与KL的解析曲线','固定目标分布0.25和0.75，预测成功概率r从0.01变到0.99。交叉熵在r等于0.75时最小，等于目标熵0.5623 nat；KL为交叉熵减去目标熵并在同一点为零。端点0和1存在支持集缺失，量为无穷。',body);
}
console.log(`Generated four chapter 03 SVG figures in ${output}`);
