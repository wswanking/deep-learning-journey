import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const out = path.join(root, 'docs/assets/01-linear-algebra');
fs.mkdirSync(out, { recursive: true });
const c = { ink: '#172d44', muted: '#40566b', blue: '#156da3', teal: '#087d77', orange: '#ad540f', light: '#edf5fb', border: '#acbecb', grid: '#cfdae3' };
const esc = x => String(x).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
const text = (x, y, label, size = 22, fill = c.ink, anchor = 'start') => `<text x="${x}" y="${y}" font-size="${size}" fill="${fill}" text-anchor="${anchor}">${esc(label)}</text>`;
const line = (x1, y1, x2, y2, color = c.border, width = 2, dash = '') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}"${dash ? ` stroke-dasharray="${dash}"` : ''}/>`;
const rect = (x, y, w, h, fill = '#fff', border = c.border, radius = 12) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${radius}" fill="${fill}" stroke="${border}" stroke-width="2"/>`;
const arrow = (x1, y1, x2, y2, color = c.blue) => `<path d="M ${x1} ${y1} L ${x2} ${y2}" fill="none" stroke="${color}" stroke-width="3" marker-end="url(#arrow)"/>`;
const matrix = (x, y, values, cw = 60, ch = 46, colorCells = {}) => values.map((row, i) => row.map((v, j) => rect(x+j*cw, y+i*ch, cw, ch, colorCells[`${i},${j}`] || '#fff', c.border, 0) + text(x+(j+.5)*cw, y+(i+.66)*ch, v, 22, c.ink, 'middle')).join('')).join('');
function save(name, title, desc, height, body) {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 ${height}" role="img" aria-labelledby="title desc">
<title id="title">${esc(title)}</title><desc id="desc">${esc(desc)}</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="context-stroke"/></marker></defs>
<rect width="1200" height="${height}" fill="#fff"/>
<g font-family="Microsoft YaHei, Noto Sans CJK SC, Arial, sans-serif">${body}</g></svg>\n`;
  fs.writeFileSync(path.join(out, name), svg);
  console.log(name);
}
let b = text(40, 46, '图01-1  数据对象、形状与轴：先说明每条轴装的是什么', 28);
const boxes = [
  [40, '标量', 's = 2', 'shape = ()', '0条轴，1个元素'],
  [330, '一个样本的特征向量', '[2, 1, 0]', 'shape = (3,)', '1条轴：特征'],
  [620, '一批样本', '2行 × 3列', 'shape = (2, 3)', '轴0：样本；轴1：特征'],
  [910, '一批序列', '2 × 3 × 4', 'shape = (2, 3, 4)', '样本、时间、特征']
];
for (const [x, title, value, shape, note] of boxes) {
  b += rect(x, 78, 250, 210, c.light) + text(x+15, 113, title, 22, c.blue);
  b += text(x+125, 169, value, 27, c.ink, 'middle') + text(x+125, 216, shape, 21, c.ink, 'middle');
  b += text(x+125, 259, note, 17, c.muted, 'middle');
}
b += text(40, 337, '同一批次矩阵：行是样本，列是特征', 24);
b += matrix(100, 365, [[2,1,0],[0,-1,2]], 85, 55, {'0,0':'#e6f4ee','0,1':'#e6f4ee','0,2':'#e6f4ee'});
b += text(60, 402, 'i=0', 18) + text(60, 457, 'i=1', 18);
b += text(142, 505, 'j=0', 18, c.muted,'middle') + text(227, 505, 'j=1', 18,c.muted,'middle') + text(312, 505, 'j=2',18,c.muted,'middle');
b += text(415, 397, 'X[0, :] = [2, 1, 0]', 22) + text(415, 441, 'X[:, 1] = [1, -1]',22);
b += text(415, 489, '索引从0开始；数学下标通常从1开始。',20,c.muted);
b += line(40, 537, 1160, 537);
b += text(40, 578, '列向量约定 ≠ 一维数组天然具有列轴',24,c.orange);
b += text(75, 627, '数学：x ∈ ℝ³，默认列向量',22);
b += matrix(100, 652, [[2],[1],[0]], 58, 40);
b += text(200, 719, '显式列矩阵：(3, 1)',22);
b += text(650, 627, '代码：x = tensor([2., 1., 0.])',22);
b += text(650, 679, 'x.shape = (3,)：没有行/列方向',21);
b += text(650, 729, 'x[:, None] → (3, 1)',22,c.blue);
b += text(650, 772, 'x[None, :] → (1, 3)',22,c.teal);
save('01-objects-and-axes.svg','数据对象、形状与轴','从标量到三阶张量；数学列向量与代码一维数组的对应。',815,b);

b = text(40, 46, '图01-2  矩阵乘法：内侧维度求和，外侧维度留下',28);
b += text(50, 96,'A：2 × 3',24,c.blue) + text(470,96,'B：3 × 2',24,c.teal) + text(850,96,'C = AB：2 × 2',24,c.orange);
b += matrix(50,125,[[1,2,3],[4,5,6]],70,52,{'0,0':'#e3f0fc','0,1':'#e3f0fc','0,2':'#e3f0fc'});
b += matrix(470,125,[[1,2],[0,1],[-1,0]],70,52,{'0,1':'#e1f2ed','1,1':'#e1f2ed','2,1':'#e1f2ed'});
b += matrix(850,125,[[-2,4],[-2,13]],70,52,{'0,1':'#fff0dd'});
b += text(350,187,'×',34,c.ink,'middle') + text(735,187,'=',34,c.ink,'middle');
b += text(50,332,'例如 c₁₂ = 第1行 · 第2列 = 1×2 + 2×1 + 3×0 = 4',24);
b += text(50,379,'形状：(m, k) @ (k, n) → (m, n)；k轴相乘求和，不是逐元素乘。',23,c.muted);
b += line(40,413,1160,413);
b += text(40,458,'行批次线性层：每行一个样本，使用同一组权重',26);
b += rect(50,497,270,144,c.light) + text(185,540,'X：(B, d) = (2, 3)',24,c.blue,'middle');
b += text(185,587,'样本 × 输入特征',22,c.ink,'middle') + text(185,620,'两条样本分别计算',19,c.muted,'middle');
b += rect(440,497,270,144,'#edf7f2') + text(575,540,'Wᵀ：(d, m) = (3, 2)',23,c.teal,'middle');
b += text(575,587,'输入特征 × 输出特征',21,c.ink,'middle') + text(575,620,'W本身存成 (m, d)',19,c.muted,'middle');
b += rect(830,497,320,144,'#fff5e8') + text(990,540,'Z：(B, m) = (2, 2)',24,c.orange,'middle');
b += text(990,587,'样本 × 输出特征',22,c.ink,'middle') + text(990,620,'批次轴B保持不变',19,c.muted,'middle');
b += arrow(330,565,430,565) + text(380,539,'@',24,c.ink,'middle');
b += arrow(720,565,820,565) + text(770,539,'+ b',23,c.ink,'middle');
b += rect(515,685,250,72,'#fff') + text(640,729,'b：(m,) = (2,)',24,c.orange,'middle');
b += arrow(775,720,991,652,c.orange);
b += text(50,786,'Z = X Wᵀ + 1_B bᵀ；代码为 X @ W.T + b，偏置沿样本轴共享。',23);
save('02-matmul-and-batch.svg','行乘列与行批次线性层','高亮一行与一列计算一个输出元素；批次输入、转置权重和共享偏置的形状。',825,b);

b = text(40,46,'图01-4  线性变换改变坐标，范数定义“有多大”',28);
function plotGrid(cx, cy, s, transform, transformed = false) {
  let part = '';
  const p = ([x,y]) => [cx+s*x, cy-s*y];
  for(let k=-2;k<=2;k++) {
    let a=p(transform([-2,k])), z=p(transform([2,k]));
    part += line(...a,...z,c.grid,1.4);
    a=p(transform([k,-2]));z=p(transform([k,2]));
    part += line(...a,...z,c.grid,1.4);
  }
  part += arrow(cx-120,cy,cx+200,cy,c.muted) + arrow(cx,cy+110,cx,cy-175,c.muted);
  for(const [v,label,color,dx,dy] of [[[1,0],'e₁',c.blue,8,21],[[0,1],'e₂',c.teal,8,-8],[[1,2],'x',c.orange,10,-10]]) {
    const end=p(transform(v));part += arrow(cx,cy,...end,color)+text(end[0]+dx,end[1]+dy,transformed ? `A${label}` : label,21,color);
  }
  part += text(cx+205,cy+21,'横坐标',17,c.muted) + text(cx+18,cy-167,'纵坐标',17,c.muted);
  return part;
}
b += text(120,94,'原空间',24,c.blue)+text(705,94,'A = [[2, 1], [0, 1]]',24,c.teal);
b += plotGrid(235,285,65,x=>x);
b += plotGrid(800,285,48,([x,y])=>[2*x+y,y],true);
b += text(72,445,'e₁=(1,0)，e₂=(0,1)，x=(1,2)',21);
b += text(620,445,'Ae₁=(2,0)，Ae₂=(1,1)，Ax=(4,2)',21);
b += text(80,487,'Ax = 1·Ae₁ + 2·Ae₂：新位置是两列的加权和；原点仍映射到原点。',23);
b += line(40,519,1160,519);
b += text(40,560,'单位范数轮廓：边界上每一点的对应范数都等于1',24);
for(const [cx,title,desc,type] of [[210,'1范数','|x₁| + |x₂| = 1','diamond'],[600,'2范数','√(x₁² + x₂²) = 1','circle'],[990,'无穷范数','max(|x₁|, |x₂|) = 1','square']]) {
  const cy=725,r=84;
  b += line(cx-115,cy,cx+115,cy,c.muted,1.5)+line(cx,cy+105,cx,cy-110,c.muted,1.5);
  if(type==='diamond') b+=`<path d="M${cx-r} ${cy} L${cx} ${cy-r} L${cx+r} ${cy} L${cx} ${cy+r} Z" fill="#e5f1fa" stroke="${c.blue}" stroke-width="3"/>`;
  if(type==='circle') b+=`<circle cx="${cx}" cy="${cy}" r="${r}" fill="#e9f5ef" stroke="${c.teal}" stroke-width="3"/>`;
  if(type==='square') b+=rect(cx-r,cy-r,2*r,2*r,'#fff2e0',c.orange,0);
  b+=text(cx,608,title,23,c.ink,'middle') + text(cx+r,cy+25,'1',17,c.muted,'middle')+text(cx+17,cy-r+6,'1',17,c.muted);
  b+=text(cx,854,desc,21,c.ink,'middle');
}
save('04-transform-and-norms.svg','二维线性变换与范数的几何意义','矩阵通过基向量变换决定平面网格；三种向量范数的单位轮廓。',895,b);

b = text(40,46,'图01-3  广播从右向左对齐；形状合法还要检查语义',28);
b += text(40,96,'正确共享偏置：每行都加同一组输出特征偏置',24,c.blue);
b += matrix(60,127,[[4,-1],[-4,7]],80,49);
b += text(267,185,'+',30);
b += matrix(335,127,[[.5,-1]],80,49);
b += text(335,224,'(2,) → (1, 2)',19,c.muted);
b += text(555,185,'=',30);
b += matrix(635,127,[[4.5,-2],[-3.5,6]],80,49,{'0,0':'#e3f0fc','0,1':'#e3f0fc','1,0':'#e1f2ed','1,1':'#e1f2ed'});
b += text(60,274,'形状：(2, 2) + (2,) → (2, 2)',23);
b += text(635,274,'概念上扩展偏置，不必复制存储。',21,c.muted);
b += line(40,309,1160,309);
b += text(40,355,'静默错误：pred 的样本轴 与 target 的样本轴对齐到了不同位置',24,c.orange);
b += text(60,402,'pred：(3, 1)',23,c.blue) + text(390,402,'target：(3,)',23,c.teal);
b += matrix(60,426,[[1],[2],[3]],65,48);
b += text(224,509,'−',30);
b += matrix(390,426,[[1,2,3]],65,48);
b += text(390,512,'缺少左侧轴 → (1, 3)',21,c.muted);
b += text(680,509,'=',30);
b += matrix(785,426,[[0,-1,-2],[1,0,-1],[2,1,0]],65,48);
b += text(785,610,'输出：(3, 3)，不是 (3, 1)',22,c.orange);
b += text(60,650,'第i行第j列 = pred[i, 0] − target[j]：计算了所有样本两两差。',23);
b += rect(40,684,1120,141,c.light);
b += text(65,722,'修正：target[:, None] → (3, 1)，同一个样本只和自己的目标相减。',23,c.blue);
b += text(65,763,'错误平均平方差 = 12 / 9 = 4/3；正确平均平方差 = 0。',23);
b += text(65,803,'判定顺序：说明轴含义 → 手算输出形状 → 运算 → 同时检查数值与形状。',21,c.muted);
save('03-broadcasting.svg','偏置广播与样本目标静默错位','正确偏置共享和一个形状合法但样本语义错误的两两相减。',860,b);
