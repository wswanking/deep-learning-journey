import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const out=path.join(root,'docs/assets/05-machine-learning');
fs.mkdirSync(out,{recursive:true});
const colors={ink:'#183044',blue:'#176aa0',orange:'#ad5015',green:'#087c63',muted:'#476276'};
const esc=x=>String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const text=(x,y,value,size=22,color=colors.ink,anchor='start')=>'<text x="'+x+'" y="'+y+'" font-size="'+size+'" fill="'+color+'" text-anchor="'+anchor+'">'+esc(value)+'</text>';
const rect=(x,y,w,h,color='#eff5fa')=>'<rect x="'+x+'" y="'+y+'" width="'+w+'" height="'+h+'" rx="12" fill="'+color+'" stroke="#a8bdcd" stroke-width="2"/>';
const arrow=(x1,y1,x2,y2,color=colors.blue)=>'<path d="M'+x1+' '+y1+' L'+x2+' '+y2+'" fill="none" stroke="'+color+'" stroke-width="3" marker-end="url(#arrow)"/>';
function save(name,height,title,body){
const data='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 '+height+'" role="img" aria-labelledby="title desc"><title id="title">'+esc(title)+'</title><desc id="desc">依据本章数学定义原创绘制，全部由矢量文字、路径和矩形组成。</desc><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="context-stroke"/></marker></defs><rect width="1200" height="'+height+'" fill="white"/><g font-family="Microsoft YaHei,Noto Sans CJK SC,Arial,sans-serif">'+body+'</g></svg>\n';
fs.writeFileSync(path.join(out,name),data);console.log(name);
}
let b=text(40,44,'图05-1  先划分数据，再拟合预处理；测试集保留到方案冻结',27);
b+=rect(420,80,360,70)+text(600,125,'带标签样本 D = {(x, y)}',25,colors.ink,'middle');
for(const [x,title,note,color] of [[40,'训练集 60%','拟合均值/标准差，更新参数',colors.blue],[440,'验证集 20%','选择学习率与检查点',colors.orange],[840,'测试集 20%','方案冻结后评估一次',colors.green]]){
b+=rect(x,218,320,120)+text(x+160,258,title,25,color,'middle')+text(x+160,305,note,21,colors.ink,'middle');
b+=arrow(600,160,x+160,207,color);
}
b+=rect(40,400,320,142,'#edf5fb')+text(200,443,'仅训练集拟合变换 T',24,colors.blue,'middle');
b+=text(200,486,'μ、σ 固定后应用到三份数据',21,colors.ink,'middle')+text(200,521,'不能用验证/测试重新估计',20,colors.muted,'middle');
b+=arrow(200,348,200,390);
b+=rect(440,400,320,142,'#fff3e6')+text(600,443,'训练候选模型',24,colors.orange,'middle');
b+=text(600,486,'预测 → 损失 → 梯度 → 更新',21,colors.ink,'middle')+text(600,521,'训练集决定参数 θ',21,colors.muted,'middle');
b+=arrow(370,469,430,469);
b+=rect(840,400,320,142,'#edf7f0')+text(1000,443,'冻结选择后的模型与 T',23,colors.green,'middle');
b+=text(1000,486,'验证选方案；测试估计效果',21,colors.ink,'middle')+text(1000,521,'测试结果不再反馈调参',21,colors.muted,'middle');
b+=arrow(770,469,830,469);
b+=text(40,605,'注意：验证集不参与梯度更新，但已经参与方案选择，因此其分数也不是最终独立报告。',22);
b+=text(40,650,'随机划分适用于这里的独立同分布合成样本；用户/病人/文档分组或时间预测需采用对应划分。',21);
save('01-data-protocol.svg',700,'训练验证测试与训练集预处理',b);

b=text(40,44,'图05-2  同一线性得分，三种任务需要不同输出含义与损失',27);
for(const [y,title,score,output,loss,example,color] of [
[95,'回归','ŷ = wᵀx + b','输出：任意实数','ℓ = (ŷ − y)²','预测(0,1)、目标(1,3)：平均损失2.5',colors.blue],
[310,'二分类','z = wᵀx + b','p = sigmoid(z)','ℓ = −y log p − (1−y) log(1−p)','z=log 3 → p=0.75；y=1时损失0.2877',colors.orange],
[525,'互斥多分类','z = Wx + b','p = softmax(z)','ℓ = −log p_y','z=(log 2,0,0) → p=(0.5,0.25,0.25)',colors.green]
]){
b+=rect(40,y,1120,180,'#f4f7fa')+text(60,y+35,title,25,color);
b+=text(65,y+84,score,23)+arrow(330,y+78,392,y+78,color);
b+=text(415,y+84,output,23,color)+arrow(695,y+78,748,y+78,color);
b+=text(775,y+84,loss,21);
b+=text(65,y+143,example,22,colors.muted);
}
// Optional Gaussian interpretation of regression: an original sigma=1 density.
let bellPoints='';
for(let j=0;j<=80;j++){const value=-3+j*6/80;const px=1015+value*29;const py=242-39*Math.exp(-value*value/2);bellPoints+=(j===0?'M':'L')+px.toFixed(2)+' '+py.toFixed(2)+' ';}
b+='<line x1="925" y1="242" x2="1110" y2="242" stroke="#7e95a5" stroke-width="1.5"/>';
b+='<path d="'+bellPoints+'" fill="none" stroke="#176aa0" stroke-width="2.5"/>';
b+=text(1015,267,'正态噪声 N(0,1)',15,colors.blue,'middle');

b+=text(40,756,'log为自然对数；所有损失先按样本计算，再对B个样本平均。类别编号本身没有大小/距离意义。',21);
b+=text(40,799,'代码：MSE接预测；BCEWithLogits接原始z；CrossEntropy接(B,C)原始z和(B,)整数标签。',21);
save('02-models-and-losses.svg',845,'回归二分类多分类的得分概率损失',b);
