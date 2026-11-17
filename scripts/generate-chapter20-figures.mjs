import {fileURLToPath} from 'node:url';
import fs from 'node:fs';import path from 'node:path';
const out=path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../docs/assets/20-language-model-pretraining');fs.mkdirSync(out,{recursive:true});
const C={ink:'#24374c',blue:'#1475b8',light:'#eaf4fc',orange:'#bc570e',warm:'#fff2e6',line:'#afc4d6'};
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;');
const t=(x,y,s,z=18,c=C.ink)=>'<text x="'+x+'" y="'+y+'" font-size="'+z+'" fill="'+c+'">'+esc(s)+'</text>';
const box=(x,y,w,h,c=C.light)=>'<rect x="'+x+'" y="'+y+'" width="'+w+'" height="'+h+'" rx="5" fill="'+c+'" stroke="'+C.line+'"/>';
const ar=(x,y,a,b)=>'<path d="M'+x+','+y+'L'+a+','+b+'" fill="none" stroke="'+C.blue+'" stroke-width="2" marker-end="url(#a)"/>';
function save(file,title,b,h=580){fs.writeFileSync(path.join(out,file),'<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="'+h+'" viewBox="0 0 1080 '+h+'" role="img"><title>'+title+'</title><desc>原创定义/数值图，依据花书12.4.1、D2L9.3及PyTorch2.15；数据管线为本章原创固定协议。</desc><defs><marker id="a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0L10,5L0,10Z" fill="'+C.blue+'"/></marker></defs><rect width="1080" height="'+h+'" fill="white"/><g font-family="Microsoft YaHei,Noto Sans CJK SC,sans-serif">'+b+'</g></svg>','utf8');}
let b=t(25,34,'图20-1  给定前缀A，逐个预测B、C、D；输入与标签右移一格',22);
for(let i=0;i<4;i++){b+=box(50+190*i,90,100,58)+t(88+190*i,127,'ABCD'[i],25);if(i<3)b+=ar(160+190*i,119,230+190*i,119)+t(166+190*i,85,['0.5','0.25','0.125'][i],18,C.blue);}
b+=t(835,119,'乘积 1/64',20)+t(835,151,'平均NLL = log4',17);
b+=t(45,220,'位置：',18);for(let i=0;i<3;i++)b+=t(187+125*i,220,String(i),18);
b+=t(45,265,'输入X',18)+t(45,340,'标签Y',18);
for(let i=0;i<3;i++){b+=box(150+125*i,235,90,48)+t(186+125*i,267,'ABC'[i],22);b+=box(150+125*i,308,90,48,C.warm)+t(186+125*i,340,'BCD'[i],22,C.orange);b+=ar(195+125*i,287,195+125*i,302);}
b+=t(45,400,'位置i可见当前及更早输入，预测位置i+1的标签。',18);
b+=t(620,220,'因果可见矩阵：行是查询i，列是键j',17);
for(let i=0;i<3;i++)for(let j=0;j<3;j++){b+=box(705+48*j,245+48*i,44,44,j<=i?C.light:'#edf0f4');b+=t(719+48*j,274+48*i,j<=i?'✓':'×',19,j<=i?C.blue:'#8c98a4');}
for(let i=0;i<3;i++)b+=t(719+48*i,236,String(i),14)+t(680,274+48*i,String(i),14);
b+=t(875,280,'蓝色：j≤i',17,C.blue)+t(875,320,'灰色：未来',17);
b+=box(45,450,985,88,C.warm)+t(65,482,'金标准历史可并行提供，但每个位置必须看不到未来；因果mask与标签移位都要正确。',18);
b+=t(65,516,'输入A却监督A，会把任务变成复制当前输入；本章用实际反例检查这一错误。',18);
save('01-autoregressive-shift.svg','自回归条件概率、移位标签与因果可见矩阵',b);
b=t(25,34,'图20-3  文档先清理、去重和分组划分，再生成训练窗口',22);
const xs=[45,410,775];const titles=['原始40条JSONL','清理后36条文档','12个family组'];
for(let i=0;i<3;i++)b+=box(xs[i],90,260,95)+t(xs[i]+20,124,titles[i],21);
b+=t(65,158,'含2重复、1空白、1过短',17)+t(430,158,'固定规则 + SHA256去重',17)+t(795,158,'同主题3个变体同组',17);
b+=ar(315,137,397,137)+ar(680,137,762,137);
b+='<path d="M905,190V224H195M540,224V250M885,224V250M195,224V250" stroke="'+C.blue+'" stroke-width="2" fill="none"/>';
for(const [i,label,detail] of [[0,'训练：8组 / 24文档','拟合计数/参数'],[1,'验证：2组 / 6文档','只选择α等配置'],[2,'测试：2组 / 6文档','冻结后仅作评估']]){b+=box(60+i*345,255,270,95)+t(80+i*345,291,label,20)+t(80+i*345,325,detail,18);}
b+=t(50,405,'组内模板变体不跨集合；精确重复与group_id冲突时拒绝，先核对来源身份。',18);
b+=box(45,440,990,90,C.warm)+t(65,472,'文档ABCD，窗口长2：[A B]→[B C]；[C]→[D]。标签B/C/D各出现一次。',18);
b+=t(65,508,'文档边界不生成相邻对；右padding输入0，但mask=False、目标=-100。',18);
b+=t(50,572,'仅按原始行先切分，本例会留下2个跨训练/测试重复哈希；不能用更多训练步修复。',17,C.orange);
save('03-corpus-pipeline.svg','原创语料去重分组划分和文档内窗口协议',b,610);
b=t(25,34,'图20-2  困惑度先汇总有效token损失，再指数；不要平均批次或padding',22);
b+=box(45,90,455,100)+t(65,125,'批次A：1个有效token，平均NLL=log2',18)+t(65,165,'批次B：3个有效token，平均NLL=log8',18);
b+=ar(515,140,585,140)+box(605,90,425,100)+t(625,125,'总NLL = log2 + 3log8',19,C.blue)+t(625,165,'总有效数 = 4；PPL = √32 ≈ 5.657',18,C.blue);
b+=t(65,240,'错误①：平均两批NLL，再指数 → 4。',19,C.orange)+t(65,278,'错误②：平均两批PPL：(2+8)/2 → 5。',19,C.orange);
b+=t(45,340,'另一个padding例：真实目标概率0.5、0.25，后面两格只是填充。',19);
for(let i=0;i<4;i++){b+=box(65+i*210,370,180,58,i<2?C.light:'#edf0f4')+t(88+i*210,405,['目标1 / 0.5','目标2 / 0.25','PAD / 不计','PAD / 不计'][i],18);}
b+=t(65,475,'正确分母2：PPL=√8≈2.828；错误分母4：PPL=⁴√8≈1.682。',19);
b+=t(65,525,'分母错了会制造更低数字；它没有提高模型预测能力。',18,C.orange);
save('02-perplexity-reduction.svg','有效token归约和困惑度的精确反例',b,570);
console.log(JSON.stringify({generated:3,output:out}));
