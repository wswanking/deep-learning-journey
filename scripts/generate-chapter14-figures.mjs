import fs from 'node:fs';
import path from 'node:path';
const out=path.resolve('docs/assets/14-tokenization-and-embeddings');fs.mkdirSync(out,{recursive:true});
const C={ink:'#24374c',blue:'#1475b8',light:'#eaf4fc',warm:'#fff2e6',orange:'#bc570e',line:'#afc4d6'};
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;');
const t=(x,y,s,z=18,c=C.ink)=>'<text x="'+x+'" y="'+y+'" font-size="'+z+'" fill="'+c+'">'+esc(s)+'</text>';
const box=(x,y,w,h,c=C.light)=>'<rect x="'+x+'" y="'+y+'" width="'+w+'" height="'+h+'" rx="5" fill="'+c+'" stroke="'+C.line+'"/>';
const arrow=(x,y,xx,yy)=>'<path d="M'+x+','+y+'L'+xx+','+yy+'" stroke="'+C.blue+'" stroke-width="2" fill="none" marker-end="url(#a)"/>';
function save(file,title,b,h=580){fs.writeFileSync(path.join(out,file),'<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="'+h+'" viewBox="0 0 1080 '+h+'" role="img"><title>'+title+'</title><desc>原创示意；嵌入依据花书12.4/15.4和PyTorch2.15，BPE依据Sennrich等2016第3.2节，Unicode依据UAX15/29。</desc><defs><marker id="a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0L10,5L0,10Z" fill="'+C.blue+'"/></marker></defs><rect width="1080" height="'+h+'" fill="white"/><g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif">'+b+'</g></svg>','utf8');}
let b=t(25,34,'图14-1  文本先离散编码，再按ID查嵌入表；ID本身没有距离含义',22);
b+=box(40,90,240,80)+t(60,120,'文本：猫 狗 猫',21)+t(60,152,'按空格分成3个token',17);
b+=arrow(290,130,355,130)+box(370,90,235,80)+t(390,120,'token ID：[2,3,2]',21)+t(390,152,'整数索引，不是连续特征',17);
b+=arrow(615,130,680,130)+box(695,80,335,110)+t(715,115,'查表：[1,2]、[0,1]、[1,2]',20)+t(715,150,'同一ID 2查到相同向量',18);
b+=t(45,245,'E的行与token ID一一对应',19);
const rows=[['0 / PAD','0','0'],['1 / UNK','−1','0'],['2 / 猫','1','2'],['3 / 狗','0','1'],['4 / 梨','−1','2']];
for(let i=0;i<rows.length;i++){
 b+=box(45,270+42*i,320,40,i===2?C.light:'white');
 b+=t(60,296+42*i,rows[i][0],17)+t(250,296+42*i,rows[i][1],17)+t(315,296+42*i,rows[i][2],17);
}
b+=t(415,280,'列向量公式：e = Eᵀ one_hot(ID)',20,C.blue);
b+=t(415,320,'ID 2出现两次：反向贡献汇总到E的同一行。',18);
b+=t(415,360,'padding_idx=0屏蔽该行查表梯度；',18)+t(415,394,'它不会自动替平均池化、注意力或损失加mask。',18);
b+=box(405,430,625,110,C.warm)+t(425,460,'Unicode提醒：汉字“我”是1个码点，UTF-8为3字节。',17);
b+=t(425,495,'码点 U+6211 → 字节 E6 88 91 → byte ID 230,136,145',17);
b+=t(425,527,'固定byte分词时，1个汉字在这里对应3个token。',17);
save('01-text-id-embedding.svg','离散token ID与嵌入查表及Unicode字节差别',b,580);
b=t(25,34,'图14-2  原创BPE手算：每轮重计频次，冻结合并顺序后再编码',22);
const heads=['初始','合并1','合并2','合并3'];
for(let j=0;j<4;j++)b+=t(185+210*j,105,heads[j],20,C.blue);
const table=[
 ['abab ×3','a b a b END','ab ab END','ab abEND','ababEND'],
 ['abac ×2','a b a c END','ab a c END','ab a c END','ab a c END'],
 ['cab ×1','c a b END','c ab END','c abEND','c abEND']
];
for(let i=0;i<3;i++){
 b+=t(35,165+70*i,table[i][0],18);
 for(let j=0;j<4;j++){b+=box(175+210*j,132+70*i,200,54,j===0?'white':C.light);b+=t(187+210*j,165+70*i,table[i][j+1],17);}
}
b+=t(35,390,'频次加权token总数：29 → 20 → 16 → 13',21);
b+=t(35,435,'① (a,b)：2×3 + 1×2 + 1×1 = 9 次，合成 ab。',18);
b+=t(35,475,'② (ab,END)：3 + 1 = 4 次，合成 abEND。',18);
b+=t(35,515,'③ (ab,abEND)：3 次，合成 ababEND。',18);
b+=t(35,558,'END在图中表示原子符号 </w>；它标词尾，不是整段文本的EOS。',17,C.orange);
save('02-bpe-hand-merges.svg','三轮BPE合并的精确频次和词尾边界',b,600);
b=t(25,34,'图14-4  静态向量按ID固定；上下文表示还取决于周围输入',22);
b+=box(40,85,480,115)+t(60,120,'水果 · 苹果 · 甜',23)+t(60,155,'邻居均值 [2,0]；同一苹果向量 [1,1]',18)+t(60,185,'h = [1,1] + [2,0] = [3,1]',19,C.blue);
b+=box(40,240,480,115,C.warm)+t(60,275,'公司 · 苹果 · 发布',23)+t(60,310,'邻居均值 [0,2]；同一苹果向量 [1,1]',18)+t(60,340,'h = [1,1] + [0,2] = [1,3]',19,C.orange);
b+='<path d="M635,425H1020M655,440V75" fill="none" stroke="'+C.line+'" stroke-width="2"/>';
for(let i=0;i<5;i++)b+=t(651+70*i,458,String(i),14)+t(625,430-70*i,String(i),14);
b+=arrow(655,425,725,355)+arrow(725,355,865,355)+arrow(725,355,725,215);
b+=t(735,380,'e=[1,1]',18)+t(845,330,'h水果=[3,1]',18,C.blue)+t(735,195,'h公司=[1,3]',18,C.orange);
b+=t(45,490,'h = 中心向量 + 两个邻居向量的均值；这里只是可手算的上下文函数。',18);
b+=t(45,530,'不是训练出来的RNN/Transformer；分词相同也不意味着后续上下文表示相同。',17);
save('04-static-and-context.svg','苹果的固定词嵌入与依上下文改变的示意表示',b,570);
console.log(JSON.stringify({generated:3,output:out}));
