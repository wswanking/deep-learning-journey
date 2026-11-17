"""Original pure SVG diagrams, with actual chapter-26 result bars."""
from pathlib import Path
import json,html
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'docs/assets/26-retrieval-augmented-generation'
def t(x,y,s,z=20):return f'<text x="{x}" y="{y}" font-size="{z}" fill="#18324f">{html.escape(str(s))}</text>'
def b(x,y,w,h,fill='#edf5ff'):return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="#88aacd"/>'
def a(x,y,x2):return f'<path d="M{x} {y} H{x2} m-8 -5 l8 5 l-8 5" fill="none" stroke="#3175bb" stroke-width="2"/>'
def box(x,y,w,label,sub):return b(x,y,w,72)+t(x+12,y+30,label,21)+t(x+12,y+56,sub,16)
def save(name,title,body):
 OUT.mkdir(parents=True,exist_ok=True)
 s='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 440" role="img"><title>'+html.escape(title)+'</title><rect width="960" height="440" rx="12" fill="#f7fafc"/><g font-family="Microsoft YaHei, SimSun, Arial, sans-serif">'+t(26,38,title,25)+body+'</g></svg>'
 (OUT/name).write_text(s,encoding='utf-8')
def main():
 r=json.loads((ROOT/'build/experiments/26-retrieval-augmented-generation/results.json').read_text(encoding='utf-8'))
 body=t(28,85,'离线：版本化的资料与表示',21)
 for x,label,sub in [(28,'原文快照','文档ID / 原文区间'),(270,'切分与编码','宽度 / 重叠 / 特征'),(512,'精确向量索引','24 × 280，float64')]:body+=box(x,106,185,label,sub)
 body+=a(214,142,264)+a(456,142,506)+t(28,226,'在线：每个中间结果都能追溯',21)
 for x,label,sub in [(28,'问题向量','沿用冻结IDF'),(263,'检索 → 重排','k=3，最多2块'),(498,'上下文 → 回答','记录真实输入'),(733,'引用核验','ID / 区间 / 支持')]:body+=box(x,248,198,label,sub)
 for x in [227,462,697]:body+=a(x,284,x+30)
 body+=t(28,375,'检索分数、回答正确与引用支持，分别评价。',21)+t(28,408,'本章检索用字符统计；生成探针读取第21章真实35,328参数基座。',18)
 save('01-pipeline.svg','图26-1 离线入库与在线回答：保留完整证据链',body)
 body=t(28,82,'原串：A B C D E F G H I J K L M N',23)
 for y,items in [(110,[('第一块 [0,8)','ABCDEFGH',28),('第二块 [6,14)','GHIJKLMN',490)]),(220,[('无重叠 [0,8)','ABCDEFGH',28),('无重叠 [8,14)','IJKLMN',490)])]:
  for title,value,x in items:body+=b(x,y,430,85)+t(x+16,y+28,title,19)+t(x+16,y+64,value,27)
 body+=t(28,350,'证据 GHIJ：重叠2时第二块完整保留；重叠0时两块都截断。',21)+t(28,394,'增加重叠也增加重复编码；字符坐标不能直接当作字节token预算。',19)
 save('02-chunk-boundary.svg','图26-2 切分边界：宽度8、重叠2、步幅6',body)
 body=b(28,86,420,245)+t(48,120,'查询 u=(1,1)',23)+t(48,168,'v₁=(2,0)：点积2，余弦0.7071',20)+t(48,214,'v₂=(1,1)：点积2，余弦1',20)+t(48,272,'先明确是否归一化，再比较排名。',20)
 body+=b(490,86,442,245)+t(510,120,'检索候选 → 联合检查 → 上下文',21)+t(510,168,'双塔：问题与片段分开编码',20)+t(510,211,'重排：只处理已收到的k个候选',20)+t(510,254,'本章规则检查实体与字段，无神经训练',18)+t(510,298,'候选外证据：重排不能自动找回',20)
 body+=t(28,385,'矩阵评分 s = Vû；小索引精确扫描。语义泛化另需训练与评价。',20)
 save('03-vectors-rerank.svg','图26-3 向量分数与候选重排的作用范围',body)
 body=t(28,85,'验证回答正确数 / 6：先选择协议',21)
 for i,c in enumerate(r['selection_candidates']):
  y=105+i*41;v=c['validation']['answer_em'];body+=t(28,y+23,f"宽{c['width']} / 重叠{c['overlap']}",18)+b(218,y,240*v if v else 1,26,'#3c7fc0')+t(470,y+22,f'{round(v*6)}/6',19)
 body+=b(560,95,370,255)+t(578,130,'冻结后的12个可回答测试',20)+t(578,181,'规则抽取：12/12',24)+t(578,229,'同一LLM，无检索：0/12',21)+t(578,277,'同一LLM，有完整证据：0/12',21)+t(578,318,'规则与生成模型分别报告。',18)
 body+=t(28,397,'测试top1 / 无重排 / 宽36 / 宽144规则回答均12/12，没有测试优势证据。',18)
 save('04-measured-results.svg','图26-4 本机实测：验证边界效应与生成负结果',body)
 body=box(28,88,250,'有完整事实','受限规则读者：可抽取')+box(355,88,250,'实际上下文','保留来源区间与原句')+box(682,88,250,'输出与引用','实体、字段和值一起核对')+a(280,124,348)+a(607,124,675)
 for y,c,d in [(200,'候选没有证据','查切分 / 表示 / 索引 / k'),(270,'候选有，上下文没有','查重排 / 去重 / 预算 / 截断'),(340,'完整证据下仍答错','微型LLM探针：12题全部失败')]:body+=b(28,y,904,56,'#fff5e8')+t(48,y+35,c,21)+t(400,y+35,d,20)
 save('05-evidence-failures.svg','图26-5 诊断失败：记录每一段真正经过的信息',body)
if __name__=='__main__':main()
