"""Original tiny corpus; facts grouped by color/object, bilingual paraphrases stay together."""
import json
from pathlib import Path
COLORS=[('red','红色'),('blue','蓝色'),('green','绿色'),('yellow','黄色'),('white','白色'),('black','黑色')]
SHAPES=[('square','方块'),('circle','圆形'),('triangle','三角形'),('star','星形')]

def original_records():
    records=[]
    for color,zh_color in COLORS:
        for shape,zh_shape in SHAPES:
            group=f'{color}-{shape}'
            texts=[f'The {color} {shape} rests on the table. A small note names this {color} {shape}. The room is quiet.\n',f'{zh_color}{zh_shape}放在桌上。记录写着：{zh_color}{zh_shape}。房间很安静，窗边有一束光。\n']
            records.extend({'id':f'{group}-{language}','group_id':group,'text':text} for language,text in zip(['en','zh'],texts))
    return records

if __name__=='__main__':
    Path(__file__).with_name('corpus.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in original_records()),encoding='utf-8')
