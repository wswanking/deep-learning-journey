import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { PDFDocument } from 'pdf-lib';
import { selectDocuments, compileBook, validateSvg, assertVectorPdf, build, projectRoot } from '../scripts/build.mjs';

const config={title:'测试书',subtitle:'测试',language:'zh-CN',samples:[],
  chapters:[{id:'00',title:'测试章',source:'docs/chapters/00-test.md'},
    {id:'01',title:'下一章',source:'docs/chapters/01-test.md'}]};
function fixture(t,text='# 测试章\n\n## 定义\n\n正文。'){
  const parent=path.join(projectRoot,'build');
  fs.mkdirSync(parent,{recursive:true});
  const root=fs.mkdtempSync(path.join(parent,'test-build-'));
  fs.mkdirSync(path.join(root,'docs/chapters'),{recursive:true});
  fs.mkdirSync(path.join(root,'docs/styles'),{recursive:true});
  fs.writeFileSync(path.join(root,'docs/chapters/00-test.md'),text);
  fs.writeFileSync(path.join(root,'docs/chapters/01-test.md'),'# 下一章\n\n## 定义\n\n下一段。');
  fs.copyFileSync(path.join(projectRoot,'docs/styles/book.css'),path.join(root,'docs/styles/book.css'));
  t.after(()=>{
    const realParent=fs.realpathSync(parent),realRoot=fs.realpathSync(root);
    const relative=path.relative(realParent,realRoot);
    assert.ok(relative.startsWith('test-build-') && !path.isAbsolute(relative));
    fs.rmSync(realRoot,{recursive:true,force:true});
  });
  return root;
}
const entry=()=>[{...config.chapters[0],draft:false}];

test('全书只包含明确完成的正式章节，待办不会混入',()=>{
  const docs=selectDocuments(config,'### [x] 00：测试章\n### [ ] 01：下一章');
  assert.deepEqual(docs.map(d=>d.id),['00']);
  assert.throws(()=>selectDocuments(config,'### [ ] 00：测试章'),/尚无已完成章节/);
});
test('明确单章可预览草稿，并保留未完成标记',()=>{
  const docs=selectDocuments(config,'',{chapter:'01'});
  assert.equal(docs.length,1);assert.equal(docs[0].draft,true);
});
test('阻止重复编号与乱序全书',()=>{
  assert.throws(()=>selectDocuments({...config,chapters:[config.chapters[0],config.chapters[0]]},''),/重复/);
  assert.throws(()=>selectDocuments({...config,chapters:[...config.chapters].reverse()},
    '### [x] 00：测试\n### [x] 01：测试'),/升序/);
});
test('跨章中文锚点改写，重复标题保持唯一',t=>{
  const root=fixture(t,'# 测试章\n\n## 定义\n\n## 定义\n\n[下一章](01-test.md#定义)\n');
  const result=compileBook(root,config,config.chapters);
  assert.match(result.html,/href="#chapter-01-定义"/);
  assert.match(result.html,/id="chapter-00-定义-1"/);
});
test('缺失本地链接和锚点会终止构建',t=>{
  const root=fixture(t,'# 测试章\n\n[不存在](missing.md)\n');
  assert.throws(()=>compileBook(root,config,entry()),/缺少本地文件/);
  fs.writeFileSync(path.join(root,'docs/chapters/00-test.md'),'# 测试章\n\n[锚点](#未定义)');
  assert.throws(()=>compileBook(root,config,entry()),/缺少本地锚点/);
});
test('位图图片及缺失 SVG 均被拒绝',t=>{
  const root=fixture(t,'# 测试章\n\n![截图](plot.png)\n');
  assert.throws(()=>compileBook(root,config,entry()),/SVG 矢量图/);
  fs.writeFileSync(path.join(root,'docs/chapters/00-test.md'),'# 测试章\n\n![图](missing.svg)');
  assert.throws(()=>compileBook(root,config,entry()),/缺少本地文件/);
});
test('SVG 不能包装位图、滤镜或外部资源',()=>{
  const prefix='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">';
  assert.throws(()=>validateSvg(prefix+'<image href="data:image/png;base64,AA=="/></svg>'),/位图/);
  assert.throws(()=>validateSvg(prefix+'<filter id="blur"/></svg>'),/矢量输出/);
  assert.throws(()=>validateSvg(prefix+'<use href="https://example.com/a.svg"/></svg>'),/外部/);
  assert.doesNotThrow(()=>validateSvg(prefix+'<path d="M0 0L10 10" stroke="black"/></svg>'));
});
test('非法数学公式失败，代码里的美元符号保持原样',t=>{
  const root=fixture(t,'# 测试章\n\n$\\thisCommandDoesNotExist{x}$');
  assert.throws(()=>compileBook(root,config,entry()),/KaTeX|Undefined|ParseError/);
  fs.writeFileSync(path.join(root,'docs/chapters/00-test.md'),'# 测试章\n\n~~~text\n$not_math$\n~~~\n\n$x^2$');
  const result=compileBook(root,config,entry());
  assert.equal(result.metadata.math_expressions,1);
  assert.match(result.html,/\$not_math\$/);
  assert.doesNotMatch(result.html,/url\(https?:/);
});
test('正式样张含三个真矢量图与本地数学字体',()=>{
  const book=JSON.parse(fs.readFileSync(path.join(projectRoot,'docs/book.json'),'utf8'));
  const result=compileBook(projectRoot,book,book.samples,{sample:true});
  assert.equal(result.metadata.vector_figures,3);
  assert.ok(result.metadata.math_expressions>=10);
  assert.doesNotMatch(result.html,/<image(?:\s|>)/i);
  assert.match(result.html,/figure-0-arrow/);
});
test('失败的构建不覆盖上一版成品',async t=>{
  const root=fixture(t,'# 测试章\n\n![缺图](no.svg)');
  fs.writeFileSync(path.join(root,'docs/book.json'),JSON.stringify(config));
  fs.writeFileSync(path.join(root,'docs/chapter-plan.md'),'### [x] 00：测试');
  fs.mkdirSync(path.join(root,'build/html'),{recursive:true});
  const output=path.join(root,'build/html/deep-learning-journey.html');
  fs.writeFileSync(output,'previous-good-output');
  await assert.rejects(build({root,format:'html'}),/缺少本地文件/);
  assert.equal(fs.readFileSync(output,'utf8'),'previous-good-output');
});
test('PDF 构建门禁识别位图对象，而非只看扩展名',async()=>{
  const vector=await PDFDocument.create();
  vector.addPage().drawLine({start:{x:0,y:0},end:{x:20,y:20}});
  const clean=await assertVectorPdf(await vector.save());
  assert.equal(clean.raster_images,0);
  const raster=await PDFDocument.create();
  const png=await raster.embedPng(Buffer.from(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jF1sAAAAASUVORK5CYII=','base64'));
  raster.addPage().drawImage(png,{x:0,y:0,width:10,height:10});
  await assert.rejects(assertVectorPdf(await raster.save()),/位图对象/);
});

test('带引号的 SVG 局部引用也会按图隔离',t=>{
  const root=fixture(t,'# 测试章\n\n![图](arrow.svg)');
  fs.writeFileSync(path.join(root,'docs/chapters/arrow.svg'),
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><defs><marker id="arrow"/></defs><path d="M0 0L9 9" marker-end="url(\\x27#arrow\\x27)"/></svg>'.replaceAll('\\x27',"'"));
  const result=compileBook(root,config,entry());
  assert.match(result.html,/id="figure-0-arrow"/);
  assert.ok(result.html.includes("url('#figure-0-arrow')"));
});
test('默认全书入口按完成清单实际输出多章 HTML',async t=>{
  const root=fixture(t);
  fs.writeFileSync(path.join(root,'docs/book.json'),JSON.stringify(config));
  fs.writeFileSync(path.join(root,'docs/chapter-plan.md'),'### [x] 00：测试\n### [x] 01：下一章');
  const report=await build({root,format:'html'});
  assert.equal(report.sample,false);
  assert.deepEqual(report.chapters.map(d=>d.id),['00','01']);
  assert.equal(report.chapters.some(d=>d.draft),false);
  assert.ok(fs.existsSync(report.html));
  const html=fs.readFileSync(report.html,'utf8');
  assert.ok(html.indexOf('id="chapter-00-测试章"')<html.indexOf('id="chapter-01-下一章"'));
});
