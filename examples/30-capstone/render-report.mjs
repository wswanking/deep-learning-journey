import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {chromium} from 'playwright-core';
import {compileBook,assertVectorPdf} from '../../scripts/build.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const folder=path.join(root,'build/experiments/30-capstone');
const config={...JSON.parse(fs.readFileSync(path.join(root,'docs/book.json'),'utf8')),title:'微型文本与证据实验报告',subtitle:'真实CPU训练、固定评估与失败证据'};
const compiled=compileBook(root,config,[{id:'capstone-report',title:'综合实践报告',source:'build/experiments/30-capstone/report.md',draft:false}]);
const html=path.join(folder,'report.html');const pdf=path.join(folder,'report.pdf');
fs.writeFileSync(html,compiled.html);
for(const [relative,source] of compiled.attachments){const target=path.join(folder,'sources',relative);fs.mkdirSync(path.dirname(target),{recursive:true});fs.copyFileSync(source,target);}
const browser=await chromium.launch({channel:'msedge',headless:true});
const results=[];
try{
 for(const width of [1440,390]){
  const page=await browser.newPage({viewport:{width,height:1000}});await page.route('**/*',route=>/^(file:|data:|about:)/.test(route.request().url())?route.continue():route.abort());await page.goto(pathToFileURL(html).href);await page.evaluate(()=>document.fonts.ready);
  const qa=await page.evaluate(()=>({width:innerWidth,document_width:document.documentElement.scrollWidth,overflow:document.documentElement.scrollWidth>innerWidth+2,math_errors:document.querySelectorAll('.katex-error').length,raster_elements:document.querySelectorAll('img,canvas').length,vector_figures:document.querySelectorAll('.vector-figure>svg').length,clipped_figures:[...document.querySelectorAll('.vector-figure>svg')].filter(svg=>{const b=svg.getBBox(),v=svg.viewBox.baseVal;return b.x<v.x-2||b.y<v.y-2||b.x+b.width>v.x+v.width+2||b.y+b.height>v.y+v.height+2;}).length}));results.push(qa);await page.close();
 }
 if(results.some(r=>r.overflow||r.math_errors||r.raster_elements||r.clipped_figures))throw Error('report browser QA failed: '+JSON.stringify(results));
 const page=await browser.newPage({viewport:{width:658,height:1000}});await page.route('**/*',route=>/^(file:|data:|about:)/.test(route.request().url())?route.continue():route.abort());await page.goto(pathToFileURL(html).href);await page.emulateMedia({media:'print'});await page.evaluate(()=>document.fonts.ready);const overflow=await page.evaluate(()=>[...document.querySelectorAll('table,pre,.katex-display,.vector-figure')].filter(e=>e.scrollWidth>e.clientWidth+3).map(e=>e.textContent.slice(0,80)));if(overflow.length)throw Error('report print overflow: '+JSON.stringify(overflow));
 await page.pdf({path:pdf,format:'A4',preferCSSPageSize:true,printBackground:true,tagged:true,outline:true,displayHeaderFooter:true,headerTemplate:'<span></span>',footerTemplate:'<div style="width:100%;font-size:9px;color:#667085;text-align:center;"><span class="pageNumber"></span> / <span class="totalPages"></span></div>',margin:{top:'18mm',bottom:'18mm',left:'18mm',right:'18mm'}});
 const structure=await assertVectorPdf(fs.readFileSync(pdf));if(!structure.tagged||!structure.has_outline)throw Error('missing PDF document structure');
 const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');const report={status:'passed',checked_at:new Date().toISOString(),html_sha256:hash(html),pdf_sha256:hash(pdf),viewports:results,pdf:structure,...compiled.metadata};fs.writeFileSync(path.join(folder,'report-publication.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
}finally{await browser.close();}
