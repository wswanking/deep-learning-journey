import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// Original vector figures, based on the chapter's analytic examples.
// No screenshot, embedded image, external font, or raster intermediate is used.
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const destination = path.join(root, 'docs/assets/02-calculus');
fs.mkdirSync(destination, { recursive: true });
const palette = { ink: '#16324f', muted: '#526b80', blue: '#235bc7', teal: '#007d75', orange: '#b45b11', line: '#c7d6e1', pale: '#eef5fa' };
const esc = text => String(text).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const text = (x, y, label, size = 25, color = palette.ink, anchor = 'start') => `<text x="${x}" y="${y}" font-size="${size}" fill="${color}" text-anchor="${anchor}">${esc(label)}</text>`;
const line = (x1, y1, x2, y2, color = palette.line, width = 2, dash = '') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}"${dash ? ` stroke-dasharray="${dash}"` : ''}/>`;
const arrow = (x1, y1, x2, y2, color = palette.ink, width = 3) => {
  const angle = Math.atan2(y2 - y1, x2 - x1), length = 12;
  const ax = x2 - length * Math.cos(angle - Math.PI / 6), ay = y2 - length * Math.sin(angle - Math.PI / 6);
  const bx = x2 - length * Math.cos(angle + Math.PI / 6), by = y2 - length * Math.sin(angle + Math.PI / 6);
  return line(x1, y1, x2, y2, color, width) + `<path d="M ${ax} ${ay} L ${x2} ${y2} L ${bx} ${by}" fill="none" stroke="${color}" stroke-width="${width}"/>`;
};
const rect = (x, y, width, height, fill = palette.pale, stroke = palette.line) => `<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="14" fill="${fill}" stroke="${stroke}" stroke-width="2"/>`;
const pathFrom = points => 'M ' + points.map(([x, y]) => x.toFixed(2) + ' ' + y.toFixed(2)).join(' L ');
function save(filename, title, desc, width, height, body) {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title desc"><title id="title">${esc(title)}</title><desc id="desc">${esc(desc)}</desc><rect width="${width}" height="${height}" fill="#ffffff"/><g font-family="Microsoft YaHei, Segoe UI, sans-serif">${body}</g></svg>\n`;
  fs.writeFileSync(path.join(destination, filename), svg, 'utf8');
  console.log('docs/assets/02-calculus/' + filename);
}

{
  const sx = s => 90 + s * 180, sy = y => 490 - y * 32;
  let body = text(40, 48, '图02-1  从割线到切线：g(s) = s²', 30);
  for (const y of [0, 2, 4, 6, 8, 10]) body += line(90, sy(y), 670, sy(y)) + text(75, sy(y) + 7, y, 21, palette.muted, 'end');
  for (const s of [0, 1, 2, 3]) body += line(sx(s), 140, sx(s), 490) + text(sx(s), 521, s, 21, palette.muted, 'middle');
  body += arrow(90, 490, 700, 490) + arrow(90, 490, 90, 110);
  body += text(510, 560, 's（无量纲）', 23) + text(34, 105, 'g(s)', 23);
  const curve = Array.from({ length: 151 }, (_, i) => { const s = i * 3.2 / 150; return [sx(s), sy(s * s)]; });
  body += `<path d="${pathFrom(curve)}" fill="none" stroke="${palette.blue}" stroke-width="4"/>`;
  body += line(sx(1.1), sy(4 * 1.1 - 4), sx(3), sy(8), palette.teal, 3);
  const secant = s => 4 + 4.7 * (s - 2);
  body += line(sx(1.25), sy(secant(1.25)), sx(3), sy(secant(3)), palette.orange, 3, '9 6');
  for (const [s, y] of [[2, 4], [2.7, 7.29]]) body += `<circle cx="${sx(s)}" cy="${sy(y)}" r="7" fill="${palette.ink}"/>`;
  body += text(sx(2) - 110, sy(4) - 15, 'P = (2, 4)', 23);
  body += text(sx(2.7) + 5, sy(7.29) - 16, 'Q', 23);
  body += rect(750, 115, 310, 360);
  body += text(775, 160, '蓝线：函数曲线', 25, palette.blue);
  body += text(775, 210, '橙虚线：h = 0.7', 24, palette.orange);
  body += text(775, 250, '割线斜率 = 4.7', 24, palette.orange);
  body += text(775, 310, '绿线：P点切线', 24, palette.teal);
  body += text(775, 350, '切线斜率 = 4', 24, palette.teal);
  body += text(775, 410, 'h → 0，割线趋近切线', 22);
  save('02-01-tangent.svg', '从割线到切线', 's平方曲线在s等于2的切线斜率为4，h等于0.7的割线斜率为4.7。', 1100, 590, body);
}

{
  const sx = x => 395 + x * 90, sy = y => 365 - y * 90;
  let body = text(40, 48, '图02-2  等高线、偏导与梯度：f(x₁, x₂) = x₁² + 2x₂²', 28);
  for (const c of [1, 3, 6, 9]) {
    body += `<ellipse cx="395" cy="365" rx="${Math.sqrt(c) * 90}" ry="${Math.sqrt(c / 2) * 90}" fill="none" stroke="${c === 3 ? palette.blue : palette.line}" stroke-width="${c === 3 ? 3 : 2}"/>`;
    body += text(sx(-Math.sqrt(c)) - 4, sy(0) - 10, 'f=' + c, 20, palette.muted, 'end');
  }
  body += arrow(85, 365, 715, 365) + arrow(395, 575, 395, 100);
  body += text(680, 399, 'x₁', 24) + text(409, 112, 'x₂', 24);
  for (const tick of [-2, -1, 1, 2, 3]) body += line(sx(tick), 360, sx(tick), 370, palette.ink) + text(sx(tick), 399, tick, 20, palette.muted, 'middle');
  for (const tick of [-2, -1, 1, 2]) body += line(390, sy(tick), 400, sy(tick), palette.ink) + text(380, sy(tick) + 7, tick, 20, palette.muted, 'end');
  const px = sx(1), py = sy(1);
  body += line(sx(0.05), sy(1.475), sx(1.95), sy(0.525), palette.orange, 3, '7 6');
  body += arrow(px, py, sx(1.8), sy(2.6), palette.teal, 5);
  body += arrow(px, py, sx(2.1), py, palette.blue, 3);
  body += `<circle cx="${px}" cy="${py}" r="7" fill="${palette.ink}"/>`;
  body += text(px - 15, py + 70, 'P=(1,1)', 23);
  body += text(sx(1.9), sy(2.5), '∇f', 27, palette.teal);
  body += text(sx(1.8), py + 36, '只改x₁', 22, palette.blue);
  body += text(100, 560, '坐标轴同尺度；绿色梯度箭头按0.4倍长度绘制。', 20, palette.muted);
  body += rect(760, 115, 300, 400);
  body += text(780, 160, '在 P 点：', 25);
  body += text(780, 208, '∂f/∂x₁ = 2', 25, palette.blue);
  body += text(780, 252, '∂f/∂x₂ = 4', 25);
  body += text(780, 304, '梯度 ∇f = (2, 4)ᵀ', 23, palette.teal);
  body += text(780, 350, '橙虚线：等高线切线', 21, palette.orange);
  body += text(780, 395, '梯度与切线垂直', 24);
  body += text(780, 451, '方向导数还取决于方向', 21);
  save('02-02-gradient.svg', '等高线与梯度', '函数x1平方加2倍x2平方的等高线为椭圆，在点1,1的梯度为列向量2,4，与f等于3的等高线切线垂直。', 1100, 600, body);
}

{
  let body = text(40, 48, '图02-3  复合函数的依赖路径：y = (x₁x₂ + x₁)²', 30);
  body += rect(50, 155, 160, 80) + text(130, 205, 'x₁ = 1', 27, palette.ink, 'middle');
  body += rect(50, 370, 160, 80) + text(130, 420, 'x₂ = 2', 27, palette.ink, 'middle');
  body += rect(365, 260, 180, 90) + text(455, 295, 'a = x₁x₂', 25, palette.ink, 'middle') + text(455, 332, 'a = 2', 24, palette.ink, 'middle');
  body += rect(655, 260, 175, 90) + text(742, 295, 'z = a + x₁', 24, palette.ink, 'middle') + text(742, 332, 'z = 3', 24, palette.ink, 'middle');
  body += rect(930, 260, 140, 90) + text(1000, 295, 'y = z²', 25, palette.ink, 'middle') + text(1000, 332, 'y = 9', 24, palette.ink, 'middle');
  body += arrow(210, 205, 365, 278, palette.blue) + text(235, 220, '× x₂ = 2', 22, palette.blue);
  body += arrow(210, 410, 365, 335, palette.muted) + text(235, 407, '× x₁ = 1', 22, palette.muted);
  body += line(130, 155, 130, 110, palette.teal, 3) + line(130, 110, 742, 110, palette.teal, 3) + arrow(742, 110, 742, 260, palette.teal);
  body += text(395, 98, '直接路径：∂z/∂x₁ = 1', 23, palette.teal);
  body += arrow(545, 305, 655, 305, palette.blue) + text(600, 286, '× 1', 22, palette.blue, 'middle');
  body += arrow(830, 305, 930, 305, palette.ink) + text(880, 245, '× 2z = 6', 22, palette.ink, 'middle');
  body += rect(80, 490, 940, 120, '#f5f9fc');
  body += text(105, 530, 'x₁的总影响：间接路径 6×1×2 + 直接路径 6×1 = 18', 25);
  body += text(105, 576, 'x₂的总影响：只有间接路径 6×1×1 = 6', 25);
  save('02-03-chain-rule.svg', '分叉计算图与链式法则', 'x1到输出有直接和经乘法节点两条路径，路径内导数相乘、各路径贡献相加，得到梯度18,6。', 1120, 650, body);
}

{
  let body = text(40, 48, '图02-4  Jacobian的行是输出，列是输入', 30);
  body += text(550, 100, '在 x = (2, 3)ᵀ 处：Δz ≈ J Δx', 27, palette.ink, 'middle');
  body += rect(50, 190, 190, 160) + text(145, 235, '输入微小变化', 24, palette.ink, 'middle') + text(145, 284, 'Δx₁', 27, palette.ink, 'middle') + text(145, 323, 'Δx₂', 27, palette.ink, 'middle');
  body += arrow(245, 270, 335, 270, palette.blue, 4);
  body += text(510, 160, '输入x₁', 23, palette.blue, 'middle') + text(650, 160, '输入x₂', 23, palette.blue, 'middle');
  const values = [[4, 1], [3, 2], [2, -1]];
  for (let r = 0; r < 3; r++) {
    body += text(405, 220 + r * 75, '输出z' + ['₁', '₂', '₃'][r], 22, palette.muted, 'end');
    for (let c = 0; c < 2; c++) body += rect(445 + c * 140, 177 + r * 75, 125, 65) + text(507 + c * 140, 220 + r * 75, values[r][c], 28, palette.ink, 'middle');
  }
  body += arrow(735, 270, 815, 270, palette.blue, 4);
  body += rect(825, 170, 235, 220) + text(942, 216, '输出微小变化', 24, palette.ink, 'middle') + text(942, 264, 'Δz₁', 27, palette.ink, 'middle') + text(942, 306, 'Δz₂', 27, palette.ink, 'middle') + text(942, 348, 'Δz₃', 27, palette.ink, 'middle');
  body += text(145, 405, '2×1', 25, palette.ink, 'middle') + text(580, 450, 'J：3×2', 27, palette.blue, 'middle') + text(942, 435, '3×1', 25, palette.ink, 'middle');
  body += text(60, 505, '映射：z₁=x₁²+x₂；z₂=x₁x₂；z₃=2x₁−x₂。', 24);
  body += text(60, 550, '若最终量是标量L，输入梯度由 Jᵀ∇zL 得到，形状回到2×1。', 24);
  save('02-04-jacobian.svg', '非方形Jacobian与维度', '二输入三输出函数的Jacobian为3行2列，将2维输入变化映射成3维输出变化；标量输出梯度用转置映射回输入。', 1100, 590, body);
}
