[CmdletBinding()]
param(
    [ValidateSet('html','pdf','all')][string]$Format = 'html',
    [string]$Chapter,
    [switch]$Sample,
    [string]$BrowserExecutable
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskNode = Get-Command node -ErrorAction SilentlyContinue
if (-not $taskNode) { throw '需要 Node.js 22.13 或更新版本。' }
if (-not (Test-Path -LiteralPath (Join-Path $taskRoot 'node_modules/markdown-it'))) {
    throw '文档依赖尚未准备。请先在仓库根目录运行 npm ci --ignore-scripts。'
}
$taskArguments = @((Join-Path $PSScriptRoot 'build.mjs'), '--format', $Format)
if ($Sample) { $taskArguments += '--sample' }
if ($Chapter) { $taskArguments += @('--chapter', $Chapter) }
if ($BrowserExecutable) { $taskArguments += @('--browser', $BrowserExecutable) }
& $taskNode.Source @taskArguments
if ($LASTEXITCODE -ne 0) { throw '构建未完成，详见上方错误。' }
