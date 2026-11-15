# Requires Windows PowerShell 5.1+ or PowerShell 7 on Windows.
[CmdletBinding()]
param(
    [string]$PythonPath,
    [switch]$CheckNetwork,
    [string]$ProxyUrl,
    [string]$OutputPath,
    [switch]$Force
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$issues = [System.Collections.Generic.List[string]]::new()

function Read-CimRows {
    param([string]$ClassName, [string[]]$Properties, [string]$Filter)
    try {
        $query = @{ClassName=$ClassName; ErrorAction='Stop'}
        if ($Filter) { $query.Filter = $Filter }
        Get-CimInstance @query | Select-Object -Property $Properties
    } catch {
        $issues.Add("CIM query failed: $ClassName")
    }
}
function Find-Tool {
    param([string]$Name)
    $command = Get-Command $Name -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($command) {
        [ordered]@{status='found_on_path'; path=$command.Source}
    } else {
        [ordered]@{status='not_found_on_path'; path=$null}
    }
}

$chinaZone = [TimeZoneInfo]::FindSystemTimeZoneById('China Standard Time')
$collected = [TimeZoneInfo]::ConvertTime([DateTimeOffset]::UtcNow, $chinaZone)
$osRows = @(Read-CimRows 'Win32_OperatingSystem' @(
    'Caption','Version','BuildNumber','OSArchitecture',
    'TotalVisibleMemorySize','FreePhysicalMemory'
))
$os = $null
if ($osRows.Count -eq 1) {
    $os = [ordered]@{
        caption=$osRows[0].Caption; version=$osRows[0].Version
        build=$osRows[0].BuildNumber; architecture=$osRows[0].OSArchitecture
        total_memory_bytes=([int64]$osRows[0].TotalVisibleMemorySize * 1024)
        free_memory_bytes=([int64]$osRows[0].FreePhysicalMemory * 1024)
    }
}
$hardware = [ordered]@{
    os=$os
    cpu=@(Read-CimRows 'Win32_Processor' @(
        'Name','NumberOfCores','NumberOfLogicalProcessors','MaxClockSpeed'
    ))
    memory_modules=@(Read-CimRows 'Win32_PhysicalMemory' @('Capacity','Speed'))
    gpu=@(Read-CimRows 'Win32_VideoController' @('Name','DriverVersion','AdapterRAM'))
    disks=@(Read-CimRows 'Win32_LogicalDisk' @('DeviceID','Size','FreeSpace') 'DriveType=3')
}
$tools = [ordered]@{}
foreach ($toolName in @(
    'git','python','python3','py','pip','uv','conda','nvidia-smi','nvcc',
    'node','npm','pandoc','quarto','xelatex','tectonic','dot'
)) { $tools[$toolName] = Find-Tool $toolName }

if (-not $PythonPath) {
    $localPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $localPython) {
        $PythonPath = $localPython
    } elseif ($tools.python.status -eq 'found_on_path' -and
              $tools.python.path -notmatch '\\WindowsApps\\') {
        $PythonPath = $tools.python.path
    }
}
$python = [ordered]@{status='not_found_or_not_selected'}
$smoke = [ordered]@{status='not_run'}
if ($PythonPath) {
    $pythonCode = @'
import importlib.metadata as m, importlib.util as u, json, platform, sys
packages = {}
for name in ["pip", "numpy", "torch", "torchvision", "transformers", "datasets",
             "accelerate", "matplotlib", "jupyter", "psutil", "venv"]:
    found = u.find_spec(name) is not None
    try:
        version = m.version(name)
    except m.PackageNotFoundError:
        version = None
    packages[name] = {"status": "available" if found else "not_installed",
                      "version": version}
print(json.dumps({"status": "passed", "executable": sys.executable,
                  "version": platform.python_version(), "bits": platform.architecture()[0],
                  "is_virtualenv": sys.prefix != sys.base_prefix, "packages": packages},
                 ensure_ascii=True))
'@
    try {
        $pythonJson = @(& $PythonPath -B -c $pythonCode)
        if ($LASTEXITCODE -ne 0) { throw 'Python package inventory failed' }
        $python = ($pythonJson -join "`n") | ConvertFrom-Json
        $smokeJson = @(& $PythonPath -B (Join-Path $PSScriptRoot 'cpu-smoke.py') --with-frameworks)
        $smokeExit = $LASTEXITCODE
        $smoke = ($smokeJson -join "`n") | ConvertFrom-Json
        if ($smokeExit -ne 0) { $issues.Add('CPU smoke test or installed framework probe failed') }
    } catch {
        $issues.Add('Python inspection or CPU smoke test failed')
        $smoke = [ordered]@{status='failed'; error=$_.Exception.GetBaseException().Message}
    }
} else { $issues.Add('No usable Python selected; Windows Store aliases are not executed') }

$nvidiaCandidates = @()
if ($tools['nvidia-smi'].status -eq 'found_on_path') {
    $nvidiaCandidates += $tools['nvidia-smi'].path
}
foreach ($candidate in @(
    (Join-Path $env:WINDIR 'System32\nvidia-smi.exe'),
    'C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe'
)) {
    if (Test-Path -LiteralPath $candidate) { $nvidiaCandidates += $candidate }
}
$nvidia = [ordered]@{
    status='not_found_on_path_or_standard_locations'
    gpus=@()
    note='Win32_VideoController AdapterRAM is not a reliable dedicated/shared VRAM budget.'
}
if ($nvidiaCandidates.Count -gt 0) {
    $nvidiaPath = @($nvidiaCandidates | Select-Object -Unique)[0]
    $gpuInfo = @(& $nvidiaPath --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader)
    $nvidia = [ordered]@{
        status=$(if ($LASTEXITCODE -eq 0) {'passed'} else {'failed'})
        path=$nvidiaPath; gpus=$gpuInfo
    }
}

$proxy = [ordered]@{
    windows_enabled=$null; safe_server=$null; pac_configured=$false
    environment_proxy_present=$false; git_proxy_configured=$false
}
try {
    $settings = Get-ItemProperty -LiteralPath 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings'
    $proxy.windows_enabled = ($settings.ProxyEnable -eq 1)
    $proxy.pac_configured = -not [string]::IsNullOrWhiteSpace($settings.AutoConfigURL)
    if ($settings.ProxyServer -and $settings.ProxyServer -notmatch '@') {
        $proxy.safe_server = $settings.ProxyServer
    }
} catch { $issues.Add('Windows proxy settings could not be inspected') }
foreach ($key in @('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY')) {
    if ([Environment]::GetEnvironmentVariable($key)) { $proxy.environment_proxy_present=$true }
}
if ($tools.git.status -eq 'found_on_path') {
    $gitProxy = & $tools.git.path config --get-urlmatch http.proxy https://github.com/
    $proxy.git_proxy_configured = -not [string]::IsNullOrWhiteSpace(($gitProxy -join ''))
}
$fontStatus = [ordered]@{}
foreach ($fontName in @('msyh.ttc','simhei.ttf','simsun.ttc')) {
    $fontStatus[$fontName] = Test-Path -LiteralPath (Join-Path $env:WINDIR ("Fonts\" + $fontName))
}

$network = [System.Collections.Generic.List[object]]::new()
if ($CheckNetwork) {
    Add-Type -AssemblyName System.Net.Http
    $proxyUri = $null
    if ($ProxyUrl) {
        $proxyUri = [Uri]$ProxyUrl
    } elseif ($proxy.windows_enabled -and $proxy.safe_server) {
        $server = [string]$proxy.safe_server
        if ($server -match '(?:^|;)https=([^;]+)') { $server=$Matches[1] }
        elseif ($server -match '(?:^|;)http=([^;]+)') { $server=$Matches[1] }
        if ($server -notmatch ';') {
            if ($server -notmatch '^https?://') { $server='http://' + $server }
            $proxyUri = [Uri]$server
        }
    }
    if ($proxyUri -and $proxyUri.UserInfo) {
        throw 'Credential-bearing proxy URLs are not accepted by this report tool'
    }
    $clients = @{}
    $pending = @()
    try {
        $modes=@('direct')
        if ($proxyUri) { $modes += 'proxy' }
        foreach ($mode in $modes) {
            $handler=[System.Net.Http.HttpClientHandler]::new()
            $handler.UseProxy=($mode -eq 'proxy')
            if ($mode -eq 'proxy') { $handler.Proxy=[System.Net.WebProxy]::new($proxyUri) }
            $client=[System.Net.Http.HttpClient]::new($handler)
            $client.Timeout=[TimeSpan]::FromSeconds(12)
            $clients[$mode]=$client
            foreach ($url in @(
                'https://github.com/wswanking/deep-learning-journey',
                'https://pypi.org/simple/numpy/',
                'https://download.pytorch.org/whl/cpu/torch/',
                'https://huggingface.co/'
            )) {
                $request=[System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::Head,$url)
                $pending += [pscustomobject]@{
                    mode=$mode; url=$url; request=$request
                    job=$client.SendAsync($request)
                }
            }
        }
        foreach ($probe in $pending) {
            try {
                $response=$probe.job.GetAwaiter().GetResult()
                $network.Add([ordered]@{
                    mode=$probe.mode; url=$probe.url
                    http_status=[int]$response.StatusCode
                    success=$response.IsSuccessStatusCode
                })
                $response.Dispose()
            } catch {
                $network.Add([ordered]@{
                    mode=$probe.mode; url=$probe.url
                    http_status=$null; success=$false
                    error=$_.Exception.GetBaseException().Message
                })
            } finally { $probe.request.Dispose() }
        }
    } finally { foreach ($client in $clients.Values) { $client.Dispose() } }
}
$report = [ordered]@{
    schema_version=1
    collected_at=$collected.ToString('o')
    timezone='Asia/Shanghai'
    powershell_version=$PSVersionTable.PSVersion.ToString()
    inspection_status=$(if ($issues.Count -eq 0) {'complete'} else {'partial'})
    hardware=$hardware; tools=$tools; python=$python; cpu_smoke=$smoke
    nvidia=$nvidia; proxy=$proxy; chinese_fonts=$fontStatus
    project_virtualenv_exists=(Test-Path -LiteralPath (Join-Path $repoRoot '.venv\Scripts\python.exe'))
    cuda_path_environment_present=(-not [string]::IsNullOrWhiteSpace($env:CUDA_PATH))
    network_probed=[bool]$CheckNetwork
    network=$network.ToArray()
    issues=$issues.ToArray()
}
$json=$report | ConvertTo-Json -Depth 12
if ($OutputPath) {
    $destination=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputPath)
    if ((Test-Path -LiteralPath $destination) -and -not $Force) {
        throw 'Output already exists; choose a new path or pass -Force explicitly'
    }
    $parent=Split-Path -Parent $destination
    if (-not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    [System.IO.File]::WriteAllText($destination, $json + [Environment]::NewLine,
                                  [System.Text.UTF8Encoding]::new($false))
}
Write-Output $json
if ($issues.Count -gt 0) { exit 1 }
