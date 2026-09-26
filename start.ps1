<#
Mustaff 启动脚本

用法：
  .\start.ps1                      启动图形界面（默认）
  .\start.ps1 gui                  启动图形界面
  .\start.ps1 cli <参数...>        调用命令行工具，参数原样透传
  .\start.ps1 install              强制重新安装/更新依赖

示例：
  .\start.ps1 cli "song.mp3" --keys 4 --format osu -o .\output
  .\start.ps1 cli --help

首次运行会自动在项目目录创建 .venv 虚拟环境并安装依赖（--system-site-packages，
可复用系统 Python 已装的包）。安装时自动在镜像源与 PyPI 官方源之间回退重试。
如提示“禁止运行脚本”，先执行：
  powershell -ExecutionPolicy Bypass -File .\start.ps1
#>

param(
    [Parameter(Position = 0)]
    [ValidateSet("gui", "cli", "install")]
    [string]$Mode = "gui",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$venvDir = Join-Path $PSScriptRoot ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$depsMarker = Join-Path $venvDir ".deps_ok"

function Install-Packages([string]$py, [string[]]$pkgs) {
    # --no-cache-dir：避免使用可能被镜像反爬页污染的 HTTP 缓存
    & $py -m pip install @pkgs --no-cache-dir --disable-pip-version-check -q
    if ($LASTEXITCODE -eq 0) { return $true }
    Write-Host "[Warn] 默认镜像安装失败，改用 PyPI 官方源重试..." -ForegroundColor Yellow
    & $py -m pip install @pkgs --no-cache-dir --disable-pip-version-check -q -i https://pypi.org/simple
    return ($LASTEXITCODE -eq 0)
}

function Install-Deps([string]$py) {
    Write-Host "安装依赖..." -ForegroundColor Cyan
    if (-not (Install-Packages $py @("-r", "requirements.txt"))) {
        # pygame 在过新的 Python 上可能没有预编译包，核心功能不依赖它
        Write-Host "[Warn] 完整依赖安装失败，尝试跳过 pygame 安装核心依赖..." -ForegroundColor Yellow
        $core = @("librosa", "numpy", "click", "soundfile", "matplotlib", "sv-ttk", "comtypes", "miniaudio")
        if (-not (Install-Packages $py $core)) {
            Write-Host "[Error] 依赖安装失败" -ForegroundColor Red
            exit 1
        }
        Write-Host "[Warn] 已跳过 pygame：预览播放功能不可用，其余功能正常" -ForegroundColor Yellow
    }
    New-Item -ItemType File -Path $depsMarker -Force | Out-Null
    Write-Host "[OK] 依赖就绪" -ForegroundColor Green
}

# 首次运行：创建项目内虚拟环境
if (-not (Test-Path $venvPython)) {
    # 优先 Python 3.12（pygame 等依赖有预编译包），再退回其他可用版本
    $baseCmd = $null
    if (Get-Command python3.12 -ErrorAction SilentlyContinue) {
        $baseCmd = @("python3.12")
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 --version *> $null
        if ($LASTEXITCODE -eq 0) { $baseCmd = @("py", "-3.12") }
    }
    if (-not $baseCmd -and (Get-Command python -ErrorAction SilentlyContinue)) {
        $baseCmd = @("python")
    } elseif (-not $baseCmd -and (Get-Command py -ErrorAction SilentlyContinue)) {
        $baseCmd = @("py")
    }
    if (-not $baseCmd) {
        Write-Host "[Error] 未找到 Python，请先安装 Python 3.9+ 并加入 PATH" -ForegroundColor Red
        exit 1
    }
    $exe = $baseCmd[0]
    $extra = @($baseCmd | Select-Object -Skip 1)
    Write-Host "首次运行，创建虚拟环境 .venv（基础解释器: $exe $extra）..." -ForegroundColor Cyan
    & $exe @extra -m venv --system-site-packages $venvDir
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPython)) {
        Write-Host "[Error] 虚拟环境创建失败" -ForegroundColor Red
        exit 1
    }
    Install-Deps $venvPython
} elseif ($Mode -eq "install" -or -not (Test-Path $depsMarker)) {
    Install-Deps $venvPython
}

switch ($Mode) {
    "install" { }
    "gui" {
        & $venvPython entry_gui.py
        exit $LASTEXITCODE
    }
    "cli" {
        & $venvPython entry_cli.py @Rest
        exit $LASTEXITCODE
    }
}
