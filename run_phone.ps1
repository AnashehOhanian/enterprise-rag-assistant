# Start the estimator so a phone on the same Wi-Fi can open it.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$python = Join-Path $root "venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Missing venv. From this folder run: python -m venv venv; .\venv\Scripts\python.exe -m pip install -r requirements.txt"
}
$env:PYTHONUNBUFFERED = "1"
$env:HOST = "0.0.0.0"
$env:PORT = "5000"
Start-Process -FilePath $python -ArgumentList "-c", "import app; app.main()" -WorkingDirectory $root
& $python -m streamlit run (Join-Path $root "safeguard_ui.py") --server.address 0.0.0.0 --server.port 8502
