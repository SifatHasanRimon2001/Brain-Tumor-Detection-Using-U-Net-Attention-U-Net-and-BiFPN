# ============================================================================
# BRISC2025 — Interactive Training & Evaluation Pipeline
# Run from project root:
#   powershell -ExecutionPolicy Bypass -File scripts/run_all.ps1
# ============================================================================
$ErrorActionPreference = "Stop"
$Py = "python"
$Line = "=" * 70
$ProgressFile = ".\.brisc_progress.json"
# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------
$SymPending = "[ ]"
$SymDone    = "[✓]"
$SymFail    = "[✗]"
$SymRun     = "[►]"
$SymSkip    = "[⊘]"
# ---------------------------------------------------------------------------
# Task Registry: 9 configurations
# ---------------------------------------------------------------------------
$Tasks = @(
    @{ Id = 1;  Name = "SEG  / UNet";     Task = "seg";   Model = "unet";    Epochs = 40;  Status = "pending"; Msg = "" },
    @{ Id = 2;  Name = "SEG  / AttUNet";  Task = "seg";   Model = "attunet"; Epochs = 40;  Status = "pending"; Msg = "" },
    @{ Id = 3;  Name = "SEG  / BiFPN";    Task = "seg";   Model = "bifpn";   Epochs = 40;  Status = "pending"; Msg = "" },
    @{ Id = 4;  Name = "CLS  / UNet";     Task = "cls";   Model = "unet";    Epochs = 25;  Status = "pending"; Msg = "" },
    @{ Id = 5;  Name = "CLS  / AttUNet";  Task = "cls";   Model = "attunet"; Epochs = 25;  Status = "pending"; Msg = "" },
    @{ Id = 6;  Name = "CLS  / BiFPN";    Task = "cls";   Model = "bifpn";   Epochs = 25;  Status = "pending"; Msg = "" },
    @{ Id = 7;  Name = "JOINT/ UNet";     Task = "joint"; Model = "unet";    Epochs = 50;  Status = "pending"; Msg = "" },
    @{ Id = 8;  Name = "JOINT/ AttUNet";  Task = "joint"; Model = "attunet"; Epochs = 50;  Status = "pending"; Msg = "" },
    @{ Id = 9;  Name = "JOINT/ BiFPN";    Task = "joint"; Model = "bifpn";   Epochs = 50;  Status = "pending"; Msg = "" }
)
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Save-Progress {
    $save = foreach ($t in $Tasks) {
        [PSCustomObject]@{
            Id      = $t.Id
            Name    = $t.Name
            Task    = $t.Task
            Model   = $t.Model
            Epochs  = $t.Epochs
            Status  = $t.Status
            Msg     = $t.Msg
        }
    }
    $save | ConvertTo-Json -Depth 4 | Set-Content $ProgressFile -Encoding UTF8
}
function Load-Progress {
    # Restore whatever was saved last session (skipped/failed/done/...), then
    # let on-disk checkpoints act as the source of truth for "done", and make
    # sure a task that was "running" when the script last exited (crash,
    # closed window, Ctrl+C) doesn't stay stuck "running" forever.
    $saved = $null
    if (Test-Path $ProgressFile) {
        try {
            $raw = Get-Content $ProgressFile -Raw -ErrorAction Stop
            if ($raw) {
                $saved = @(ConvertFrom-Json $raw -ErrorAction Stop)
            }
        }
        catch {
            Write-Warning "  Could not read $ProgressFile ($($_.Exception.Message)) — starting fresh."
            $saved = $null
        }
    }
    foreach ($t in $Tasks) {
        $prev = $null
        if ($saved) {
            $prev = $saved | Where-Object { $_.Id -eq $t.Id } | Select-Object -First 1
        }
        $ckpt = "runs/$($t.Task)_$($t.Model)/best.ckpt"
        if (Test-Path $ckpt) {
            $t.Status = "done"
            $t.Msg = "checkpoint found"
        }
        elseif ($prev) {
            if ($prev.Status -eq "running") {
                $t.Status = "pending"
                $t.Msg = "interrupted — reset to pending"
            }
            else {
                $t.Status = $prev.Status
                $t.Msg = $prev.Msg
            }
        }
        else {
            $t.Status = "pending"
            $t.Msg = ""
        }
    }
    Save-Progress
}
function Get-Symbol($task) {
    switch ($task.Status) {
        "done"    { return $SymDone }
        "failed"  { return $SymFail }
        "running" { return $SymRun }
        "skipped" { return $SymSkip }
        default   { return $SymPending }
    }
}
function Show-Dashboard {
    Clear-Host
    Write-Host $Line -ForegroundColor Cyan
    Write-Host "  BRISC2025 — Interactive Training Pipeline" -ForegroundColor Cyan
    Write-Host $Line -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  ID | Status | Configuration          | Epochs | Note"
    Write-Host "  ---|--------|------------------------|--------|------------------------------"
    foreach ($t in $Tasks) {
        $sym = Get-Symbol $t
        $color = switch ($t.Status) {
            "done"    { "Green" }
            "failed"  { "Red" }
            "running" { "Yellow" }
            "skipped" { "DarkGray" }
            default   { "White" }
        }
        $note = if ($t.Msg) { "  $($t.Msg)" } else { "" }
        Write-Host ("  {0,2} | {1} | {2,-22} | {3,6} |{4}" -f $t.Id, $sym, $t.Name, $t.Epochs, $note) -ForegroundColor $color
    }
    Write-Host ""
    Write-Host "  Legend: $SymDone Done  $SymFail Failed  $SymRun Running  $SymSkip Skipped  $SymPending Pending"
    Write-Host ""
}
function Read-Choice($prompt, $valid) {
    do {
        $c = Read-Host "  $prompt"
    } until ($valid -contains $c)
    return $c
}
function Toggle-Task($id) {
    $t = $Tasks | Where-Object { $_.Id -eq $id }
    if ($t.Status -eq "pending") {
        $t.Status = "skipped"
        $t.Msg = "user skipped"
    } elseif ($t.Status -eq "skipped") {
        $t.Status = "pending"
        $t.Msg = ""
    }
    # If done/failed, allow reset to pending
    elseif ($t.Status -in @("done","failed")) {
        $t.Status = "pending"
        $t.Msg = "reset by user"
    }
}
function Reset-All {
    foreach ($t in $Tasks) { $t.Status = "pending"; $t.Msg = "" }
    Save-Progress
    Write-Host "  All tasks reset to pending." -ForegroundColor Yellow
    Start-Sleep -Seconds 1
}
function Confirm-Task($task) {
    Show-Dashboard
    Write-Host "  >>> About to run: $($task.Name) ($($task.Task) + $($task.Model))" -ForegroundColor Cyan
    Write-Host "      Epochs: $($task.Epochs) | Checkpoint: runs/$($task.Task)_$($task.Model)/best.ckpt"
    Write-Host ""
    $ans = Read-Host "  Proceed? [Y]es / [N]o / [A]lways yes to remaining / [Q]uit pipeline"
    switch ($ans.ToLower().Trim()) {
        "y" { return "yes" }
        "a" { return "always" }
        "q" { return "quit" }
        default { return "no" }
    }
}
function Run-Task($task) {
    $task.Status = "running"
    $task.Msg = "training..."
    Save-Progress
    Show-Dashboard
    try {
        # --- TRAIN ---
        Write-Host ""
        Write-Host "  [TRAIN] $($task.Name)" -ForegroundColor Yellow
        $trainArgs = @(
            "train.py",
            "--task", $task.Task,
            "--model", $task.Model,
            "--data_root", "./brisc2025",
            "--img_size", "256",
            "--batch", "8",
            "--epochs", "$($task.Epochs)"
        )
        & $Py @trainArgs
        if ($LASTEXITCODE -ne 0) {
            $task.Status = "failed"
            $task.Msg = "training failed"
            Save-Progress
            Write-Warning "  Training failed for $($task.Name)"
            return
        }
        # --- CHECK CHECKPOINT ---
        $ckpt = "runs/$($task.Task)_$($task.Model)/best.ckpt"
        if (-not (Test-Path $ckpt)) {
            $task.Status = "failed"
            $task.Msg = "no checkpoint"
            Save-Progress
            Write-Warning "  Checkpoint not found: $ckpt"
            return
        }
        # --- INFERENCE ---
        $task.Msg = "inferring..."
        Save-Progress
        Show-Dashboard
        Write-Host "  [INFER] $($task.Name)" -ForegroundColor Yellow
        $inferArgs = @(
            "infer.py",
            "--task", $task.Task,
            "--model", $task.Model,
            "--data_root", "./brisc2025",
            "--size", "256",
            "--ckpt", $ckpt
        )
        & $Py @inferArgs
        if ($LASTEXITCODE -ne 0) {
            $task.Status = "failed"
            $task.Msg = "inference failed"
            Save-Progress
            Write-Warning "  Inference failed for $($task.Name)"
            return
        }
        $task.Status = "done"
        $task.Msg = "completed"
        Save-Progress
        Write-Host "  $SymDone $($task.Name) finished successfully." -ForegroundColor Green
        Start-Sleep -Seconds 1
    }
    catch {
        # Catches things $LASTEXITCODE can't, e.g. $Py itself not being found.
        $task.Status = "failed"
        $task.Msg = "error: $($_.Exception.Message)"
        Save-Progress
        Write-Warning "  Unexpected error running $($task.Name): $($_.Exception.Message)"
        Start-Sleep -Seconds 2
    }
}
function Run-Visualization {
    Show-Dashboard
    Write-Host "  >>> Run visualization (visualize.py)?" -ForegroundColor Cyan
    $ans = Read-Host "  Proceed? [Y]es / [N]o"
    if ($ans.ToLower().Trim() -eq "y") {
        Write-Host ""
        Write-Host "  [VISUALIZE] Running comparison..." -ForegroundColor Yellow
        try {
            & $Py visualize.py --data_root "./brisc2025" --index 0
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "  Visualization failed (non-critical)"
            } else {
                Write-Host "  $SymDone Visualization complete." -ForegroundColor Green
            }
        }
        catch {
            Write-Warning "  Visualization failed (non-critical): $($_.Exception.Message)"
        }
        Read-Host "`n  Press Enter to continue..."
    }
}
# ---------------------------------------------------------------------------
# Main Menu
# ---------------------------------------------------------------------------
Load-Progress
while ($true) {
    Show-Dashboard
    Write-Host "  MAIN MENU"
    Write-Host "  --------"
    Write-Host "  [1] Run all PENDING tasks (asks before each)"
    Write-Host "  [2] Run only FAILED tasks (asks before each)"
    Write-Host "  [3] Toggle / select specific tasks (enable/disable)"
    Write-Host "  [4] Reset all tasks to pending"
    Write-Host "  [5] Run visualization only"
    Write-Host "  [6] Save & Exit"
    Write-Host ""
    $choice = Read-Choice "Select option (1-6)" @("1","2","3","4","5","6")
    switch ($choice) {
        "1" {
            $pending = $Tasks | Where-Object { $_.Status -eq "pending" }
            if (-not $pending) { Write-Host "  No pending tasks!"; Start-Sleep -Seconds 2; continue }
            $autoYes = $false
            foreach ($t in $pending) {
                if (-not $autoYes) {
                    $decision = Confirm-Task $t
                    if ($decision -eq "quit") { break }
                    if ($decision -eq "no") { $t.Status = "skipped"; $t.Msg = "skipped by user"; Save-Progress; continue }
                    if ($decision -eq "always") { $autoYes = $true }
                }
                Run-Task $t
            }
            Run-Visualization
        }
        "2" {
            $failed = $Tasks | Where-Object { $_.Status -eq "failed" }
            if (-not $failed) { Write-Host "  No failed tasks!"; Start-Sleep -Seconds 2; continue }
            $autoYes = $false
            foreach ($t in $failed) {
                $t.Status = "pending"; $t.Msg = "retry"
                if (-not $autoYes) {
                    $decision = Confirm-Task $t
                    if ($decision -eq "quit") { break }
                    if ($decision -eq "no") { $t.Status = "skipped"; $t.Msg = "skipped by user"; Save-Progress; continue }
                    if ($decision -eq "always") { $autoYes = $true }
                }
                Run-Task $t
            }
        }
        "3" {
            while ($true) {
                Show-Dashboard
                Write-Host "  TOGGLE TASKS — enter ID to flip Pending↔Skipped, or [0] to go back"
                $tid = Read-Host "  Task ID"
                if ($tid -eq "0") { break }
                $parsedId = 0
                if (-not [int]::TryParse($tid, [ref]$parsedId)) {
                    Write-Host "  '$tid' is not a valid task ID." -ForegroundColor Red
                    Start-Sleep -Seconds 1
                    continue
                }
                $match = $Tasks | Where-Object { $_.Id -eq $parsedId }
                if ($match) {
                    Toggle-Task $parsedId
                    Save-Progress
                }
                else {
                    Write-Host "  No task with ID $parsedId." -ForegroundColor Red
                    Start-Sleep -Seconds 1
                }
            }
        }
        "4" { Reset-All }
        "5" { Run-Visualization }
        "6" {
            Save-Progress
            Write-Host ""
            Write-Host $Line -ForegroundColor Cyan
            Write-Host "  Progress saved to $ProgressFile" -ForegroundColor Cyan
            Write-Host "  Goodbye!" -ForegroundColor Cyan
            Write-Host $Line -ForegroundColor Cyan
            exit
        }
    }
}