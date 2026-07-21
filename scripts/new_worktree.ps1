# Create a worktree for a feature: .\scripts\new_worktree.ps1 geo
param([Parameter(Mandatory=$true)][string]$Name)
$Root = git rev-parse --show-toplevel
$Dir = Join-Path (Split-Path $Root -Parent) "sorta-worktrees/$Name"
git -C $Root worktree add -b "feature/$Name" $Dir main
Write-Host "Worktree: $Dir (branch feature/$Name)"
Write-Host "Start the worker session:  cd '$Dir'; claude"
