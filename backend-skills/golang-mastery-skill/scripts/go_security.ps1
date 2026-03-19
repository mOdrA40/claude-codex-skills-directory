#!/usr/bin/env pwsh
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

param(
  [string]$ModuleRoot = "."
)

Push-Location $ModuleRoot
try {
  Write-Host "== go version ==" -ForegroundColor Cyan
  go version

  Write-Host "== go env (GOMOD/GOVERSION) ==" -ForegroundColor Cyan
  go env GOMOD GOVERSION

  Write-Host "== go mod verify ==" -ForegroundColor Cyan
  go mod verify

  $govulncheck = Get-Command govulncheck -ErrorAction SilentlyContinue
  if ($null -ne $govulncheck) {
    Write-Host "== govulncheck ==" -ForegroundColor Cyan
    govulncheck ./...
  } else {
    Write-Host "== govulncheck not found (skipping) ==" -ForegroundColor Yellow
  }

  $golangci = Get-Command golangci-lint -ErrorAction SilentlyContinue
  if ($null -ne $golangci) {
    Write-Host "== golangci-lint ==" -ForegroundColor Cyan
    golangci-lint run
  } else {
    Write-Host "== golangci-lint not found (skipping) ==" -ForegroundColor Yellow
  }

  $gosec = Get-Command gosec -ErrorAction SilentlyContinue
  if ($null -ne $gosec) {
    Write-Host "== gosec ==" -ForegroundColor Cyan
    gosec ./...
  } else {
    Write-Host "== gosec not found (optional; skipping) ==" -ForegroundColor Yellow
  }

  $staticcheck = Get-Command staticcheck -ErrorAction SilentlyContinue
  if ($null -ne $staticcheck) {
    Write-Host "== staticcheck ==" -ForegroundColor Cyan
    staticcheck ./...
  } else {
    Write-Host "== staticcheck not found (optional; skipping) ==" -ForegroundColor Yellow
  }
} finally {
  Pop-Location
}

