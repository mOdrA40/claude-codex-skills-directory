#!/usr/bin/env pwsh
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

param(
  [string]$ModuleRoot = ".",
  [switch]$Race,
  [switch]$Bench
)

Push-Location $ModuleRoot
try {
  Write-Host "== go version ==" -ForegroundColor Cyan
  go version

  Write-Host "== go env (GOMOD/GOVERSION) ==" -ForegroundColor Cyan
  go env GOMOD GOVERSION

  Write-Host "== go mod verify ==" -ForegroundColor Cyan
  go mod verify

  Write-Host "== format ==" -ForegroundColor Cyan
  $dirs = @(go list -f '{{.Dir}}' ./...)
  if ($dirs.Count -gt 0) {
    gofmt -w $dirs
  }

  Write-Host "== vet ==" -ForegroundColor Cyan
  go vet ./...

  Write-Host "== test ==" -ForegroundColor Cyan
  go test ./...

  if ($Race) {
    Write-Host "== test -race ==" -ForegroundColor Cyan
    go test -race ./...
  }

  if ($Bench) {
    Write-Host "== bench ==" -ForegroundColor Cyan
    go test -bench . -benchmem ./...
  }

  $golangci = Get-Command golangci-lint -ErrorAction SilentlyContinue
  if ($null -ne $golangci) {
    Write-Host "== golangci-lint ==" -ForegroundColor Cyan
    golangci-lint run
  } else {
    Write-Host "== golangci-lint not found (skipping) ==" -ForegroundColor Yellow
  }
} finally {
  Pop-Location
}
