# Download Shipping Lanes shapefile from GitHub and place as shipping_routes_wgs for UrbEm.
# https://github.com/newzealandpaul/Shipping-Lanes/tree/main/data/Shipping-Lanes-v1
# Run from PDM project root: .\code\scripts\utilities\download_shipping_lanes.ps1

$ErrorActionPreference = "Stop"
$repoUrl = "https://github.com/newzealandpaul/Shipping-Lanes.git"
$cloneDir = Join-Path $PSScriptRoot "..\..\..\temp_shipping_lanes"
$sourceDir = Join-Path $cloneDir "data\Shipping-Lanes-v1"
$targetDir = Join-Path $PSScriptRoot "..\..\..\data\Shipping_Routes"

if (-not (Test-Path $targetDir)) {
    New-Item -ItemType Directory -Path $targetDir -Force
}

if (-not (Test-Path $sourceDir)) {
    if (Test-Path $cloneDir) {
        Remove-Item -Recurse -Force $cloneDir
    }
    Write-Host "Cloning Shipping-Lanes repo..."
    git clone --depth 1 $repoUrl $cloneDir
}

$prefix = "Shipping-Lanes-v1"
$destPrefix = "shipping_routes_wgs"
Get-ChildItem -Path $sourceDir -Filter "$prefix.*" | ForEach-Object {
    $ext = $_.Extension
    $dest = Join-Path $targetDir "$destPrefix$ext"
    Copy-Item $_.FullName -Destination $dest -Force
    Write-Host "Copied $($_.Name) -> Shipping_Routes\$destPrefix$ext"
}

Write-Host "Done. Shapefile is at data\Shipping_Routes\$destPrefix.shp"
if (Test-Path $cloneDir) {
    Remove-Item -Recurse -Force $cloneDir
    Write-Host "Removed temporary clone."
}
