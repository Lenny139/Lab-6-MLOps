$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$candidateArchivePaths = @(
  (Join-Path $projectRoot 'archive'),
  (Join-Path (Split-Path -Parent $projectRoot) 'archive')
)

$archivePath = $candidateArchivePaths | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $archivePath) {
  throw "No se encontró la carpeta archive. Rutas intentadas: $($candidateArchivePaths -join ', ')"
}
$outPath = Join-Path $PSScriptRoot 'dataset.csv'

$fileNames = @(
  'male_players (legacy).csv',
  'female_players (legacy).csv',
  'male_players.csv',
  'female_players.csv'
)

$files = $fileNames |
  ForEach-Object { Join-Path $archivePath $_ } |
  Where-Object { Test-Path $_ }

if (-not $files -or $files.Count -eq 0) {
  throw "No se encontraron CSV de jugadores en: $archivePath"
}

$rows = foreach ($file in $files) { Import-Csv -Path $file }
if (-not $rows) { throw 'No se pudieron cargar datos' }

$sample = $rows | Select-Object -First 1
$columns = $sample.PSObject.Properties.Name

$positionCol = if ($columns -contains 'position') { 'position' } elseif ($columns -contains 'player_positions') { 'player_positions' } else { $null }
$nationalityCol = if ($columns -contains 'nationality') { 'nationality' } elseif ($columns -contains 'nationality_name') { 'nationality_name' } else { $null }

if (-not $positionCol) { throw 'No existe columna position/player_positions' }
if (-not $nationalityCol) { throw 'No existe columna nationality/nationality_name' }

$featureCols = @('age','overall','potential','pace','shooting','passing','dribbling','defending','physic')
$targetCol = 'value_eur'
$requiredCols = $featureCols + @($positionCol, $nationalityCol, $targetCol)

foreach ($c in ($featureCols + @($targetCol))) {
  if (-not ($columns -contains $c)) {
    throw "Falta columna requerida: $c"
  }
}

$seen = New-Object 'System.Collections.Generic.HashSet[string]'
$clean = New-Object 'System.Collections.Generic.List[object]'
$removed = 0

foreach ($r in $rows) {
  $valid = $true
  $vals = [ordered]@{}

  foreach ($c in $requiredCols) {
    $v = [string]$r.$c
    if ([string]::IsNullOrWhiteSpace($v)) {
      $valid = $false
      break
    }
    $vals[$c] = $v.Trim()
  }

  if (-not $valid) {
    $removed++
    continue
  }

  $key = ($requiredCols | ForEach-Object { $vals[$_] }) -join '||'
  if (-not $seen.Add($key)) {
    $removed++
    continue
  }

  $clean.Add([pscustomobject]$vals) | Out-Null
}

if ($clean.Count -eq 0) { throw 'No quedaron filas tras limpieza' }

function Get-SafeName([string]$prefix, [string]$value) {
  $name = ($value -replace '[^A-Za-z0-9]+','_').Trim('_').ToLower()
  if ([string]::IsNullOrWhiteSpace($name)) { $name = 'unknown' }
  return "$prefix$name"
}

$positions = $clean | Select-Object -ExpandProperty $positionCol -Unique | Sort-Object
$nationalities = $clean | Select-Object -ExpandProperty $nationalityCol -Unique | Sort-Object

$positionNameMap = [ordered]@{}
$usedCols = New-Object 'System.Collections.Generic.HashSet[string]'
foreach ($p in $positions) {
  $base = Get-SafeName 'position_' $p
  $name = $base
  $i = 2
  while (-not $usedCols.Add($name)) {
    $name = "${base}_$i"
    $i++
  }
  $positionNameMap[$p] = $name
}

$nationalityNameMap = [ordered]@{}
foreach ($n in $nationalities) {
  $base = Get-SafeName 'nationality_' $n
  $name = $base
  $i = 2
  while (-not $usedCols.Add($name)) {
    $name = "${base}_$i"
    $i++
  }
  $nationalityNameMap[$n] = $name
}

$final = foreach ($r in $clean) {
  $o = [ordered]@{}

  foreach ($c in $featureCols) { $o[$c] = $r.$c }

  foreach ($p in $positions) {
    $col = $positionNameMap[$p]
    $o[$col] = if ($r.$positionCol -eq $p) { 1 } else { 0 }
  }

  foreach ($n in $nationalities) {
    $col = $nationalityNameMap[$n]
    $o[$col] = if ($r.$nationalityCol -eq $n) { 1 } else { 0 }
  }

  $o[$targetCol] = $r.$targetCol
  [pscustomobject]$o
}

$final | Export-Csv -Path $outPath -NoTypeInformation -Encoding UTF8

$headers = (Import-Csv -Path $outPath | Select-Object -First 1).PSObject.Properties.Name
Write-Output "Dataset generado: $outPath"
Write-Output "Filas: $($final.Count)"
Write-Output "Columnas: $($headers.Count)"
Write-Output "Última columna: $($headers[-1])"
Write-Output "Eliminadas por nulos/duplicados: $removed"
