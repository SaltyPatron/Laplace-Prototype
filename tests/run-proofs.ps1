param(
    [Parameter(Mandatory)][string]$Engine,
    [Parameter(Mandatory)][string]$NativeProof,
    [Parameter(Mandatory)][string]$Tier0,
    [Parameter(Mandatory)][string]$Highway,
    [Parameter(Mandatory)][string]$ConnInfo,
    [Parameter(Mandatory)][string]$Scratch,
    [Parameter(Mandatory)][string]$Formats,
    [string]$Psql = 'psql'
)
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Path $Scratch -Force | Out-Null
$Scratch = (Resolve-Path -LiteralPath $Scratch).Path
$prior = @{}
foreach ($name in @('LAPLACE_TIER0','LAPLACE_HIGHWAY','LAPLACE_CONNINFO','LAPLACE_INGEST_ONE','LAPLACE_TRUNKS','LAPLACE_RECIPES','LAPLACE_PROOF_DATA','LAPLACE_DATA')) {
    $prior[$name] = [Environment]::GetEnvironmentVariable($name)
}
function Invoke-Proof([string]$File, [string[]]$Arguments, [string]$Name) {
    $lines = @(& $File @Arguments 2>&1)
    $code = $LASTEXITCODE
    $text = ($lines | ForEach-Object { $_.ToString() }) -join [Environment]::NewLine
    [IO.File]::WriteAllText((Join-Path $Scratch ($Name+'.log')), $text)
    if ($code) { throw "$Name exited ${code}: $($text.Substring([Math]::Max(0,$text.Length-4000)))" }
    return $text
}
function Read-Sql([string]$Query) {
    $rows = @(& $Psql -XAt -v ON_ERROR_STOP=1 --dbname $ConnInfo -c $Query 2>&1)
    if ($LASTEXITCODE) { throw ($rows -join [Environment]::NewLine) }
    return ($rows -join [Environment]::NewLine).Trim()
}
function Test-Refusal([string]$Path, [string]$Name) {
    $env:LAPLACE_HIGHWAY=$Path
    $lines=@(& $Engine ingest --highway -j 2 2>&1)
    $code=$LASTEXITCODE
    $text=($lines | ForEach-Object {$_.ToString()}) -join [Environment]::NewLine
    [IO.File]::WriteAllText((Join-Path $Scratch ($Name+'.log')),$text)
    if (!$code -or $text -notmatch 'rejected before database access') { throw "$Name did not reject the invalid artifact before connecting" }
}
try {
    $env:LAPLACE_TIER0=$Tier0; $env:LAPLACE_HIGHWAY=$Highway; $env:LAPLACE_CONNINFO=$ConnInfo
    $env:LAPLACE_INGEST_ONE='1'; $env:LAPLACE_TRUNKS='1'
    $db=Read-Sql 'SELECT current_database()'
    if (!$db.StartsWith('laplace_proof_')) { throw 'Use an explicitly named laplace_proof_* scratch database' }
    if ((Read-Sql 'SELECT count(*) FROM public.entity') -ne '0') { throw 'Gate A needs an empty substrate; this runner never resets databases' }
    $versions=Read-Sql "SELECT current_setting('server_version_num'),postgis_lib_version(),(SELECT extversion FROM pg_extension WHERE extname='laplace')"
    if (!$versions.StartsWith('18')) { throw 'PostgreSQL 18 is required' }
    Invoke-Proof $NativeProof @('cache',$Tier0,$Highway) 'gate-b-native' | Write-Host
    $regenerated=Join-Path $Scratch 'highway-regenerated.bin'
    $env:LAPLACE_DATA=Join-Path $Scratch 'absent-corpora'
    $env:LAPLACE_RECIPES=Join-Path $Scratch 'absent-recipes'
    Invoke-Proof $Engine @('highway','-o',$regenerated) 'gate-b-build' | Out-Null
    $first=Invoke-Proof $Engine @('highway','--fingerprint','-o',$Highway) 'gate-b-fingerprint'
    $second=Invoke-Proof $Engine @('highway','--fingerprint','-o',$regenerated) 'gate-b-regenerated-fingerprint'
    if ($first -ne $second) { throw 'Frozen-manifest regeneration changed the fingerprint' }
    $env:LAPLACE_CONNINFO='host=127.0.0.1 port=1 connect_timeout=1 dbname=unused'
    Invoke-Proof $Engine @('ingest','--highway','--no-load','-j','4') 'gate-b-admission-offline' | Write-Host
    Test-Refusal (Join-Path $Scratch 'missing.bin') 'gate-b-missing'
    $corrupt=Join-Path $Scratch 'corrupt.bin'
    $file=[IO.File]::OpenRead($Highway); $bad=[IO.File]::Create($corrupt)
    try {
        $file.CopyTo($bad); $bad.Position=110; $bad.WriteByte(255)
    } finally { $file.Dispose(); $bad.Dispose() }
    Test-Refusal $corrupt 'gate-b-corrupt'
    $env:LAPLACE_HIGHWAY=$Highway
    $env:LAPLACE_CONNINFO=$ConnInfo
    $recipes=Join-Path $Scratch 'recipes'
    New-Item -ItemType Directory -Path $recipes -Force | Out-Null
    Copy-Item -Path (Join-Path $PSScriptRoot 'recipes/*') -Destination $recipes -Recurse -Force
    Copy-Item -LiteralPath $Formats -Destination (Join-Path $recipes 'formats') -Recurse -Force
    $env:LAPLACE_RECIPES=$recipes
    $data=Join-Path $Scratch 'data'
    New-Item -ItemType Directory -Path $data -Force | Out-Null
    $fixtures=@(Get-ChildItem (Join-Path $PSScriptRoot 'fixtures') -File | Where-Object Name -ne 'README.md')
    foreach ($file in $fixtures) { Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $data $file.Name) -Force }
    $env:LAPLACE_PROOF_DATA=$data
    $output=Invoke-Proof $Engine @('ingest','-j','4','raw') 'gate-a-ingest-raw'
    Invoke-Proof $Engine @('ingest','-j','4','claims') 'gate-a-ingest-claims' | Out-Null
    $roots=@{}
    foreach ($match in [regex]::Matches($output,'trunk ([0-9a-f]{32})  ([^\r\n]+)')) {
        $roots[[IO.Path]::GetFileName($match.Groups[2].Value.Trim())]=$match.Groups[1].Value
    }
    $bytes=0; $hashes=@{}
    foreach ($file in $fixtures) {
        $hashes[$file.Name]=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($file.Name -eq 'sentences.attested') { continue }
        $root=$roots[$file.Name]; if (!$root) { throw "No content trunk: $($file.Name)" }
        $hex=Read-Sql "SELECT encode(convert_to(laplace_text('$root'::blake3),'UTF8'),'hex')"
        $expected=[IO.File]::ReadAllBytes($file.FullName)
        if ($hex -cne [Convert]::ToHexString($expected).ToLowerInvariant()) { throw "Database-only byte reconstruction differs: $($file.Name)" }
        $bytes+=$expected.Length
    }
    Invoke-Proof $NativeProof @('database',$Tier0,$ConnInfo) 'gate-a-native-database' | Write-Host
    if ((Read-Sql 'SHOW max_prepared_transactions') -ne '0') { throw 'The zero-2PC proof requires max_prepared_transactions=0' }
    if ((Read-Sql "SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='stage' AND c.relkind='r' AND c.relpersistence<>'u'") -ne '0') { throw 'A stage table is logged' }
    $countsSql='SELECT (SELECT count(*) FROM entity),(SELECT count(*) FROM physicality),(SELECT count(*) FROM witness),(SELECT count(*) FROM consensus)'
    $counts=Read-Sql $countsSql
    if ([int]($counts.Split('|')[-1]) -ne 4) { throw 'Four actual language claims were not materialized' }
    $standings="SELECT md5(string_agg(claim::text||rating::text||deviation::text||volatility::text||matches::text,',' ORDER BY claim)) FROM consensus"
    $before=Read-Sql $standings
    $dedupe=Invoke-Proof $Engine @('ingest','-j','4','raw') 'gate-a-dedupe-raw'
    $dedupe+=Invoke-Proof $Engine @('ingest','-j','4','claims') 'gate-a-dedupe-claims'
    if ($dedupe -notmatch '0 entities, 0 paths' -or $dedupe -notmatch '0 IDs checked in 0 rounds') { throw 'File trunks did not prune before descendant probes and COPY' }
    $recorded=0
    foreach ($match in [regex]::Matches($dedupe,'(\d+) files recorded already')) { $recorded += [int]$match.Groups[1].Value }
    if ($recorded -ne $fixtures.Count -or $counts -ne (Read-Sql $countsSql) -or $before -ne (Read-Sql $standings)) { throw 'Dedupe changed the recorded substrate or standings' }
    $receipt=[ordered]@{database=$db;versions=$versions;files=$fixtures.Count-1;bytes=$bytes;counts=$counts;highway_fingerprint=$first;fixture_sha256=$hashes;gate_a='pass';gate_b='pass';gate_c='pending Sprint 3 Wave 3B'}
    $receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Scratch 'receipt.json') -Encoding utf8
    $receipt | ConvertTo-Json -Depth 5
} finally {
    foreach ($name in $prior.Keys) { [Environment]::SetEnvironmentVariable($name,$prior[$name]) }
}
