import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const [engine, mcp, tier0, receiptPath, ...packages] = process.argv.slice(2);
if (!engine || !mcp || !tier0 || !receiptPath || !packages.length)
  throw new Error('usage: node verify-tokenizer-source.mjs ENGINE MCP TIER0 RECEIPT.json MODEL_DIR...');
if (!process.env.LAPLACE_CONNINFO || !process.env.LAPLACE_RECIPES)
  throw new Error('Select LAPLACE_CONNINFO and LAPLACE_RECIPES before admitting source content.');
if (!path.isAbsolute(receiptPath)) throw new Error('Select an absolute receipt path outside the source checkout.');
const checkout = path.resolve(fileURLToPath(new URL('..', import.meta.url)));
if (receiptPath === checkout || receiptPath.startsWith(checkout + path.sep))
  throw new Error('Keep receipts and logs outside the source checkout.');

const inputs = packages.map(directory => path.resolve(directory, 'tokenizer.json'));
if (new Set(inputs).size !== inputs.length) throw new Error('Duplicate tokenizer input.');
for (const file of [engine, mcp, tier0, ...inputs]) fs.accessSync(file, fs.constants.R_OK);
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const enginePath = fs.realpathSync(engine), mcpPath = fs.realpathSync(mcp);
const env = { ...process.env, LAPLACE_TIER0: path.resolve(tier0), LAPLACE_ENGINE: enginePath,
  LAPLACE_RECIPES: fs.realpathSync(process.env.LAPLACE_RECIPES) };
const receipt = {
  startedAt: new Date().toISOString(),
  scope: 'Tokenizer JSON source structure: native admission, exact PostgreSQL reconstruction, and replay.',
  engine: enginePath, mcp: mcpPath, tier0: path.resolve(tier0),
  files: [], checks: {},
};
fs.mkdirSync(path.dirname(receiptPath), { recursive: true });

function run(executable, args, input) {
  const result = spawnSync(executable, args, {
    input, env, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024, timeout: 300_000,
  });
  if (result.error || result.status !== 0)
    throw new Error(`${path.basename(executable)}: ${result.error?.message ?? `exit ${result.status}`}\n${result.stderr?.slice(-4000)}`);
  return result;
}

try {
  // The existing Engine owns parsing, composition, deduplication and persistence.
  // This consumer implements neither a tokenizer decoder nor another ingest path.
  const args = ['ingest', '-d', env.LAPLACE_CONNINFO, '-t', path.resolve(tier0),
    '-r', env.LAPLACE_RECIPES, '-j', '2', ...inputs];
  const admission = run(enginePath, args);
  fs.writeFileSync(`${receiptPath}.admission.log`, admission.stdout + admission.stderr);
  receipt.checks.admission = true;

  const requests = [
    { jsonrpc: '2.0', id: 1, method: 'initialize', params: {
      protocolVersion: '2024-11-05', capabilities: {},
      clientInfo: { name: 'Laplace-Prototype tokenizer source check', version: '1' },
    } },
    { jsonrpc: '2.0', method: 'notifications/initialized' },
    ...inputs.map((file, i) => ({ jsonrpc: '2.0', id: i + 2, method: 'tools/call',
      params: { name: 'file_outline', arguments: { path: file, include_text: true } } })),
  ];
  const readback = run(mcpPath, ['mcp'], requests.map(request => JSON.stringify(request)).join('\n') + '\n');
  const responses = readback.stdout.split('\n').filter(Boolean).map(line => JSON.parse(line));
  const initialized = responses.find(response => response.id === 1);
  if (!initialized?.result || initialized.error) throw new Error('MCP initialization failed.');
  for (let i = 0; i < inputs.length; i++) {
    const response = responses.find(row => row.id === i + 2);
    if (!response?.result || response.error || response.result.isError)
      throw new Error(`Database readback failed for ${inputs[i]}.`);
    const payload = response.result.content.find(block => block.type === 'text');
    if (!payload) throw new Error('Database readback returned no text result.');
    const result = JSON.parse(payload.text);
    if (result.files?.length !== 1) throw new Error(`Expected exactly one persisted file for ${inputs[i]}.`);
    const file = result.files[0], original = fs.readFileSync(inputs[i]);
    const reconstructed = Buffer.from(file.text, 'utf8');
    if (!original.equals(reconstructed)) throw new Error(`Database bytes differ for ${inputs[i]}.`);
    const tokenizer = JSON.parse(file.text), vocab = tokenizer.model?.vocab;
    if (!vocab || typeof vocab !== 'object') throw new Error('Reconstructed source has no vocabulary.');
    const ids = new Set(Array.isArray(vocab) ? vocab.map((_, id) => id) : Object.values(vocab));
    const baseCount = ids.size, added = tokenizer.added_tokens ?? [];
    for (const token of added) ids.add(token.id);
    receipt.files.push({
      input: inputs[i], database: result.database, file: file.file,
      bytes: original.length, reconstructedBytes: reconstructed.length,
      sha256: hash(original), reconstructedSha256: hash(reconstructed), matchesSource: true,
      baseVocabularyEntries: baseCount, addedTokenRecords: added.length,
      addedOnlyIds: ids.size - baseCount, distinctTokenIds: ids.size,
      decoder: tokenizer.decoder, readbackMs: result.ms,
    });
  }
  receipt.checks.exactDatabaseReconstruction = true;

  const replay = run(enginePath, args), replayLog = replay.stdout + replay.stderr;
  fs.writeFileSync(`${receiptPath}.replay.log`, replayLog);
  const skipped = Number(replayLog.match(/(\d+) files recorded already/)?.[1] ?? -1);
  if (skipped !== inputs.length || !/0 entities,\s+0 paths/.test(replayLog))
    throw new Error('Replay did not report every input skipped with zero entity/path inserts.');
  receipt.checks.replayFilesSkipped = skipped;
  receipt.checks.replayZeroEntityAndPathInserts = true;
} catch (error) {
  receipt.error = error.message;
  process.exitCode = 1;
} finally {
  receipt.finishedAt = new Date().toISOString();
  fs.writeFileSync(receiptPath, JSON.stringify(receipt, null, 2) + '\n');
  console.log(JSON.stringify({ receipt: receiptPath, ...receipt }, null, 2));
}
