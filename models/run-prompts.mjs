import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';

const [modelPath, runtimePath, directory] = process.argv.slice(2);
if (!modelPath || !runtimePath || !directory) throw new Error('usage: node run-prompts.mjs MODEL CPU_RUNTIME OUTPUT_DIR');
fs.mkdirSync(directory, {recursive:true});
const runtime = fs.realpathSync(runtimePath);
const env = {...process.env, LD_LIBRARY_PATH:path.dirname(runtime),
  CUDA_VISIBLE_DEVICES:'', HIP_VISIBLE_DEVICES:'', ROCR_VISIBLE_DEVICES:'', GGML_VK_VISIBLE_DEVICES:''};
const version = spawnSync(runtime, ['--version'], {env, encoding:'utf8', timeout:15000});
if (version.status !== 0) throw new Error(`runtime startup failed: ${version.stderr}`);
const words = ['king','dog','gold','queen','fire','water','cat','sea'];
const prompts = [...words.map(word => ({prompt:' '+word, group:'june-prompts'})),
  ...words.map(word => ({prompt:word, group:'bare-word-diagnostic'})),
  {prompt:' King', group:'case-diagnostic'}, {prompt:'King', group:'case-diagnostic'}];
const results = [];
for (let i = 0; i < prompts.length; i++) {
  const {prompt, group} = prompts[i];
  const args = ['-m',modelPath,'-p',prompt,'-n','6','--temp','0','--repeat-penalty','1',
    '-t','6','-c','128','-b','32','--fit','off','--no-warmup',
    '--no-display-prompt','--no-conversation','--simple-io','--verbose-prompt'];
  const begin = Date.now();
  const run = spawnSync(runtime, args, {env, encoding:'utf8', timeout:60000, maxBuffer:8*1024*1024});
  const base = `${String(i+1).padStart(2,'0')}-${group}-${prompt.trim()}`;
  const stdoutPath = path.join(directory,base+'.stdout.txt'), stderrPath = path.join(directory,base+'.stderr.txt');
  fs.writeFileSync(stdoutPath, run.stdout ?? '');
  fs.writeFileSync(stderrPath, run.stderr ?? '');
  const result = {prompt, group, exit_code:run.status, signal:run.signal,
    error:run.error?.message ?? null, elapsed_ms:Date.now()-begin,
    completion:(run.stdout ?? '').trim(), stdout_path:stdoutPath, stderr_path:stderrPath, argv:args};
  results.push(result);
  console.log(JSON.stringify({prompt, exit_code:run.status, completion:result.completion}));
}
const report = {observed_at:new Date().toISOString(), model_path:modelPath, runtime_path:runtime,
  runtime_sha256:crypto.createHash('sha256').update(fs.readFileSync(runtime)).digest('hex'),
  runtime_version:(version.stdout+version.stderr).trim(), temperature:0, requested_tokens:6,
  results, runtime_execution_passed:results.every(r => r.exit_code === 0),
  semantic_results_automatically_scored:false};
const reportPath = path.join(directory,'results.json');
fs.writeFileSync(reportPath, JSON.stringify(report,null,2)+'\n');
if (!report.runtime_execution_passed) process.exitCode=1;
