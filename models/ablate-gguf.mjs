import fs from 'node:fs';
import crypto from 'node:crypto';
const [model,reportFile,output,kind]=process.argv.slice(2);
if(!model||!reportFile||!output||!['attention','ffn','both'].includes(kind))
  throw new Error('usage: node ablate-gguf.mjs MODEL INSPECTION.json NEW_MODEL attention|ffn|both');
const report=JSON.parse(fs.readFileSync(reportFile));
if(!report.container_checks_passed||!Number.isSafeInteger(report.data_offset))throw new Error('invalid inspection receipt');
const sha=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
if(sha(model)!==report.sha256)throw new Error('model differs from inspected artifact');
const chosen=report.tensors.filter(t=>kind!=='ffn'&&/\.attn_output\.weight$/.test(t.name)||kind!=='attention'&&/\.ffn_down\.weight$/.test(t.name));
if(!chosen.length||chosen.some(t=>t.type!==0))throw new Error('expected F32 output projections');
fs.copyFileSync(model,output,fs.constants.COPYFILE_EXCL);
const fd=fs.openSync(output,'r+'),zeros=Buffer.alloc(4*1024*1024);
try {for(const tensor of chosen)for(let i=0;i<tensor.elements*4;i+=zeros.length){
  const count=Math.min(zeros.length,tensor.elements*4-i);
  if(fs.writeSync(fd,zeros,0,count,report.data_offset+tensor.offset+i)!==count)throw new Error('incomplete ablation write');
}}finally{fs.closeSync(fd);}
fs.writeFileSync(output+'.ablation.json',JSON.stringify({observed_at:new Date().toISOString(),source:model,
  source_sha256:report.sha256,output,sha256:sha(output),zeroed_tensors:chosen.map(t=>t.name),
  output_head_unchanged:true,refitted:false},null,2)+'\n');
console.log(JSON.stringify({output,zeroed_tensors:chosen.map(t=>t.name)}));
