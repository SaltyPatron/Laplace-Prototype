import fs from 'node:fs';
import crypto from 'node:crypto';
import { once } from 'node:events';

const [modelPath, reportPath] = process.argv.slice(2);
if (!modelPath || !reportPath) throw new Error('usage: node inspect-gguf.mjs MODEL REPORT.json');
const stat = fs.statSync(modelPath);
const fd = fs.openSync(modelPath, 'r');
const header = Buffer.alloc(Math.min(stat.size, 32 * 1024 * 1024));
fs.readSync(fd, header, 0, header.length, 0);
let pos = 0;
const take = n => {
  if (!Number.isSafeInteger(n) || n < 0 || pos + n > header.length) throw new Error('metadata inspection bound exceeded');
  const start = pos; pos += n; return start;
};
const u32 = () => header.readUInt32LE(take(4));
const u64 = () => Number(header.readBigUInt64LE(take(8)));
const str = () => { const n = u64(); return header.toString('utf8', take(n), pos); };
function value(type) {
  switch (type) {
    case 0: return header.readUInt8(take(1));
    case 1: return header.readInt8(take(1));
    case 2: return header.readUInt16LE(take(2));
    case 3: return header.readInt16LE(take(2));
    case 4: return u32();
    case 5: return header.readInt32LE(take(4));
    case 6: return header.readFloatLE(take(4));
    case 7: return !!header.readUInt8(take(1));
    case 8: return str();
    case 9: {
      const itemType = u32(), n = u64();
      if (n > 1_000_000) throw new Error('metadata array exceeds inspection bound');
      return Array.from({length:n}, () => value(itemType));
    }
    case 10: return u64();
    case 11: return Number(header.readBigInt64LE(take(8)));
    case 12: return header.readDoubleLE(take(8));
    default: throw new Error(`unknown GGUF metadata type ${type}`);
  }
}
if (header.toString('ascii', take(4), pos) !== 'GGUF') throw new Error('not GGUF');
const version = u32(), tensorCount = u64(), metadataCount = u64();
if (version !== 3 || tensorCount > 100_000 || metadataCount > 100_000) throw new Error('unsupported container or excessive descriptor count');
const metadata = new Map();
for (let i = 0; i < metadataCount; i++) {
  const key = str();
  if (metadata.has(key)) throw new Error(`duplicate metadata key ${key}`);
  metadata.set(key, value(u32()));
}
const tensors = [];
for (let i = 0; i < tensorCount; i++) {
  const name = str(), rank = u32();
  if (rank > 4) throw new Error('unsupported tensor rank');
  const shape = Array.from({length:rank}, u64), type = u32(), offset = u64();
  tensors.push({name, shape, type, offset});
}
const alignment = metadata.get('general.alignment') ?? 32;
const dataStart = Math.ceil(pos / alignment) * alignment;
const tokens = metadata.get('tokenizer.ggml.tokens') ?? [];
const ids = metadata.get('laplace.vocab.blake3_ids') ?? [];
const seen = new Map(), duplicates = [];
tokens.forEach((token, slot) => {
  if (seen.has(token)) duplicates.push({token, slots:[seen.get(token), slot]});
  else seen.set(token, slot);
});
const errors = [];
if (duplicates.length) errors.push('duplicate tokenizer strings');
if (ids.length !== tokens.length || ids.some(id => !/^[0-9a-f]{32}$/i.test(id))) errors.push('BLAKE3 vocabulary map is absent or malformed');
if (tokens.length !== metadata.get('llama.vocab_size')) errors.push('vocabulary size disagrees with architecture metadata');
const chunk = Buffer.alloc(4 * 1024 * 1024);
for (const tensor of tensors) {
  if (tensor.type !== 0) { tensor.values_checked = false; continue; }
  const elements = tensor.shape.reduce((a,b) => a*b, 1), bytes = elements * 4;
  if (!Number.isSafeInteger(elements) || dataStart + tensor.offset + bytes > stat.size) throw new Error(`tensor outside file: ${tensor.name}`);
  const tensorHash=crypto.createHash('sha256');
  const heads=/attn_(q|k|v|output)\.weight$/.test(tensor.name)
    ? metadata.get(/attn_(k|v)\.weight$/.test(tensor.name)?'llama.attention.head_count_kv':'llama.attention.head_count') : 0;
  const byColumns=tensor.name.endsWith('attn_output.weight');
  const width=tensor.shape[0], rows=tensor.shape[1], span=(byColumns?width:rows)/heads;
  const headStats=Array.from({length:heads},()=>({hash:crypto.createHash('sha256'),nonzero:0}));
  let nonzero = 0, nonfinite = 0;
  for (let offset = 0; offset < bytes; offset += chunk.length) {
    const count = Math.min(chunk.length, bytes-offset);
    if (fs.readSync(fd, chunk, 0, count, dataStart + tensor.offset + offset) !== count) throw new Error('truncated tensor');
    tensorHash.update(chunk.subarray(0,count));
    if(heads) {
      if(!byColumns) {
        for(let h=0;h<heads;h++) {
          const first=Math.max(0,h*span*width*4-offset),last=Math.min(count,(h+1)*span*width*4-offset);
          if(first<last)headStats[h].hash.update(chunk.subarray(first,last));
        }
      } else {
        for(let start=0;start<count;) {
          const column=((offset+start)/4)%width,h=Math.floor(column/span);
          const size=Math.min(count-start,((h+1)*span-column)*4);
          headStats[h].hash.update(chunk.subarray(start,start+size));start+=size;
        }
      }
    }
    for (let i = 0; i < count; i += 4) {
      const x = chunk.readFloatLE(i);
      if (!Number.isFinite(x)) nonfinite++;
      if (x !== 0) {
        nonzero++;
        if(heads) {
          const index=(offset+i)/4;
          const h=Math.floor((byColumns?index%width:Math.floor(index/width))/span);
          headStats[h].nonzero++;
        }
      }
    }
  }
  Object.assign(tensor, {elements, nonzero, nonfinite, values_checked:true,sha256:tensorHash.digest('hex')});
  if(heads)tensor.heads=headStats.map((h,index)=>({head:index,nonzero:h.nonzero,sha256:h.hash.digest('hex')}));
  if (nonfinite) errors.push(`nonfinite tensor values: ${tensor.name}`);
}
const tokensToProbe = tokens.flatMap((token, slot) => /^(Ġ|▁)?(king|King|dog|gold|queen|fire|water|cat|sea)$/.test(token)
  ? [{token, slot, blake3_id:ids[slot]}] : []);
const embed = tensors.find(t => t.name === 'token_embd.weight');
const output = tensors.find(t => t.name === 'output.weight');
const norm = tensors.find(t => t.name === 'output_norm.weight');
const blockProjections = tensors.filter(t => /^blk\..*\.(attn_(q|k|v|output)|ffn_(gate|up|down))\.weight$/.test(t.name));
const readoutProbes = [];
if (embed?.type === 0 && output?.type === 0 && norm?.type === 0 &&
    blockProjections.length && blockProjections.every(t => t.values_checked && t.nonzero === 0)) {
  const width = embed.shape[0];
  const read = tensor => {
    const buffer = Buffer.alloc(tensor.elements*4);
    if (fs.readSync(fd, buffer, 0, buffer.length, dataStart+tensor.offset) !== buffer.length) throw new Error('truncated readout');
    return buffer;
  };
  const inputBuffer = read(embed), headBuffer = read(output), normBuffer = read(norm);
  const epsilon = metadata.get('llama.attention.layer_norm_rms_epsilon');
  for (const token of tokensToProbe) {
    const e = new Float64Array(width);
    let sumsq = 0;
    for (let i = 0; i < width; i++) { e[i] = inputBuffer.readFloatLE((token.slot*width+i)*4); sumsq += e[i]*e[i]; }
    const inv = 1/Math.sqrt(sumsq/width+epsilon);
    for (let i = 0; i < width; i++) e[i] *= inv*normBuffer.readFloatLE(i*4);
    const scores = [];
    let nonzero = 0;
    for (let y = 0; y < tokens.length; y++) {
      let score = 0;
      for (let c = 0; c < width; c++) if (e[c] !== 0) score += e[c]*headBuffer.readFloatLE((y*width+c)*4);
      if (score !== 0) nonzero++;
      scores.push({token:tokens[y], slot:y, logit:score});
    }
    scores.sort((a,b) => b.logit-a.logit || a.slot-b.slot);
    readoutProbes.push({...token, nonzero_logits:nonzero, all_logits_tied:nonzero === 0, top:scores.slice(0,5)});
  }
}
fs.closeSync(fd);
const hash = crypto.createHash('sha256');
const stream = fs.createReadStream(modelPath);
stream.on('data', block => hash.update(block));
await once(stream, 'end');
const report = {
  observed_at:new Date().toISOString(), path:modelPath, bytes:stat.size, sha256:hash.digest('hex'),
  version, architecture:metadata.get('general.architecture'), tensor_count:tensorCount,data_offset:dataStart,
  recipe:metadata.has('laplace.export.recipe')?JSON.parse(metadata.get('laplace.export.recipe')):null,
  provenance_sha256:metadata.get('laplace.export.provenance_sha256')??null,
  vocabulary_size:tokens.length, blake3_scheme:metadata.get('laplace.tokenizer.scheme'),
  blake3_slots:ids.length, duplicate_token_strings:duplicates, selected_tokens:tokensToProbe,
  lookup_readout_probes:readoutProbes,
  tensors, errors, container_checks_passed:errors.length === 0,
  behavior_verified:false,
};
fs.writeFileSync(reportPath, JSON.stringify(report, null, 2)+'\n');
console.log(JSON.stringify({report:reportPath, bytes:stat.size, vocabulary:tokens.length, duplicates:duplicates.length, errors}));
if (errors.length) process.exitCode = 2;
