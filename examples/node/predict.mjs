import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

/** Convert a DOM/JS UTF-16 selection span to the model's Unicode code point span. */
export function codePointTarget(text, start, end) {
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end <= start || end > text.length) throw new Error('Invalid UTF-16 span');
  for (const offset of [start, end]) {
    if (offset > 0 && offset < text.length && /[\uD800-\uDBFF]/.test(text[offset-1]) && /[\uDC00-\uDFFF]/.test(text[offset])) throw new Error('Span splits a surrogate pair');
  }
  return { start: Array.from(text.slice(0,start)).length, end: Array.from(text.slice(0,end)).length, text: text.slice(start,end) };
}

export function prompt(example, metadata, maskToken) {
  const chars=Array.from(example.text), {start,end,text}=example.target;
  if (!Number.isInteger(start)||!Number.isInteger(end)||start<0||end<=start||end>chars.length||chars.slice(start,end).join('')!==text) throw new Error('Invalid target');
  if (!example.candidates.length||example.candidates.length>127) throw new Error('Need 1..127 candidates');
  if (new Set(example.candidates.map(c=>c.id)).size!==example.candidates.length) throw new Error('Duplicate candidate ids');
  for (const content of [example.text,...example.candidates.map(c=>c.definition)]) {
    if (typeof content!=='string'||content.includes('*')||content.includes(maskToken)) throw new Error('Unsafe literal marker in prompt');
  }
  const marked=chars.slice(0,start).join('')+'*'+text+'*'+chars.slice(end).join('');
  const choices=example.candidates.map((candidate,i)=>`${metadata.letters[i]}. ${candidate.definition}`);
  choices.push(`${metadata.letters[metadata.nota_index]}. ${metadata.none_text}`);
  return `${marked}\n${choices.join('\n')}\n${metadata.separator} ${maskToken}`;
}

export function decode(logits, example, metadata) {
  const cal=metadata.calibration??{}, temperature=cal.temperature??1;
  const scores=[...logits.slice(0,example.candidates.length),logits[metadata.nota_index]].map(x=>x/temperature);
  const maximum=Math.max(...scores), exp=scores.map(x=>Math.exp(x-maximum)), total=exp.reduce((a,b)=>a+b,0), probs=exp.map(x=>x/total);
  const order=probs.map((p,i)=>({p,i})).sort((a,b)=>b.p-a.p||a.i-b.i), best=order[0];
  const uncertain=best.p<(cal.min_probability??0)||best.p-order[1].p<(cal.min_margin??0);
  const reason=uncertain?'uncertain':best.i===example.candidates.length?'none_fits':'selected';
  return {example_id:example.id,selected_id:reason==='selected'?example.candidates[best.i].id:null,reason,
    probabilities:Object.fromEntries(example.candidates.map((c,i)=>[c.id,probs[i]])),none_probability:probs.at(-1),
    model_revision:metadata.model_revision??'local',calibration_version:cal.version??'uncalibrated'};
}

export async function predict(directory, examples, {quantized=false}={}) {
  const {AutoTokenizer,env}=await import('@huggingface/transformers');
  const ort=await import('onnxruntime-node');
  env.allowRemoteModels=false;
  const metadata=JSON.parse(await fs.readFile(path.join(directory,'model_metadata.json'),'utf8'));
  try {metadata.calibration=JSON.parse(await fs.readFile(path.join(directory,'calibration.json'),'utf8'));} catch(e) {if(e.code!=='ENOENT')throw e;}
  const tokenizer=await AutoTokenizer.from_pretrained(path.resolve(directory),{local_files_only:true});
  const nonempty=examples.filter(e=>e.candidates.length);
  const results=new Map(examples.filter(e=>!e.candidates.length).map(e=>[e.id,{example_id:e.id,selected_id:null,reason:'no_candidates',probabilities:{},none_probability:1,model_revision:metadata.model_revision??'local',calibration_version:metadata.calibration?.version??'uncalibrated'}]));
  if(nonempty.length){
    const encoded=tokenizer(nonempty.map(e=>prompt(e,metadata,tokenizer.mask_token)),{padding:true,truncation:false});
    if(encoded.input_ids.dims[1]>metadata.max_length)throw new Error('Prompt exceeds max_length');
    const width=encoded.input_ids.dims[1],positions=nonempty.map((_,i)=>{
      const row=Array.from(encoded.input_ids.data.slice(i*width,(i+1)*width),Number),matches=row.flatMap((x,j)=>x===tokenizer.mask_token_id?[j]:[]);
      if(matches.length!==1)throw new Error('Expected exactly one mask');return BigInt(matches[0]);
    });
    const session=await ort.InferenceSession.create(path.join(directory,'onnx',quantized?'model.int8.onnx':'model.onnx'),{executionProviders:['cpu']});
    try {
      const output=await session.run({input_ids:new ort.Tensor('int64',encoded.input_ids.data,encoded.input_ids.dims),attention_mask:new ort.Tensor('int64',encoded.attention_mask.data,encoded.attention_mask.dims),prediction_positions:new ort.Tensor('int64',BigInt64Array.from(positions),[nonempty.length])});
      nonempty.forEach((e,i)=>results.set(e.id,decode(Array.from(output.logits.data.slice(i*128,(i+1)*128)),e,metadata)));
    } finally {await session.release();}
  }
  return examples.map(e=>results.get(e.id));
}

if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const [directory,file]=process.argv.slice(2);
  if(!directory||!file)throw new Error('Usage: node predict.mjs MODEL_DIRECTORY INPUT.jsonl');
  const examples=(await fs.readFile(file,'utf8')).split('\n').filter(l=>l.trim()).map(l=>JSON.parse(l));
  for(const result of await predict(directory,examples)) console.log(JSON.stringify(result));
}
