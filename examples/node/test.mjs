import test from 'node:test';
import assert from 'node:assert/strict';
import {codePointTarget,prompt,decode,predict} from './predict.mjs';
import fs from 'node:fs/promises';

const metadata={letters:Array.from({length:128},(_,i)=>String.fromCharCode(0x400+i)),nota_index:127,none_text:'none of the above',separator:'[unused0]',max_length:256};
test('UTF-16 selection offsets and repeated occurrence',()=>{
  assert.deepEqual(codePointTarget('😀 banco banco',9,14),{start:8,end:13,text:'banco'});
  assert.throws(()=>codePointTarget('😀banco',1,3));
  const e={id:'pt',text:'😀 banco banco',target:{start:8,end:13,text:'banco'},candidates:[{id:'seat',definition:'assento'}]};
  assert.match(prompt(e,metadata,'[MASK]'),/^😀 banco \*banco\*/);
});
test('candidate and none abstention',()=>{
  const e={id:'x',candidates:[{id:'a'},{id:'b'}]}, logits=Array(128).fill(0);logits[1]=10;
  assert.equal(decode(logits,e,metadata).selected_id,'b');
  logits[127]=20;
  assert.equal(decode(logits,e,metadata).reason,'none_fits');
  assert.equal(decode(Array(128).fill(0),e,{...metadata,calibration:{min_probability:.99}}).reason,'uncertain');
});
test('exported fixture parity',{skip:!process.env.WSD_ARTIFACT},async()=>{
  const directory=process.env.WSD_ARTIFACT;
  const parity=JSON.parse(await fs.readFile(`${directory}/parity.json`,'utf8'));
  const actual=await predict(directory,parity.examples);
  for(let i=0;i<actual.length;i++){
    assert.equal(actual[i].selected_id,parity.expected_predictions[i].selected_id);
    assert.equal(actual[i].reason,parity.expected_predictions[i].reason);
    for(const [id,p] of Object.entries(actual[i].probabilities)) assert.ok(Math.abs(p-parity.expected_predictions[i].probabilities[id])<1e-4);
  }
});
