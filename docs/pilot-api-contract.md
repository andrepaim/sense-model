# Implementation contract

Parent owns annotation_server.py CLI/package integration/tests browser; agents
must coordinate only through these APIs. Python stdlib sqlite/http; optional UI
assets src/wsd/annotation_web/index.html app.js style.css.

Store API src/wsd/annotation_store.py: Store(path); methods import_examples(rows:list[dict],demo=False)->dict;
list_examples(reviewer,language='',status='')->list summaries;
get_example(id,reviewer,adjudicate=False)->dict;
vote(id,reviewer,outcome,gold_ids:list,note='',expected_revision:int=0)->dict;
record_assistance(id,reviewer,result:dict)->dict (enforce first vote);
adjudicate(id,reviewer,outcome,gold_ids,note='',expected_revision:int=0)->dict;
export_gold(include_demo=False)->dict {examples:[Example dicts],manifest:dict};
backup()->dict; stats()->dict.
SQLite connections per method safe ThreadingHTTPServer. Exceptions ValueError
validation; PermissionError blind/gating; conflict RuntimeError maps HTTP409.
Example detail result {example:Example dict WITHOUT gold_ids, revision:int,
my_vote:dict|null, assistance:list, reviews:list only when adjudicate=true,
adjudication:dict|null only adjudication mode, candidate_order:list ids}.
my_vote {outcome,gold_ids,note,initial:{...}, ai_exposed:bool}; after first vote
server may show only this reviewer's assistance. Outcomes selected/none_fits/
insufficient_context/extraction_issue/inventory_issue/skip. Detail must not leak
source gold labels, others votes or adjudication in normal mode. Revision returned
per-example global counter; optimistic expected_revision. Import strips any gold.

HTTP (all JSON): GET /api/session -> {csrf,reviewer,credential:{configured,provider,model}};
POST /api/session {reviewer}; POST /api/key {key,provider,model}; DELETE /api/key.
GET /api/examples?language=&status= -> {examples:[]}; GET /api/example?id= -> Store detail;
GET /api/adjudication?id= -> Store detail adjudicate mode;
POST /api/import {examples:[],demo:false}; POST /api/vote {id,outcome,gold_ids,note,expected_revision};
POST /api/assist {id,refresh:false}; POST /api/adjudicate {id,outcome,gold_ids,note,expected_revision};
GET /api/stats -> store stats;
POST /api/export {include_demo:false} -> export_gold; POST /api/backup -> backup.
Cookie session random token HttpOnly SameSite=Strict, CSRF X-CSRF-Token from session
bootstrap required all mutations. API error {error:string} sanitized. Session key
memory only. HTTP no arbitrary file writes; export/download client Blob.

Jev API src/wsd/jev.py: assist(example:Example,key:str,provider:str='typesafe',model:str='jev-latest')->dict.
No caller-supplied URLs; only typesafe/openrouter allowed. Returned structure
{provider,model,prompt_version,request_fingerprint,choice:str,selected_ids:list,
probabilities:dict candidateID->float,none_probability:float,
insufficient_probability:float,confidence:float|null,usage:dict,created_at}.
Support opaque candidate IDs by sending generated slot keys c0,c1,... plus none/
insufficient, map back. Probability sums checked. No prose generation or fake
explanation. Use urllib timeout30 no redirects max2MB response. Errors safe no
key/body leakage. Agent owns jev tests and optional live status keyless no calls.

Candidate collector src/wsd/pilot.py main(argv): command collect reads documents
JSONL and current local lexicon; outputs unlabeled examples + manifest; optional
audit prepared examples. Sources required; definition language explicit. User
selects local inventory files. Keep current codepoint offsets and full candidate
order; no inferred gold. Demo UI uses bundled existing6 fixture seed with marker
provenance annotation='synthetic', source/demo tracked exports excluded.
