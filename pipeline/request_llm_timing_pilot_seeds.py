"""Request one genuine OpenAI warm start per development regime."""
import json
import os
from pathlib import Path
import sys
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from time import sleep

out=Path(sys.argv[1]).resolve()
states=json.loads((out/"states.json").read_text(encoding="utf-8"))
protocol=json.loads((out/"protocol.json").read_text(encoding="utf-8"))
inputs={"states":{k:v["enhanced_state"] for k,v in states.items()},
        "bounds":protocol["parameters"],"default_weights":protocol["defaults"]}
model="gpt-4.1-mini-2025-04-14"
answer={}
for group,state in inputs["states"].items():
    payload={
        "model":model,
        "temperature":0,
        "max_tokens":450,
        "response_format":{"type":"json_object"},
        "messages":[
            {"role":"system","content":"Propose one numeric warm start for a hard-cap EV rolling scheduler. Primary objective is qualified 2 kW downward delivery at 17:00, recovery at 18:00, matched departure energy and zero incremental SOH loss. Tie breaks favor higher demand served and lower mean SOH loss. lambda_risk multiplies normalized grid-signal standard deviation and CVaR premiums. w_price weights normalized electricity price. unmet_penalty_per_kwh increases the value of serving energy. recovery_reserve_kw withholds that many kW of reference planning capacity at 18:00 while activation can use the full ceiling. A positive recovery-minus-activation price means recovery is more expensive. Return a JSON object of exactly four numeric keys matching bounds, within their stated ranges."},
            {"role":"user","content":json.dumps({"state":state,"bounds":inputs["bounds"],
                                                   "default_weights":inputs["default_weights"]},allow_nan=False)}]}
    req=Request("https://api.openai.com/v1/chat/completions",data=json.dumps(payload).encode(),
                headers={"Content-Type":"application/json","Authorization":"Bearer "+os.environ["OPENAI_API_KEY"]})
    for attempt in range(3):
        try:
            with urlopen(req, timeout=45) as response:
                data=json.load(response)
                req_id=response.headers.get("x-request-id")
            break
        except HTTPError as exc:
            try:
                code=json.load(exc).get("error", {}).get("code")
            except (ValueError, AttributeError):
                code=None
            if exc.code==429 and code=="rate_limit_exceeded" and attempt<2:
                retry=exc.headers.get("Retry-After", "")
                sleep(min(30, float(retry)) if retry.replace(".", "", 1).isdigit() else 2**attempt)
                continue
            raise RuntimeError(f"OpenAI API HTTP {exc.code} ({code or 'unspecified'})") from None
    params=json.loads(data["choices"][0]["message"]["content"])
    assert set(params)==set(inputs["bounds"])
    assert all(type(v) in (float,int) and inputs["bounds"][k][0]<=v<=inputs["bounds"][k][1] for k,v in params.items())
    answer[group]={"parameters":params,"model":data["model"],"response_id":data["id"],
                   "request_id":req_id,"usage":data.get("usage"),"messages":payload["messages"]}
    (out/"llm_seeds.partial.json").write_text(json.dumps(answer,indent=2),encoding="utf-8")
    print(f"{group}: {data['model']} genuine API response",flush=True)
(out/"llm_seeds.partial.json").replace(out/"llm_seeds.json")



