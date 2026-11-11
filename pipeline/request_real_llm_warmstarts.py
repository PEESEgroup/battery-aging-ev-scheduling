"""Request one genuine OpenAI warm start per development regime."""
import json
import os
from pathlib import Path
import sys
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from time import sleep

out=Path(sys.argv[1]).resolve()
inputs=json.loads((out/"proposal_inputs.json").read_text(encoding="utf-8"))
model="gpt-4.1-mini-2025-04-14"
answer={}
for group,state in inputs["states"].items():
    payload={
        "model":model,
        "temperature":0,
        "max_tokens":450,
        "response_format":{"type":"json_object"},
        "messages":[
            {"role":"system","content":
             "You propose one numeric warm start for an EV charging scheduler. The objective prioritizes departure service and reduction of tail unmet energy, then energy cost, emissions, aging, congestion, and peak demand. "
             "lambda_risk increases std and CVaR premiums on grid signals. w_price, w_emission, w_health and w_congestion weight energy price, emissions, aging and congestion. unmet_penalty_per_kwh values service. "
             "capacity_std_factor increases conservative capacity derating. capacity_blend_actual gives more weight to the actual capacity input. "
             "State metrics with a _rel suffix are ratios to the normal-development reference. "
             "Return a JSON object containing exactly the eight keys under bounds, with finite numeric values inside those bounds. This is a single initial point, to be evaluated by the scheduler."},
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
    (out/"real_llm_seeds.partial.json").write_text(json.dumps(answer,indent=2),encoding="utf-8")
    print(f"{group}: {data['model']} genuine API response",flush=True)
(out/"real_llm_seeds.partial.json").replace(out/"real_llm_seeds.json")


