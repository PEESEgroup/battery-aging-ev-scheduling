"""Send only frozen development aggregates, defaults and parameter bounds."""
import json
import os
from pathlib import Path
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen


output = Path(sys.argv[1])
states = json.loads((output/"api_states.json").read_text(encoding="utf-8"))
protocol = json.loads((output/"protocol.json").read_text(encoding="utf-8"))
model = "gpt-4.1-mini-2025-04-14"
bounds = {"lambda_risk": [.6, .9, 1.2], "recovery_reserve_kw": [6., 8., 10.]}
defaults = dict(zip(bounds, protocol["default"]))
prompt = (
    "Propose one initial setting for a 17:00 2 kW downward EV charging request, recovered at 18:00. "
    "Use only the supplied development summaries. Seek high qualified delivery, retain departure service, "
    "and limit predicted SOH loss. Qualification preserves each vehicle's reference departure energy and "
    "does not increase its predicted SOH loss. CVaR95 of unmet departure energy should stay within "
    "0.5 kWh of the fixed reference both overall and in high-exposure states. "
    "lambda_risk scales the grid-signal standard deviation and CVaR premiums. Higher values can defer "
    "or suppress charging. recovery_reserve_kw withholds reference-plan capacity at 18:00 so activation "
    "can refill curtailed energy using the actual hard cap. If supplied, confidence is the fraction of "
    "four forecast paths that support 2 kW of grid and same-car recovery; high-SOC recovery energy "
    "share includes predicted 18:00 charging plus proportional 2 kWh recovery. SOH losses and rates "
    "are fractions. Select values from the listed discrete parameter bounds. Return a JSON object with "
    "exactly two numeric keys, lambda_risk and recovery_reserve_kw."
)
result = {}
for arm, view in (("llm_original", "original"), ("llm_extended", "extended")):
    result[arm] = {}
    for group, rows in states.items():
        payload = dict(model=model, temperature=0, max_tokens=180, response_format={"type": "json_object"},
                       messages=[{"role": "system", "content": prompt},
                                 {"role": "user", "content": json.dumps(
                                     dict(development_summary=rows[view], default_weights=defaults,
                                          parameter_bounds=bounds), allow_nan=False)}])
        request = Request("https://api.openai.com/v1/chat/completions", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json",
                                   "Authorization": "Bearer "+os.environ["OPENAI_API_KEY"]})
        try:
            with urlopen(request, timeout=45) as response:
                reply = json.load(response)
                request_id = response.headers.get("x-request-id")
        except HTTPError as exc:
            try:
                code = json.load(exc).get("error", {}).get("code")
            except (ValueError, AttributeError):
                code = None
            raise RuntimeError(f"OpenAI API HTTP {exc.code} ({code or 'unspecified'})") from None
        values = json.loads(reply["choices"][0]["message"]["content"])
        assert set(values) == set(bounds)
        assert all(type(values[k]) in (float, int) and values[k] in allowed for k, allowed in bounds.items())
        result[arm][group] = dict(point=[values[k] for k in bounds], model=reply["model"],
                                  request_id=request_id, response_id=reply["id"], messages=payload["messages"])
        (output/"api_seeds.partial.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"{arm} {group}: {reply['model']}", flush=True)
(output/"api_seeds.partial.json").replace(output/"api_seeds.json")
