"""Replay the archived composite-score benchmark with a real GPT-6 Sol proposal."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


GROUPS = ("joint", "evening_peak_cluster", "sustained_high_price", "capacity_shortage")
MODEL = "gpt-6-sol"


def load_benchmark(source):
    sys.path.insert(0, str(source))
    os.chdir(source)
    path = source / "llm_top_config_optimization_actual_eval_v6_llm_guided_feedback_bo.py"
    spec = importlib.util.spec_from_file_location("archived_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SubscriptionAgent:
    def __init__(self, module, path):
        self.module, self.path = module, path

    def propose_parameters(self, state_summary, objective_preference):
        from llm_top_config_gemini.prompt_templates import build_prompt
        from llm_top_config_gemini.parse_llm_output import parse_llm_output

        if self.path.exists():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            assert saved["model"] == MODEL
            return saved["proposal"]
        bounds = self.module.PARAM_BOUNDS
        number = {"type": "number"}
        ranges = {"type": "array", "items": number}
        def obj(properties):
            return dict(type="object", properties=properties, required=list(properties), additionalProperties=False)
        schema = obj(dict(parameters=obj(dict.fromkeys(bounds, number)),
                          ranges=obj(dict.fromkeys(bounds, ranges))))
        prompt = build_prompt(state_summary, objective_preference=objective_preference)
        prompt += ("\nReturn only parameters and ranges as required by the response schema. "
                   "Each range contains exactly two numbers. Do not use tools. "
                   "Respect these archived evaluator bounds: " + json.dumps(bounds))
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
        with TemporaryDirectory(prefix="benchmark-gpt6-") as temp:
            work = Path(temp)
            (work/"schema.json").write_text(json.dumps(schema), encoding="utf-8")
            cmd = [env["MATCHED_FEEDBACK_CODEX_EXE"], "exec", "--ephemeral", "--ignore-user-config",
                   "--disable", "shell_tool", "--disable", "plugins", "-c", 'web_search="disabled"',
                   "-c", 'approval_policy="never"', "--sandbox", "read-only", "--skip-git-repo-check",
                   "-C", temp, "-m", MODEL, "--output-schema", str(work/"schema.json"),
                   "-o", str(work/"result.json"), "-"]
            proc = subprocess.run(cmd, input=prompt, cwd=work, env=env, text=True, encoding="utf-8",
                                  capture_output=True, timeout=180)
            if proc.returncode:
                raise RuntimeError("GPT-6 Sol proposal failed: " +
                                   " ".join(line for line in proc.stderr.splitlines() if line.startswith("ERROR:")))
            raw = json.loads((work/"result.json").read_text(encoding="utf-8"))
            session = re.search(r"session id:\s*([0-9a-f-]+)", proc.stderr)
        for key,(lower,upper) in bounds.items():
            x = raw["parameters"][key]
            lo,hi = raw["ranges"][key]
            if not all(np.isfinite([x,lo,hi])) or not lower <= lo <= x <= hi <= upper:
                raise ValueError(f"Out-of-bounds GPT-6 Sol proposal for {key}: {(lo,x,hi)}")
        raw["regime_assessment"] = {}
        raw["reasoning"] = {}
        proposal = parse_llm_output(raw, fallback_params=self.module.BASELINE_PARAMS)
        self.path.write_text(json.dumps(dict(model=MODEL, session_id=session[1] if session else None,
                                            proposal=proposal), indent=2), encoding="utf-8")
        return proposal


def run_group(source, output, group):
    import torch
    torch.set_num_threads(1)
    m = load_benchmark(source)
    m.OUT_DIR = output
    archive = source/"llm_top_config_results_actual_eval_v4"
    history = pd.read_csv(archive/"llm_candidate_history.csv").query("group == @group")
    old_state = json.loads((archive/f"llm_first_pass_{group}.json").read_text(encoding="utf-8"))
    state = m.StateSummary(**old_state["state_summary"])
    inputs = archive/"_actual_eval_inputs"/group/"dev"
    assert len(list(inputs.glob("scenario_*"))) == 8
    optimizer, optimizer_path = m.load_optimization_module()
    optimizer = m.patch_optimizer_for_top_config(optimizer)
    base = history.loc[history.method=="Fixed baseline"].iloc[0].to_dict()
    with (output/f"log_{group}.txt").open("w", encoding="utf-8") as log, redirect_stdout(log):
        audited = m.evaluate_actual_candidate(optimizer,group,"baseline",m.BASELINE_PARAMS,inputs,None,"Fixed baseline")
        for key in ("service_ratio","cvar_unmet_energy_kwh","mean_energy_cost","mean_delta_soh","mean_peak_ev_power_kw"):
            if not np.isclose(audited[key],base[key],rtol=1e-6,atol=1e-7):
                raise RuntimeError(f"{group} {optimizer_path.name}: baseline {key} {audited[key]} != {base[key]}")
        assert np.isclose(m.actual_eval_score(audited,audited),base["actual_score"],rtol=1e-6)
        cfg = m.EvaluationConfig(
            objective_preference=dict(
                primary_goal="generate diverse scenario-aware candidate configurations, then validate them by actual optimization",
                secondary_goals=["propose more than one strategy for service, risk, cost, health, and congestion tradeoffs",
                                 "use feedback from actual optimization results for refinement",
                                 "compare against fixed, rule-based, Random/LHS, and Bayesian optimization baselines under the same budget"],
                baseline_params=m.BASELINE_PARAMS),
            refiner=m.RefinementConfig(n_samples=64, shrink_factor=.70, random_seed=m.RANDOM_SEED,
                                      include_baseline=True, include_initial=True, lhs_sampling=True),
            enable_refinement=True, return_best_among_baseline=False, verbose=True)
        agent = SubscriptionAgent(m,output/f"proposal_{group}.json")
        llm = m.evaluate_llm_configuration(state,m.BASELINE_PARAMS,m.build_proxy_evaluator(state),agent,cfg)
        budget = int((history.method=="LLM one-shot").sum())
        candidates = m.generate_llm_candidates(llm,state)[:budget]
        if len(candidates)!=budget:
            raise RuntimeError(f"{group}: archived one-shot count {budget}, generated {len(candidates)}")
        metric_keys = ("service_ratio", "mean_unmet_energy_kwh", "cvar_unmet_energy_kwh",
                       "mean_energy_cost", "mean_emission_ton", "mean_delta_soh",
                       "mean_peak_ev_power_kw", "mean_optimizer_proxy", "n_scenarios",
                       "cvar_peak_ev_power_kw", "weighted_price_per_kwh", "weighted_emission_per_kwh",
                       "delta_soh_per_kwh", "mean_congestion_exposure")
        baseline_metrics = {key:base[key] for key in metric_keys if key in base}
        one = [m.evaluate_actual_candidate(optimizer,group,name,params,inputs,baseline_metrics,"LLM one-shot")
               for name,params in candidates]
        assert all(np.isclose(r["actual_score"], m.actual_eval_score(r,baseline_metrics)) for r in one)
        feedback = m.run_llm_feedback_refinement(optimizer,group,inputs,baseline_metrics,one,np.random.default_rng(m.RANDOM_SEED))
        assert len(feedback)==int((history.method=="LLM feedback").sum())
        result = pd.concat([history[~history.method.str.startswith("LLM")],pd.DataFrame(one+feedback)],ignore_index=True)
        result.to_csv(output/f"history_{group}.csv",index=False)
    print(f"{group}: {len(one)} one-shot and {len(feedback)-len(one)} additional feedback evaluations complete.",flush=True)


def main(source,output):
    output.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        tasks = [pool.submit(run_group,source,output,g) for g in GROUPS]
        for task in tasks:
            task.result()
    history = pd.concat([pd.read_csv(output/f"history_{g}.csv") for g in GROUPS],ignore_index=True)
    history.to_csv(output/"llm_candidate_history.csv",index=False)
    best = history.loc[history.groupby(["group","method"]).actual_score.idxmin()].copy()
    best["method_label"] = best.method
    best["best_actual_score"] = best.actual_score
    best.to_csv(output/"ablation_summary.csv",index=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("output",type=Path)
    main(*(v.resolve() for v in vars(p.parse_args()).values()))
