"""Independent post-run Copying review. Standard library only; never uses a GPU.

Preserves the original runner's hash-indexed review and result artifacts.
"""
import csv
import datetime
import json
import math
import struct
from collections import Counter

from common import HERE, ROOT, SOURCE, MODES, PAIRED_TS, now, run_dir, save, sha


def read(path):
    return json.loads(path.read_text())


def tensor_shapes(path):
    with path.open('rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(size))
    return {key: value['shape'] for key, value in header.items() if key != '__metadata__'}


def elapsed(command):
    return (datetime.datetime.fromisoformat(command['finished']) -
            datetime.datetime.fromisoformat(command['started'])).total_seconds()


def flatten(records, key):
    return [row for batch in records for row in batch[key]]


def main():
    dest = HERE / 'results'
    campaign, audit = read(ROOT / 'campaign.json'), read(dest / 'review.json')
    original_review_hash = sha(dest / 'review.json')
    assert campaign['state'] == 'complete-awaiting-review'
    assert campaign['whole_gpu_batch_hours'] < campaign['time_limit_hours'] == 7.5
    assert campaign['gpus'] == [0, 1] and campaign['gpu_limit'] == 2
    assert not campaign['next_stage_queued']
    assert read(HERE / 'campaign.json') == campaign
    assert audit['passed'] and audit['standard_cells'] == 164 and audit['paired_cells'] == 136
    manifest = campaign['source_manifest']
    assert manifest == read(ROOT / 'source_manifest.json') == read(HERE / 'source_manifest.json')
    for relative, expected in manifest['files'].items():
        assert sha(SOURCE / relative) == expected, ('source', relative)
    for relative, expected in campaign['scripts'].items():
        assert sha(HERE / relative) == expected, ('script', relative)
    for relative, expected in audit['result_hashes'].items():
        assert sha(dest / relative) == expected, ('result', relative)
    initial = ROOT / 'initial_model/model.safetensors'
    preflight = read(HERE / 'preflight.json')
    assert preflight == audit['initial_preflight']
    assert preflight['passed'] and preflight['initial_weights_identical']
    assert preflight['initial_rng_identical'] and preflight['repeat_training_bitexact']
    assert sha(initial) == preflight['initial_weights_sha256']
    shapes = tensor_shapes(initial)
    num_params = sum(math.prod(shape) for shape in shapes.values())
    assert num_params == preflight['num_params'] == 5786112
    rows = read(dest / 'metrics.json')
    assert len(rows) == 300
    indexed = {(r['mode'], r['checkpoint'], r['evaluation'], r['T']): r for r in rows}
    assert len(indexed) == 300
    with (dest / 'metrics.csv').open() as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == len(rows)
    for a, b in zip(rows, csv_rows):
        assert a == {key: (value if key in ['mode', 'position_mode', 'checkpoint', 'evaluation']
                          else json.loads(value)) for key, value in b.items()}
    standard_ts = sorted(set(range(1, 15)) | {2**k for k in range(4, 18)} |
                         {3 * 2**(k - 1) for k in range(4, 17)})
    assert len(standard_ts) == 41 and len(PAIRED_TS) == 34
    shared_targets, configs, run_configs, timing, summary, details = {}, {}, {}, {}, [], []
    audited_tokens = 0
    for mode in MODES:
        raw, local, train_dir = ROOT / mode, dest / mode, run_dir(mode)
        for file in raw.iterdir():
            if file.suffix in ['.json', '.log', '.png']:
                assert sha(file) == sha(local / file.name), ('archive', file)
        for name in ['train_log.jsonl', 'run_config.json', 'best.json', 'initialization_audit.json']:
            assert sha(train_dir / name) == sha(local / name), ('train archive', name)
        worker = read(raw / 'worker.json')
        assert worker['state'] == 'complete' and worker['deterministic_training']
        assert [c['stage'] for c in worker['commands']] == ['train', 'best', 'final']
        assert all(c['returncode'] == 0 for c in worker['commands'])
        init = read(train_dir / 'initialization_audit.json')
        assert init['passed'] and init['initial_weights_identical'] and init['deterministic_algorithms']
        assert init['cublas_workspace_config'] == ':4096:8'
        configs[mode] = read(train_dir / 'model/config.json')
        run_configs[mode] = read(train_dir / 'run_config.json')
        rc = run_configs[mode]
        expected = dict(d_model=512, num_heads=8, d_ff=1024, num_layers=2, chunk_size=4,
                        steps=50000, batch_size=64, grad_accum=1, max_t=2028,
                        seed=0, lr=0.0003, warmup=1000, num_params=num_params,
                        phase_emb=False, self_slot=True, compressor_rope=False,
                        gated_attention=True, aligned_rope=(mode == 'aligned'),
                        retrieval_rope=(mode == 'local-control'), aligned_rope_scale=1.0,
                        retrieval_rope_scale=1.0, level_decay_scale=1.0)
        assert all(rc[k] == value for k, value in expected.items())
        train = [json.loads(line) for line in (train_dir / 'train_log.jsonl').read_text().splitlines()]
        intervals = [r for r in train if 'loss' in r]
        assert [r['step'] for r in intervals] == list(range(100, 50001, 100))
        assert all(math.isfinite(r[k]) for r in intervals for k in
                   ['loss', 'ema_loss', 'token_acc', 'string_acc', 'lr', 'elapsed_sec'])
        assert all(0 <= r['string_acc'] <= r['token_acc'] <= 1 for r in intervals)
        assert all(a['elapsed_sec'] < b['elapsed_sec'] for a, b in zip(intervals, intervals[1:]))
        selected = max(intervals, key=lambda r: (r['string_acc'], r['token_acc'], -r['ema_loss']))
        best = read(train_dir / 'best.json')
        assert best == {key: selected[key] for key in best}
        assert [r['step'] for r in train if 'quick_eval' in r] == list(range(5000, 50001, 5000))
        timing[mode] = dict(train_log_seconds=intervals[-1]['elapsed_sec'],
                            train_process_seconds=elapsed(worker['commands'][0]),
                            evaluation_seconds=sum(elapsed(c) for c in worker['commands'][1:]),
                            best=best, final_training_interval=intervals[-1],
                            last_100_intervals_string_range=[min(r['string_acc'] for r in intervals[-100:]),
                                                             max(r['string_acc'] for r in intervals[-100:])])
        for cp, folder in [('best', 'model_best'), ('final', 'model')]:
            weights = train_dir / folder / 'model.safetensors'
            expected_sha = audit['weights'][f'{mode}/{cp}']
            assert sha(weights) == worker['weights'][cp] == expected_sha
            assert tensor_shapes(weights) == shapes
            cfg = read(train_dir / folder / 'config.json')
            assert cfg == configs[mode] == read(local / f'config_{cp}.json')
            assert all(cfg[k] == expected[k] for k in expected if k in cfg)
            for kind, ts, n in [('standard', standard_ts, 256), ('paired', PAIRED_TS, 32)]:
                if kind == 'standard':
                    data = read(raw / f'results_{cp}.json')
                    records = read(raw / f'digits_{cp}.json')
                    assert data['precision'] == 'bf16' and data['samples'] == 256
                    assert data['train_max_t'] == 2028
                else:
                    data = read(raw / f'paired_{cp}.json')
                    records = data['records']
                    assert data['seed'] == 20260911
                assert sorted(map(int, data['results'])) == sorted(map(int, records)) == ts
                for t in ts:
                    target = flatten(records[str(t)], 'target')
                    prediction = flatten(records[str(t)], 'prediction')
                    margins = flatten(records[str(t)], 'margin')
                    assert len(target) == len(prediction) == len(margins) == n
                    assert all(len(r) == 10 for r in target + prediction + margins)
                    assert all(type(v) is int and 1 <= v <= 8 for row in target for v in row)
                    assert all(type(v) is int and 0 <= v <= 9 for row in prediction for v in row)
                    assert all(math.isfinite(v) for row in margins for v in row)
                    target_key = (kind, t) if kind == 'standard' else (kind, 'shared')
                    assert shared_targets.setdefault(target_key, target) == target
                    # A tied correct logit can have zero margin and either argmax result.
                    assert all((m >= 0 if a == b else m <= 0)
                               for tr, pr, mr in zip(target, prediction, margins)
                               for a, b, m in zip(tr, pr, mr))
                    digit_errors = [sum(a[i] != b[i] for a, b in zip(target, prediction)) for i in range(10)]
                    correct = sum(a == b for a, b in zip(target, prediction))
                    token_correct = 10 * n - sum(digit_errors)
                    row = indexed[(mode, cp, kind, t)]
                    assert row == dict(mode=mode, position_mode=mode, checkpoint=cp, evaluation=kind,
                                       T=t, n=n, string_correct=correct, token_correct=token_correct,
                                       string_acc=correct/n, token_acc=token_correct/(10*n),
                                       digit_errors=digit_errors)
                    assert data['results'][str(t)] == dict(n=n, token_acc=token_correct/(10*n), string_acc=correct/n)
                    audited_tokens += 10 * n
                    if (kind == 'standard' and t in [3, 4, 5, 16, 48, 2048, 3072, 4096, 6144, 8192, 12288, 131072]) or (
                            kind == 'paired' and t in [49146, 49152, 49157, 65535, 131071, 131072]):
                        next_equal = [sum(a[i] != b[i] and b[i] == a[i+1] for a, b in zip(target, prediction))
                                      for i in range(9)] + [0]
                        previous_equal = [0] + [sum(a[i] != b[i] and b[i] == a[i-1] for a, b in zip(target, prediction))
                                               for i in range(1, 10)]
                        strings = Counter(tuple(pr) for pr in prediction)
                        details.append(dict(**row, unique_predictions=len(strings),
                            most_common_predictions=[dict(prediction=list(pr), count=count)
                                                     for pr, count in strings.most_common(3)],
                            wrong_output_value_counts=dict(Counter(str(b)
                            for tr, pr in zip(target, prediction) for a, b in zip(tr, pr) if a != b)),
                            errors_equal_next_target_by_digit=next_equal,
                            errors_equal_previous_target_by_digit=previous_equal))
                subset = [indexed[(mode, cp, kind, t)] for t in ts]
                summary.append(dict(mode=mode, checkpoint=cp, evaluation=kind, horizons=len(ts),
                    perfect_horizons=[r['T'] for r in subset if r['string_correct'] == n],
                    zero_exact_horizons=[r['T'] for r in subset if not r['string_correct']],
                    first_nonperfect_evaluated_extrapolation=next((r for r in subset if r['T'] > 2028 and r['string_correct'] < n), None),
                    worst_horizons=[dict(T=r['T'], string_correct=r['string_correct'], n=n)
                                    for r in sorted(subset, key=lambda r: (r['string_correct'], r['T']))[:5]]))
    assert {k: v for k, v in configs['aligned'].items() if k not in ['aligned_rope', 'retrieval_rope']} == {
        k: v for k, v in configs['local-control'].items() if k not in ['aligned_rope', 'retrieval_rope']}
    assert {k: v for k, v in run_configs['aligned'].items() if k not in ['run_name', 'aligned_rope', 'retrieval_rope']} == {
        k: v for k, v in run_configs['local-control'].items() if k not in ['run_name', 'aligned_rope', 'retrieval_rope']}
    assert sha(dest / 'review.json') == original_review_hash
    save(dest / 'manual_review.json', dict(passed=True, reviewed=now(), gpu_used=False,
        all_original_hashes_passed=True, source_files_checked=len(manifest['files']),
        original_scripts_checked=len(campaign['scripts']), result_files_checked=len(audit['result_hashes']),
        original_review_sha256=original_review_hash, review_script_sha256=sha(HERE / 'review.py'),
        raw_archive_matches=True, all_checkpoint_shapes_match_initial=True, num_params=num_params,
        audited_cells=len(rows), audited_answer_tokens=audited_tokens,
        standard_cells=164, paired_cells=136, train_intervals_per_mode=500,
        identical_model_dimensions_and_matched_training_config=True,
        campaign_started=campaign['started'], campaign_finished=campaign['finished'],
        whole_gpu_batch_started=campaign['gpu_batch_started'], elapsed_hours=campaign['elapsed_hours'],
        whole_gpu_batch_hours=campaign['whole_gpu_batch_hours'], timing=timing, summary=summary,
        selected_digit_details=details,
        limitations=[
            'Single training seed per mode; deterministic 20-step repeats do not prove 50k or cross-environment repeatability.',
            'Best is selected by training intervals, not held-out extrapolation accuracy.',
            'Standard horizons use different memories; only paired horizons reuse the same 32 memories.',
            'First failure refers to evaluated horizons; this is not a continuous failure-onset sweep.',
            'Wrong output matching a neighboring target can occur by chance with repeated target values.',
            'Count audit uses saved targets, argmax predictions and margins, without rerunning model inference.',
            'Aligned failure does not isolate rotation frequency, compression retention, numerical drift or readout causally.',
            'No 16M evaluation or main merge; this review covers Copying only.']))
    print(json.dumps(dict(passed=True, audited_cells=len(rows), audited_answer_tokens=audited_tokens,
                         timing=timing, whole_gpu_batch_hours=campaign['whole_gpu_batch_hours']), indent=2))


if __name__ == '__main__':
    main()
