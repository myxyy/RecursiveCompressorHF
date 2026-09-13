"""Independent CPU-only Selective campaign audit; preserve original artifacts."""
import csv
import datetime
import importlib.util
import json
import math
import os
import struct
from collections import Counter

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch

from common import (COPY_ROOT, HERE, INITIAL, MODES, REPO, ROOT, SOURCE,
                    TASK_NAME, TASK_SOURCE, now, run_dir, save, sha)


def read(path):
    return json.loads(path.read_text())


def shapes(path):
    with path.open('rb') as stream:
        header = json.loads(stream.read(struct.unpack('<Q', stream.read(8))[0]))
    return {key: value['shape'] for key, value in header.items() if key != '__metadata__'}


def duration(command):
    return (datetime.datetime.fromisoformat(command['finished']) -
            datetime.datetime.fromisoformat(command['started'])).total_seconds()


def task_metadata(data):
    assert data['task'] == TASK_NAME == 'selective-copying'
    assert data['task_path'] == str(TASK_SOURCE)
    assert data['task_sha256'] == sha(TASK_SOURCE)


def main():
    torch.set_num_threads(1)
    dest = HERE / 'results'
    campaign, original = read(ROOT / 'campaign.json'), read(dest / 'review.json')
    original_review_hash = sha(dest / 'review.json')
    assert campaign == read(HERE / 'campaign.json')
    assert campaign['state'] == 'complete-awaiting-review' and campaign['task'] == TASK_NAME
    assert campaign['gpus'] == [0, 1] and campaign['gpu_limit'] == 2
    assert campaign['whole_gpu_batch_hours'] < campaign['time_limit_hours'] == 7.5
    assert not campaign['next_stage_queued']
    assert original['passed'] and original['standard_cells'] == 132
    assert original['task'] == TASK_NAME and original['scatter_positions_matched']
    copy_review = HERE.parent / 'logkv-aligned-rope-20260913/results/manual_review.json'
    assert sha(copy_review) == campaign['copying_review_sha256'] and read(copy_review)['passed']
    manifest = campaign['source_manifest']
    assert manifest == read(ROOT / 'source_manifest.json') == read(COPY_ROOT / 'source_manifest.json')
    assert campaign['source_root'] == str(SOURCE)
    for relative, digest in manifest['files'].items():
        assert sha(SOURCE / relative) == digest, ('source', relative)
    for relative, digest in campaign['scripts'].items():
        assert sha(HERE / relative) == digest, ('script', relative)
    for relative, digest in campaign['prerequisite_hashes'].items():
        assert sha(HERE / relative) == digest, ('prerequisite', relative)
        assert read(HERE / relative)['passed']
    for relative, digest in original['result_hashes'].items():
        assert sha(dest / relative) == digest, ('result', relative)
    preflight, validation = read(HERE / 'preflight.json'), read(HERE / 'validation.json')
    assert preflight == original['initial_preflight'] and validation == original['model_validation']
    assert sha(REPO / validation['reused_model_validation_path']) == validation['reused_model_validation_sha256']
    assert sha(COPY_ROOT / 'source_manifest.json') == validation['source_manifest_sha256']
    assert preflight['initial_weights_identical'] and preflight['initial_rng_identical']
    assert preflight['repeat_training_bitexact'] and preflight['deterministic_algorithms']
    task_metadata(preflight)
    initial_hash = sha(INITIAL)
    assert initial_hash == campaign['initial_weights_sha256'] == preflight['initial_weights_sha256']
    initial_shapes = shapes(INITIAL)
    num_params = sum(math.prod(shape) for shape in initial_shapes.values())
    assert num_params == preflight['num_params'] == 5786112

    grid = sorted(set(range(1, 15)) | {2**k for k in range(4, 14)} |
                  {3 * 2**(k - 1) for k in range(4, 13)})
    assert len(grid) == 33 and grid[-1] == 8192
    spec = importlib.util.spec_from_file_location('independent_selective_task', TASK_SOURCE)
    task = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(task)
    assert task.TASK_NAME == TASK_NAME
    generator, expected = torch.Generator().manual_seed(12345), {}
    for t in grid:
        size, remaining, batches = max(1, min(256, 2**19 // (t + 20))), 256, []
        while remaining:
            n = min(size, remaining)
            inputs, labels = task.make_batch(t, n, generator=generator)
            positions = ((inputs >= 1) & (inputs <= 8)).nonzero()[:, 1].reshape(n, 10)
            assert torch.equal(inputs.gather(1, positions), labels[:, -10:])
            batches.append(dict(target=labels[:, -10:].tolist(), positions=positions.tolist()))
            remaining -= n
        expected[str(t)] = batches

    rows = read(dest / 'metrics.json')
    assert len(rows) == 132
    indexed = {(r['mode'], r['checkpoint'], r['T']): r for r in rows}
    assert len(indexed) == 132
    with (dest / 'metrics.csv').open() as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == len(rows)
    for a, b in zip(rows, csv_rows):
        assert a == {key: (value if key in ['mode', 'position_mode', 'checkpoint', 'evaluation']
                          else json.loads(value)) for key, value in b.items()}
    timing, configs, run_configs, init_rngs, summary, details, exact_masks = {}, {}, {}, {}, [], [], {}
    digits_audited = 0
    for mode in MODES:
        raw, archived, train_dir = ROOT / mode, dest / mode, run_dir(mode)
        for file in raw.iterdir():
            if file.suffix in ['.json', '.log', '.png']:
                assert sha(file) == sha(archived / file.name), ('archive', file)
        for file in ['train_log.jsonl', 'run_config.json', 'best.json', 'initialization_audit.json']:
            assert sha(train_dir / file) == sha(archived / file), ('train archive', file)
        worker, init = read(raw / 'worker.json'), read(train_dir / 'initialization_audit.json')
        assert worker['state'] == 'complete' and worker['task'] == TASK_NAME
        assert worker['deterministic_training']
        assert [c['stage'] for c in worker['commands']] == ['train', 'best', 'final']
        assert all(c['returncode'] == 0 for c in worker['commands'])
        assert init['passed'] and init['initial_weights_identical'] and init['deterministic_algorithms']
        assert init['cublas_workspace_config'] == ':4096:8'
        assert init['initial_weights_path'] == str(INITIAL) and init['initial_weights_sha256'] == initial_hash
        task_metadata(init)
        init_rngs[mode] = init['rng_sha256']
        configs[mode] = read(train_dir / 'model/config.json')
        run_configs[mode] = read(train_dir / 'run_config.json')
        config_expected = dict(d_model=512, num_heads=8, d_ff=1024, num_layers=2,
            chunk_size=4, steps=50000, batch_size=64, grad_accum=1, max_t=2028,
            seed=0, lr=0.0003, warmup=1000, num_params=num_params, phase_emb=False,
            self_slot=True, compressor_rope=False, gated_attention=True,
            aligned_rope=(mode == 'aligned'), retrieval_rope=(mode == 'local-control'),
            aligned_rope_scale=1.0, retrieval_rope_scale=1.0, level_decay_scale=1.0)
        assert all(run_configs[mode][key] == value for key, value in config_expected.items())
        copy_config = read(COPY_ROOT / 'exp/copying' / train_dir.name / 'run_config.json')
        assert copy_config == run_configs[mode]
        train = [json.loads(line) for line in (train_dir / 'train_log.jsonl').read_text().splitlines()]
        intervals = [r for r in train if 'loss' in r]
        assert [r['step'] for r in intervals] == list(range(100, 50001, 100))
        assert all(math.isfinite(r[key]) for r in intervals for key in
                   ['loss', 'ema_loss', 'token_acc', 'string_acc', 'lr', 'elapsed_sec'])
        assert all(0 <= r['string_acc'] <= r['token_acc'] <= 1 for r in intervals)
        assert all(a['elapsed_sec'] < b['elapsed_sec'] for a, b in zip(intervals, intervals[1:]))
        selected = max(intervals, key=lambda r: (r['string_acc'], r['token_acc'], -r['ema_loss']))
        best = read(train_dir / 'best.json')
        assert best == {key: selected[key] for key in best}
        assert [r['step'] for r in train if 'quick_eval' in r] == list(range(5000, 50001, 5000))
        timing[mode] = dict(train_log_seconds=intervals[-1]['elapsed_sec'],
            train_process_seconds=duration(worker['commands'][0]),
            evaluation_seconds=sum(duration(c) for c in worker['commands'][1:]),
            best=best, final_training_interval=intervals[-1],
            last_100_intervals_string_range=[min(r['string_acc'] for r in intervals[-100:]),
                                             max(r['string_acc'] for r in intervals[-100:])])
        for cp, folder in [('best', 'model_best'), ('final', 'model')]:
            weights = train_dir / folder / 'model.safetensors'
            assert sha(weights) == worker['weights'][cp] == original['weights'][f'{mode}/{cp}']
            assert shapes(weights) == initial_shapes
            cfg = read(train_dir / folder / 'config.json')
            assert cfg == configs[mode] == read(archived / f'config_{cp}.json')
            assert all(cfg[key] == value for key, value in config_expected.items() if key in cfg)
            payload, records = read(raw / f'results_{cp}.json'), read(raw / f'digits_{cp}.json')
            task_metadata(payload)
            assert payload['precision'] == 'bf16' and payload['samples'] == 256 and payload['train_max_t'] == 2028
            evaluation = read(raw / f'evaluation_audit_{cp}.json')
            task_metadata(evaluation)
            assert evaluation['passed'] and evaluation['mode'] == mode and evaluation['checkpoint'] == cp
            assert evaluation['generated_samples_observed'] and not evaluation['extra_random_draws']
            assert evaluation['seed'] == 12345 and evaluation['samples_per_horizon'] == 256
            assert evaluation['token_budget'] == 2**19 and evaluation['chunk_len'] == 8192
            assert sorted(map(int, records)) == sorted(map(int, payload['results'])) == grid == evaluation['horizons']
            for t in grid:
                batches = records[str(t)]
                assert [{key: batch[key] for key in ['target', 'positions']} for batch in batches] == expected[str(t)]
                assert all(batch['memory'] == batch['target'] for batch in batches)
                target, prediction, margins, positions = ([row for batch in batches for row in batch[key]]
                    for key in ['target', 'prediction', 'margin', 'positions'])
                assert len(target) == len(prediction) == len(margins) == len(positions) == 256
                assert all(len(row) == 10 for row in target + prediction + margins + positions)
                assert all(type(v) is int and 1 <= v <= 8 for row in target for v in row)
                assert all(type(v) is int and 0 <= v <= 9 for row in prediction for v in row)
                assert all(math.isfinite(v) for row in margins for v in row)
                assert all(all(type(v) is int and 0 <= v < t + 9 for v in row) and
                           all(a < b for a, b in zip(row, row[1:])) for row in positions)
                assert all((margin >= 0 if a == b else margin <= 0)
                           for tr, pr, mr in zip(target, prediction, margins) for a, b, margin in zip(tr, pr, mr))
                errors = [sum(a[i] != b[i] for a, b in zip(target, prediction)) for i in range(10)]
                exact = [a == b for a, b in zip(target, prediction)]
                exact_masks[(mode, cp, t)] = exact
                strings, tokens = sum(exact), 2560 - sum(errors)
                row = indexed[(mode, cp, t)]
                assert row == dict(mode=mode, position_mode=mode, checkpoint=cp, evaluation='standard',
                    T=t, n=256, token_correct=tokens, string_correct=strings,
                    token_acc=tokens/2560, string_acc=strings/256, digit_errors=errors)
                assert payload['results'][str(t)] == dict(n=256, token_acc=tokens/2560, string_acc=strings/256)
                digits_audited += 2560
                if t in [1, 3, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
                    frequencies = Counter(tuple(p) for p in prediction)
                    details.append(dict(**row, unique_predictions=len(frequencies),
                        most_common_predictions=[dict(prediction=list(p), count=count) for p, count in frequencies.most_common(3)],
                        wrong_output_value_counts=dict(Counter(str(b) for tr, pr in zip(target, prediction)
                                                              for a, b in zip(tr, pr) if a != b)),
                        errors_equal_next_target_by_digit=[sum(a[i] != b[i] and b[i] == a[i+1]
                            for a, b in zip(target, prediction)) for i in range(9)] + [0],
                        errors_equal_previous_target_by_digit=[0] + [sum(a[i] != b[i] and b[i] == a[i-1]
                            for a, b in zip(target, prediction)) for i in range(1, 10)]))
            subset = [indexed[(mode, cp, t)] for t in grid]
            summary.append(dict(mode=mode, checkpoint=cp, horizons=len(grid),
                perfect_horizons=[r['T'] for r in subset if r['string_correct'] == 256],
                zero_exact_horizons=[r['T'] for r in subset if r['string_correct'] == 0],
                first_evaluated_zero_exact=next((r['T'] for r in subset if r['string_correct'] == 0), None)))
    assert len(set(init_rngs.values())) == 1
    assert {k: v for k, v in configs['aligned'].items() if k not in ['aligned_rope', 'retrieval_rope']} == {
        k: v for k, v in configs['local-control'].items() if k not in ['aligned_rope', 'retrieval_rope']}
    assert {k: v for k, v in run_configs['aligned'].items() if k not in ['run_name', 'aligned_rope', 'retrieval_rope']} == {
        k: v for k, v in run_configs['local-control'].items() if k not in ['run_name', 'aligned_rope', 'retrieval_rope']}
    paired_outcomes = []
    for cp in ['best', 'final']:
        for t in grid:
            a, b = exact_masks[('local-control', cp, t)], exact_masks[('aligned', cp, t)]
            paired_outcomes.append(dict(checkpoint=cp, T=t,
                both_correct=sum(x and y for x, y in zip(a, b)),
                only_local_correct=sum(x and not y for x, y in zip(a, b)),
                only_aligned_correct=sum(y and not x for x, y in zip(a, b)),
                both_wrong=sum(not x and not y for x, y in zip(a, b))))
    assert sha(dest / 'review.json') == original_review_hash
    save(dest / 'manual_review.json', dict(passed=True, task=TASK_NAME, reviewed=now(), gpu_used=False,
        all_original_hashes_passed=True, source_files_checked=len(manifest['files']),
        original_scripts_checked=len(campaign['scripts']), prerequisite_files_checked=len(campaign['prerequisite_hashes']),
        result_files_checked=len(original['result_hashes']), original_review_sha256=original_review_hash,
        review_script_sha256=sha(HERE / 'review.py'), raw_archive_matches=True,
        audited_cells=len(rows), audited_answer_tokens=digits_audited, independently_replayed_horizons=len(grid),
        all_scatter_positions_and_memories_replayed=True, seed=12345,
        all_checkpoint_shapes_match_initial=True, num_params=num_params, initial_rng_hashes=init_rngs,
        identical_model_dimensions_and_matched_training_config=True, copying_initial_weights_reused=True,
        train_intervals_per_mode=500, campaign_started=campaign['started'], campaign_finished=campaign['finished'],
        whole_gpu_batch_started=campaign['gpu_batch_started'], elapsed_hours=campaign['elapsed_hours'],
        whole_gpu_batch_hours=campaign['whole_gpu_batch_hours'], timing=timing, summary=summary,
        selected_digit_details=details, matched_mode_exact_outcomes=paired_outcomes,
        limitations=[
            'Single training seed per mode; 20-step deterministic repeats do not establish full-run or cross-environment repeatability.',
            'Best is chosen by training intervals; evaluation checkpoints are not chosen by held-out results.',
            'Memories and scatter positions match across modes/checkpoints at each T, not across T.',
            'Counts use saved targets, predictions and margins without rerunning inference; sample generation is independently replayed on CPU.',
            'Neighbor-value error matches can occur by chance when target digits repeat; these are descriptive, not causal diagnoses.',
            'Zero exact strings in 256 examples does not prove zero success probability or irrecoverable memory loss.',
            'This campaign evaluates Selective Copying through T8192 only; no extra seeds, 16M evaluation or main merge.']))
    print(json.dumps(dict(passed=True, audited_cells=len(rows), audited_answer_tokens=digits_audited,
                         whole_gpu_batch_hours=campaign['whole_gpu_batch_hours'], timing=timing), indent=2))


if __name__ == '__main__':
    main()
