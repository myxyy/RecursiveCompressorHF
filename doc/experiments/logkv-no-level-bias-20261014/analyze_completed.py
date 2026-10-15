"""Read-only CPU recount of archived campaign; write only new analysis artifacts."""
import json
import math
from pathlib import Path
from common import HERE, ROOT, SOURCE, BASE_ROOT, MODES, run_dir, sha, save, now


def read(path):
    return json.loads(path.read_text())


def main():
    campaign = read(HERE/'campaign.json')
    assert campaign == read(ROOT/'campaign.json')
    assert campaign['state'] == 'complete-awaiting-review'
    checked = {}
    def verify(path, expected):
        assert sha(path) == expected, path
        checked[str(path)] = expected
    for path, digest in campaign['source_manifest']['files'].items():
        verify(SOURCE/path, digest)
    for path, digest in campaign['scripts'].items():
        verify(HERE/path, digest)
    for path, digest in campaign['prerequisite_hashes'].items():
        verify(HERE/path, digest)
    verify(ROOT/'initial_model/model.safetensors', campaign['initial_weights_sha256'])
    verify(BASE_ROOT/'initial_model/model.safetensors', campaign['base_initial_sha256'])
    assert campaign['initial_weights_sha256'] == campaign['base_initial_sha256']
    paired = read(HERE/'results/baseline_comparison.json')
    assert paired['passed'] and paired['identical_memories_and_positions']
    for path, digest in paired['input_hashes'].items():
        verify(Path(path), digest)
    summaries, errors, cells = {}, {}, 0
    baseline = HERE.parent/'logkv-causal-conv-20260914'
    for label, archive, raw in [('zero', HERE, ROOT), ('fixed', baseline, BASE_ROOT)]:
        review = read(archive/'results/review.json'); assert review['passed']
        for path, digest in review['result_hashes'].items():
            verify(archive/'results'/path, digest)
        metrics = read(archive/'results/metrics.json')
        assert len(metrics) == 164
        for mode in MODES:
            result = archive/'results'/mode
            config = read(result/'run_config.json')
            run = raw/'exp'/mode/config['run_name']
            logs = [json.loads(line) for line in (result/'train_log.jsonl').read_text().splitlines() if '"loss"' in line]
            assert [r['step'] for r in logs] == list(range(100, 50001, 100))
            assert all(math.isfinite(r[k]) for r in logs for k in ['loss','ema_loss','token_acc','string_acc'])
            chosen = max(logs, key=lambda r:(r['string_acc'], r['token_acc'], -r['ema_loss']))
            assert chosen['step'] == read(result/'best.json')['step']
            if label == 'zero':
                worker = read(raw/mode/'worker.json')
                assert worker['state'] == 'complete'
                assert all(c['returncode'] == 0 for c in worker['commands'])
                assert worker['first_300_steps_match_benchmark']
            for cp, folder in [('best','model_best'),('final','model')]:
                verify(run/folder/'model.safetensors', review['weights'][f'{mode}/{cp}'])
                cfg = read(run/folder/'config.json')
                assert bool(cfg.get('no_level_bias', False)) == (label == 'zero')
                assert not cfg['learnable_decay'] and not cfg['level_amplify']
                records = read(result/f'digits_{cp}.json')
                rows = [r for r in metrics if r['task'] == mode and r['checkpoint'] == cp]
                summary = dict(best_step=chosen['step'], samples=0, exact=0, digit_errors=[0]*10, perfect_horizons=[], min_margins={})
                bad = []
                for row in rows:
                    t = row['T']; count = exact = correct = 0; digit_errors = [0]*10; margins=[]
                    for batch in records[str(t)]:
                        for target, prediction, margin in zip(batch['target'],batch['prediction'],batch['margin']):
                            assert len(target) == len(prediction) == len(margin) == 10
                            assert all(math.isfinite(v) for v in margin)
                            wrong = [i for i,(a,b) in enumerate(zip(target,prediction)) if a != b]
                            exact += not wrong; correct += 10-len(wrong)
                            for i in wrong: digit_errors[i] += 1
                            if wrong:
                                bad.append(dict(T=t,example=count,target=target,prediction=prediction,
                                    wrong_digits=[i+1 for i in wrong],wrong_margins=[margin[i] for i in wrong]))
                            count += 1; margins.extend(margin)
                    assert count == row['n'] == 256
                    assert exact == row['string_correct'] and correct == row['token_correct']
                    assert exact/count == row['string_acc'] and correct/(count*10) == row['token_acc']
                    assert digit_errors == row['digit_errors']
                    summary['samples'] += count; summary['exact'] += exact
                    summary['digit_errors'] = [a+b for a,b in zip(summary['digit_errors'],digit_errors)]
                    summary['min_margins'][str(t)] = min(margins)
                    if exact == count: summary['perfect_horizons'].append(t)
                    cells += 1
                key=f'{label}/{mode}/{cp}'
                summaries[key] = summary
                if label == 'zero' and mode == 'copying': errors[cp] = bad
    save(HERE/'analysis/copying_errors.json', errors)
    save(HERE/'analysis/summary.json', summaries)
    save(HERE/'analysis/review.json', dict(passed=True,checked_at=now(),cells_recounted=cells,
        hashes_verified=checked,initial_weights_identical=True,checkpoints_verified=8,
        completed=campaign['finished'],execution_hours=campaign['elapsed_hours'],
        including_preflight_hours=campaign['whole_gpu_batch_hours'],new_gpu_runs=False,
        limitations='One seed; checkpoints selected on training intervals; no 16M evaluation of zero-bias weights.'))
    print('Verified hashes, 8 checkpoints, best selection, and all 328 fixed/zero prediction cells.')
    for key,value in summaries.items():
        print(key, 'exact',value['exact'],'/',value['samples'],'errors',value['digit_errors'])


if __name__ == '__main__':
    main()
