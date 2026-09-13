"""Check the batch-size confound at the two already evaluated boundary points."""
import datetime
import json
import time
from boundary import OUT, CHECKPOINT, torch, LogKVLM, make_batch, save


@torch.no_grad()
def main():
    if (OUT / 'batch_check.json').exists():
        raise FileExistsError(OUT / 'batch_check.json')
    baseline = json.loads((OUT / 'answers.json').read_text())
    torch.set_num_threads(1);torch.set_float32_matmul_precision('high')
    model = LogKVLM.from_pretrained(CHECKPOINT).to(device='cuda',dtype=torch.bfloat16).eval()
    result = dict(complete=False,started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  purpose='Swap dynamic batch sizes at T4076/4077 to check the shape confound',results=[])
    start = time.monotonic()
    for T,batch in ((4076,127),(4077,128)):
        generator = torch.Generator().manual_seed(12345)
        predictions,targets = [],[]
        for i in range(0,256,batch):
            x,y = make_batch(T,min(batch,256-i),generator=generator,device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits,_ = model.step(x,None)
            predictions += [''.join(map(str,row)) for row in logits[:,-10:].float().argmax(-1).cpu().tolist()]
            targets += [''.join(map(str,row)) for row in y[:,-10:].cpu().tolist()]
        assert targets == baseline['targets']
        correct = torch.tensor([[p==t for p,t in zip(pred,target)] for pred,target in zip(predictions,targets)])
        row = dict(T=T,batch=batch,n=256,token_correct=int(correct.sum()),
                   string_correct=int(correct.all(1).sum()),digit_correct=correct.sum(0).tolist(),
                   predictions=predictions,identical_predictions=predictions==baseline['predictions'][str(T)])
        result['results'].append(row)
        print(T,batch,row['string_correct'],row['identical_predictions'],flush=True)
    result.update(complete=True,elapsed_sec=time.monotonic()-start,
                  finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save('batch_check.json',result)


if __name__ == '__main__':
    main()
