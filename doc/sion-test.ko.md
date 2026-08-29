# sion_test.py (Needle-in-a-Haystack) LogKV 실행 기록

친구가 제공한 평가 파이프라인 [sion_test.py](../sion_test.py)를 LogKV([doc/logkv.md](logkv.md), 일본어)로
실행한 기록입니다. 브랜치 `logkv-sion-test`. (일본어판: [sion-test.md](sion-test.md))

## 1. 스크립트 개요

- **태스크**: 합성 Needle-in-a-Haystack. `[10, vocab−100)` 범위의 균등 난수 토큰 시퀀스(길이 L)의
  `[0.1L, 0.8L)` 위치에 `key=7777, value∈[1000, 5000)`를 인접하게 삽입하고, 마지막 토큰을
  `query=9999`로 둔 뒤, 마지막 위치의 logits로 value를 맞힙니다(4000-클래스 분류)
- **학습**: 커리큘럼 L=32 → 512, 각 300 스텝, batch 16, AdamW lr 3e-4,
  학습 정확도 >98% (step>100)에서 조기 종료
- **평가**: L ∈ {512, 1024, 2048, 4096, 8192}, 각 20회 시행
- **래퍼** `LanguageModel`: `nn.Embedding(16000, 256)` (기본 init N(0,1))을 `lm_head`와
  tied, LayerNorm → head. 백본은 `(B, L, 256) → (B, L, 256)`인 임의의 모듈

## 2. 변경 사항

`sion_test.py`는 플레이스홀더 `backbone = #MODEL INITIALIZATION` (문법 오류)을 LogKV 백본으로
채운 것 외에는 변경하지 않았습니다:

```python
LogKVBackbone(HIDDEN_DIM)   # LogKVBlock × 2 layers, 4 heads, d_ff 512, chunk_size 4,
                            # phase_emb (levels=2), gated_attention, 고정 log C 레벨 감쇠
```

= LogKV 표준 구성(doc/logkv.md §6.8). 파라미터 1.45M.

추가 스크립트:

- [sion_test_diag.py](../sion_test_diag.py): 원인 분리용 드라이버. `sion_test.py`의 함수를 그대로
  사용하며, init 변형과 스텝 수를 인자로 바꿀 수 있고, 학습 후 가중치를
  `$DATA_DIR/exp/sion_test_{variant}_{steps}.pt`에 저장
- [sion_test_eval.py](../sion_test_eval.py): 저장된 가중치의 긴 문맥 평가. 지표는
  `evaluate_needle_in_a_haystack`와 동일하지만, (a) 시행을 토큰 예산으로 나누어 처리하고,
  (b) 백본을 `step()` 청크 순차 처리(hidden 이월, 한 번에 forward하는 것과 수치적으로 동등)로
  실행하며, (c) norm/head를 마지막 위치에만 적용합니다(예측은 동일). 래퍼의 전체 위치 logits
  `(B, L, 16000)` fp32는 L=2048 × 200회 시행에서 26 GB가 되어 OOM이 나기 때문입니다

## 3. 결과

### 3.1 원래 프로토콜 그대로 (각 페이즈 300 스텝): 모든 길이에서 0%

```
Phase 32 : Loss 233 → 21.7, Train Acc 0%
Phase 512: Loss      → 20.0, Train Acc 0%
Context 512 / 1024 / 2048 / 4096 / 8192: 0.00% (0/20) 전부
```

학습 자체가 이루어지지 않았습니다. 원인은 래퍼의 초기화입니다: tied embedding이 N(0,1)이라
**초기 logit std ≈ 16, 초기 CE ≈ 248** (균등 분포라면 log 16000 ≈ 9.7)이며, lr 3e-4 × 300 스텝으로는
logit 스케일 정규화조차 끝나지 않습니다.

### 3.2 원인 분리 (`sion_test_diag.py`)

| 변형 | 스텝/페이즈 | 결과 |
|---|---|---|
| init 그대로 | 300 | 학습 정확도 0%, 평가 0% (위와 동일) |
| embedding init std 0.02만 변경 | 300 | 초기 CE 9.7로 정상화되지만 loss 8.4에서 미수렴, 평가 0% |
| init 그대로 | 2000 | **512 페이즈가 step 466에서 조기 수렴 (>98%)** |

→ LogKV는 이 태스크를 학습할 수 있습니다. 300 스텝이라는 예산이 이 래퍼에 비해 너무 짧습니다
(init을 고쳐도 300으로는 부족. 약 1500–2500 스텝 필요).

### 3.3 학습 성립 후 긴 문맥 평가 (init 그대로 · 2000 스텝, 200회 시행)

| 문맥 길이 | 512 (학습 길이) | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 | 65536 | 131072 |
|---|---|---|---|---|---|---|---|---|---|
| 정확도 | 75.5% | 72.0% | 72.5% | 62.0% | 56.5% | 43.0% | 25.0% | 22.0% | 6.5% |

- 우연 수준은 1/4000 = 0.025%. 학습 길이의 16배(8192)에서 56.5%, **256배(131072)에서도 6.5% =
  우연의 260배**. 절벽 없는 완만한 감쇠
- 학습 범위 내에서 75%에 머무는 것은 조기 종료 조건이 "1개 배치(16 샘플)에서 >98%"라서 수렴이
  얕기 때문입니다. 스텝을 늘리면 전체가 올라갈 것으로 예상됩니다
- hidden state는 문맥 길이에 대해 로그 크기(레벨 수 × C 요소 × d)이므로, 평가 시 메모리는 문맥 길이와
  무관하게 청크 폭으로 결정됩니다(131072에서도 문제없음)

## 4. 재현 절차

```bash
git checkout logkv-sion-test
uv sync

# 원래 프로토콜 그대로 (300 스텝)
uv run python sion_test.py

# 원인 분리: init 그대로 2000 스텝 + 간이 평가(20회). 가중치를 $DATA_DIR/exp/ 에 저장
uv run python sion_test_diag.py asis 2000 20
# embedding init 0.02 변형
uv run python sion_test_diag.py emb002 300 20

# 저장된 가중치의 긴 문맥 평가 (200회 시행, 131072까지)
uv run python sion_test_eval.py $DATA_DIR/exp/sion_test_asis_2000.pt 200 \
    512,1024,2048,4096,8192,16384,32768,65536,131072
```

환경: RTX 3090 (24 GB) 1장, PyTorch (`uv sync` 환경), `tqdm` 필요
(없으면 `uv run --with tqdm python ...`). 학습은 2000 스텝 × 2 페이즈에 약 3.5분,
131072까지의 평가는 몇 분.

## 5. 보충 (태스크의 성질)

이 태스크는 haystack이 균등 난수 토큰이라, Copying의 blank처럼 "버리기 쉬운 내용"이 없습니다.
LogKV의 압축은 attention pooling으로 두드러진 내용을 남기는 구조이므로, needle의 (key, value)가
난수 토큰의 바다 속에서 압축을 살아남아야 하며, doc/logkv.md의 Copying(1M 토큰까지 완전 해결)보다
어려운 설정입니다. Selective Copying과 마찬가지로 "절벽 없는 감쇠" 프로파일을 보입니다.
