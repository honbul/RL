# Synthetic Schema RLVR 최종 보고서

## 1. 실험 목적

4B instruct 모델이 source/target JSON schema와 자연어 요구사항으로부터 실행 가능한 제한형 JSON DSL을 생성하도록 학습했다. 핵심 질문은 **SFT-1K 이후 verifier 기반 GRPO가 SFT-6K보다 Compositional-OOD Problem Pass@1을 높이는가**이다.

## 2. 데이터 생성 방식과 수량

정답 transformation graph → source/target schema → canonical reference DSL → hidden input → expected output → instruction 순서로 파생했다. 4개 shard worker가 후보 20,000건을 생성했고 reference interpreter 및 schema 검증을 통과한 데이터만 사용했다.

- SFT-1K: 1,000 (SFT-6K의 prefix subset)
- SFT-6K: 6,000
- GRPO train: 4,000, prompt당 hidden case 8
- Development: 500
- Test: IID 1,000, Linguistic OOD 500, Compositional OOD 1,000, Structural OOD 500; prompt당 hidden case 32
- 누출 검사: sample ID, graph fingerprint, compositional signature, linguistic template overlap 모두 0

## 3. 모델 및 학습 설정

- 모델: `Qwen/Qwen3-4B-Instruct-2507`, seed 42
- SFT: BF16 LoRA r=16, alpha=32, dropout=0.05, LR 2e-5, 1 epoch
- GRPO: E1 adapter 시작, group size 4, temperature 0.9, top-p 0.95, LR 5e-6, beta 0, DAPO loss, 1 epoch
- GRPO 실행 batch: 8 completions = 2 prompt groups/step, 2,000 optimizer steps, GPU 0 policy / GPU 1 vLLM
- 평가: temperature 0, max completion 256, 동일 evaluator와 동일 4개 split

## 4. E0–E4 결과

| Run | IID exact | Linguistic OOD exact | Compositional OOD exact | Structural OOD exact | Hidden case pass | Field macro F1 | JSON valid | Avg tokens | Peak VRAM MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.961 | 52.8 | 21494.2 |
| E1 | 0.171 | 0.158 | 0.001 | 0.000 | 0.084 | 0.084 | 0.920 | 106.7 | 24373.2 |
| E2 | 1.000 | 1.000 | 0.568 | 0.000 | 0.690 | 0.690 | 0.993 | 99.5 | 24525.2 |
| E3 | 0.461 | 0.482 | 0.084 | 0.000 | 0.262 | 0.267 | 0.946 | 104.5 | 22741.6 |
| E4 | 0.449 | 0.448 | 0.074 | 0.000 | 0.249 | 0.249 | 1.000 | 100.0 | 24135.2 |

주 지표인 Compositional OOD exact는 E2 0.568, E3 0.084, E4 0.074였다. E3/E4가 E2를 넘지 않았으므로 preregistered 조건에 따라 seed 43/44 반복은 실행하지 않았다.

## 5. IID와 OOD 비교

- E2는 IID와 Linguistic OOD에서 1.000을 달성했지만 Compositional OOD는 0.568, Structural OOD는 0이었다.
- E3/E4는 JSON parse rate를 대체로 높게 유지했지만 IID, Linguistic OOD, Compositional OOD 실행 정확도가 E2보다 크게 낮았다.
- 모든 조건이 Structural OOD 0이므로 더 긴 프로그램, 깊은 nesting, 복합 array edge case 일반화는 해결되지 않았다.

## 6. Binary와 Dense reward 비교

Binary E3가 Dense E4보다 Compositional OOD에서 0.084 대 0.074, IID에서 0.461 대 0.449로 소폭 높았다. Dense E4는 rollout reward가 거의 항상 nonzero였지만(15,806/16,000), 이것이 problem-level exact 개선으로 이어지지 않았다. Binary E3도 E2보다 낮아 reward 형태 선택보다 SFT-1K 시작 policy와 on-policy update에 따른 성능 저하가 더 큰 결과였다.

## 7. 주요 raw output 사례

- **E0 failure** `sample_016000` (`dsl_schema_error`) — `outputs/e0_zero_shot/test_compositional_ood/raw_generations.jsonl:1`
  - completion: `{"s16000_class":"positive","s16000_identifier":"c7024","s16000_scaled_amount":5.641670974727246}`
- **E2 success** `sample_016000` — `outputs/e2_sft_6k/test_compositional_ood/raw_generations.jsonl:1`
  - completion: `{"steps":[{"from":"s16000_identifier","op":"copy","to":"s16000_identifier"},{"factor":0.5,"from":"s16000_amount","op":"scale","to":"s16000_scaled_amount"},{"condition":{"operator":"gt","path":"s16000_amount","value":10},"op":"conditional","otherwise":{"value":"non_positive"},"then":{"value":"positive"},"to":"s16000_class"}]}`
- **E2 failure** `sample_016001` (`dsl_schema_error`) — `outputs/e2_sft_6k/test_compositional_ood/raw_generations.jsonl:2`
  - completion: `{"steps":[{"op":"coalesce","sources":["s16001_preferred","s16001_fallback"],"to":"s16001_name"},{"op":"concat","delimiter":"-","from":["s16001_prefix","s16001_code"],"to":"s16001_label"},{"op":"extract","from":"s16001_code","to":"s16001_meta.code"},{"op":"extract","from":"s16001_source","to":"s16001_meta.source"}]}`
- **E3 success** `sample_016000` — `outputs/e3_grpo_binary/test_compositional_ood/raw_generations.jsonl:1`
  - completion: `{"steps":[{"from":"s16000_identifier","op":"copy","to":"s16000_identifier"},{"condition":{"operator":"gt","path":"s16000_amount","value":10},"otherwise":{"value":"non_positive"},"then":{"value":"positive"},"to":"s16000_class","op":"conditional"},{"from":"s16000_amount","op":"scale","factor":0.5,"to":"s16000_scaled_amount","offset":0.0}]}`
- **E3 failure** `sample_016001` (`dsl_schema_error`) — `outputs/e3_grpo_binary/test_compositional_ood/raw_generations.jsonl:2`
  - completion: `{"steps":[{"from":"s16001_preferred","op":"coalesce","others":["s16001_fallback"],"to":"s16001_name"},{"from":["s16001_prefix","s16001_code"],"op":"join","separator":"-","to":"s16001_label"},{"from":"s16001_code","to":"s16001_meta.code"},{"from":"s16001_source","to":"s16001_meta.source"}]}`
- **E4 success** `sample_016008` — `outputs/e4_grpo_dense/test_compositional_ood/raw_generations.jsonl:9`
  - completion: `{"steps":[{"from":"s16008_items","op":"map_array","to":"s16008_items","steps":[{"from":"name","op":"copy","to":"label"},{"from":"status","op":"map_enum","mapping":{"on":true,"off":false},"to":"active","default":false},{"from":"value","op":"scale","factor":0.25,"to":"scaled","offset":0.0}]}]}`
- **E4 failure** `sample_016000` (`dsl_schema_error`) — `outputs/e4_grpo_dense/test_compositional_ood/raw_generations.jsonl:1`
  - completion: `{"steps":[{"from":"s16000_identifier","op":"copy","to":"s16000_identifier"},{"from":"s16000_amount","op":"scale","factor":0.5,"to":"s16000_scaled_amount"},{"from":"s16000_amount","op":"conditional","then":{"value":"positive"},"else":{"value":"non_positive"},"condition":{"operator":"gt","value":10},"to":"s16000_class"}]}`

## 8. 실패 유형

| Run | Top failure types across all splits |
|---|---|
| E0 | `dsl_schema_error` 2884, `truncated_output` 91, `json_parse_error` 25 |
| E1 | `dsl_schema_error` 2487, `json_parse_error` 182, `truncated_output` 55, `invalid_source_path` 22 |
| E2 | `dsl_schema_error` 835, `invalid_source_path` 60, `json_parse_error` 17, `target_schema_error` 13 |
| E3 | `dsl_schema_error` 1991, `json_parse_error` 128, `invalid_source_path` 43, `truncated_output` 27 |
| E4 | `dsl_schema_error` 2186, `invalid_source_path` 50, `target_schema_error` 14, `partial_value_mismatch` 2 |

세부 run/split/failure count는 `reports/failure_summary.csv`에 저장했다. E0는 target JSON을 직접 출력하는 DSL schema 오류가 지배적이었다. E3/E4는 parse 가능한 출력 비율은 높았지만 invalid path/schema 또는 partial mismatch가 남아 실행 exact가 낮았다.

## 9. 결론

### **GRPO 이점 없음**

이 실험에서는 추가 SFT 데이터가 verifier 기반 GRPO보다 효과적이었다. E2의 Compositional-OOD Problem Pass@1 0.568에 비해 E3는 0.084, E4는 0.074였고 hidden execution 성능도 같은 방향이었다. 따라서 개선이 단순 형식 준수에 그치지 않았는지를 묻기 전에, GRPO가 강한 SFT-6K 기준선의 실행 정확도를 보존하지 못했다.

인과 해석은 단일 4B 모델, 단일 seed 42, 고정 LoRA/GRPO 설정, 합성 DSL benchmark에 한정된다. 추가 seed는 사전 정의된 개선 조건이 충족되지 않아 생략했다. 실제 데이터·임상 유효성·production readiness에 대한 주장은 하지 않는다.

## 10. 재현 명령

```bash
.venv/bin/python -m src.data.generate --config configs/data.yaml
.venv/bin/python -m src.evaluate --run-dir outputs/e0_zero_shot --config configs/e0_zero_shot.yaml --all-splits
PYTHONHASHSEED=42 .venv/bin/python -m src.train_sft --config configs/sft_1k.yaml
.venv/bin/python -m src.evaluate --run-dir outputs/e1_sft_1k --all-splits --physical-gpu 1
PYTHONHASHSEED=42 .venv/bin/python -m src.train_sft --config configs/sft_6k.yaml
.venv/bin/python -m src.evaluate --run-dir outputs/e2_sft_6k --all-splits --physical-gpu 1
PYTHONHASHSEED=42 .venv/bin/python -m src.train_grpo --config configs/grpo_binary.yaml
.venv/bin/python -m src.evaluate --run-dir outputs/e3_grpo_binary --all-splits --physical-gpu 1
PYTHONHASHSEED=42 .venv/bin/python -m src.train_grpo --config configs/grpo_dense.yaml
.venv/bin/python -m src.evaluate --run-dir outputs/e4_grpo_dense --all-splits --physical-gpu 1
.venv/bin/python -m src.rescore --run-dir outputs/e4_grpo_dense
```

환경의 정확한 Python 패키지는 `requirements-lock.txt`, 실제 hardware/runtime은 `environment.txt`에 기록했다.
