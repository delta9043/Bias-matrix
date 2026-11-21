"""Tables 2-3 and the in-text numbers of the paper from the result JSON files -> tables.md, tables.json."""
import json

import numpy as np

from .common import RESULTS_DIR, load_json

TABLE2_COLUMNS = ["방법", "추가량/N", "정확도 (%)", "보정 GAP [95% 구간]", "소수 그룹 (%)", "다수 그룹 (%)"]
TABLE3_COLUMNS = ["방법", "정확도 (%)", "보정 GAP [95% 구간]", "소수 그룹 (%)", "다수 그룹 (%)"]
TABLE2_CAPTION = ("Bias in Bios에서의 BME 결과 표. 표의 상단 부분은 임베딩 편집 기법들의 결과이며, 표의 하단 부분은 학습 데이터 교정 기법들의 결과이다. "
                  "추가량/N은 원본 데이터 수 대비 증강시킨 데이터 수 비율이다. 소수 그룹과 다수 그룹은 각 클래스에서의 해당 그룹 재현율을 평균한 값이다.")
TABLE3_CAPTION = "위해위험 소비자상담 내역 데이터 적용 결과. 표의 상단은 이진 그룹의 경우이며, 표의 하단은 다중 그룹인 경우의 결과이다."


def pct(v, nd=1):
    return f"{100 * v:.{nd}f}"


def num(v, nd=3):
    return f"{v:.{nd}f}"


def gap_cell(r):
    """Bias-corrected RMS gap with its 95% interval."""
    return f"{r['gap_rms_bc']:.3f} [{r['gap_rms_ci'][0]:.3f}, {r['gap_rms_ci'][1]:.3f}]"


def reduction(r, base):
    """Reduction of the corrected gap against the original model of the same run (%)."""
    return f"{100 * (1 - r['gap_rms_bc'] / base['gap_rms_bc']):.0f}"


def acc_drop(r, base):
    """Accuracy loss against the original model of the same run (%p)."""
    return f"{100 * (base['acc'] - r['acc']):.1f}"


def table_row(label, r, block, ours=False, added=None):
    """One table row: formatted cells (after the label) and the raw values."""
    cells = ([] if added is None else [added]) + [pct(r["acc"]), gap_cell(r), pct(r["tpr_minority"]), pct(r["tpr_majority"])]
    values = {k: r[k] for k in ("acc", "gap_rms_bc", "gap_rms_ci", "tpr_minority", "tpr_majority")}
    return {"label": label, "cells": cells, "block": block, "ours": ours, "values": values}


def markdown(table, title):
    lines = [f"## {title}", "", table["caption"], "", "| " + " | ".join(table["columns"]) + " |", "|" + "---|" * len(table["columns"])]
    block = None
    for row in table["rows"]:
        if row["block"] != block and table["blocks"][row["block"]]:  # block title row
            lines.append(f"| *{table['blocks'][row['block']]}* |" + " |" * (len(table["columns"]) - 1))
        block = row["block"]
        lines.append("| " + " | ".join([row["label"]] + row["cells"]) + " |")
    return "\n".join(lines)


def main():
    by = lambda res, key="rows": {r["method"]: r for r in res[key]}  # noqa: E731
    spec, removal, frozen, aug = (load_json(f"{n}.json") for n in ("spectrum", "removal", "frozen", "augmentation"))
    hazard, multi, rare = (load_json(f"{n}.json") for n in ("hazard", "multigroup", "rare_cells"))
    dose = {d: load_json(f"dose_{d}.json") for d in ("bios", "hazard")}
    rm, fz, au, hz, mg, rc = by(removal), by(frozen), by(aug), by(hazard), by(multi), by(rare)

    # Table 2: Bias in Bios (edits of the embeddings, then training-data methods); row names as in the figure legends
    def added(m):
        return f"{au[m]['added_over_n']:+.3f}"

    table2 = {"caption": TABLE2_CAPTION, "columns": TABLE2_COLUMNS, "blocks": {"original": "", "embedding": "임베딩 편집", "data": "학습 데이터 교정"}, "rows": [
        table_row("Original", rm["none"], "original", added="--"),
        table_row("Data matrix X (k=28)", rm["X-top-28"], "embedding", added="--"),
        table_row("LEACE", rm["leace"], "embedding", added="--"),
        table_row("INLP (32회)", rm["inlp-32"], "embedding", added="--"),
        table_row("INLP (직업 보정, 128회)", rm["inlp-class-128"], "embedding", added="--"),
        table_row("BME (Ours, soft λ=0.25)", rm["bme-soft-0.25"], "embedding", True, "--"),
        table_row("BME (Ours, 직교)", rm["bme-orth-28"], "embedding", True, "--"),
        table_row("BME (Ours)", rm["bme"], "embedding", True, "--"),
        table_row("BME (Ours, 모델 고정)", fz["bme-frozen"], "embedding", True, "--"),
        table_row("Reweighing", au["reweighing"], "data", added=added("reweighing")),
        table_row("CDA (all)", au["cda-all"], "data", added=added("cda-all")),
        table_row("CDA (partial)", au["cda-partial"], "data", added=added("cda-partial")),
        table_row("BMA (Ours)", au["bma"], "data", True, added("bma")),
    ]}

    # Table 3: hazard consultations with two and three groups
    labels = [("원본", "none"), ("BME (재학습)", "bme"), ("BME (모델 고정)", "bme-frozen"), ("reweighing", "reweighing"), ("BMA", "bma")]
    table3 = {"caption": TABLE3_CAPTION, "columns": TABLE3_COLUMNS,
              "blocks": {"binary": f"판매방식 2개 그룹 (일반, 비대면), 품목 {hazard['n_classes']}종",
                         "multi": f"판매방식 3개 그룹 (일반, 인터넷, 홈쇼핑), 품목 {multi['n_classes']}종"},
              "rows": [table_row(lab, rows[m], block, m in ("bme", "bme-frozen", "bma")) for block, rows in (("binary", hz), ("multi", mg)) for lab, m in labels]}

    # In-Text Numbers: (key, value, unit, description)
    def draws(d, m, a, name="gap_rms_bc"):
        """Values of one method at one alpha over the subsample draws."""
        return np.array([r[name] for r in dose[d]["rows"] if r["method"] == m and r["alpha"] == a])

    def removed(d, m):
        """Share of the gap added by the dependence (original model, alpha 0 -> 1) that the method removes at alpha = 1 (%)."""
        g0, g1 = draws(d, "none", 0.0).mean(), draws(d, "none", 1.0).mean()
        return f"{100 * (g1 - draws(d, m, 1.0).mean()) / (g1 - g0):.0f}"

    numbers = [
        ("bios.bme.acc_drop", acc_drop(rm["bme"], rm["none"]), "%p", "Bias in Bios, BME (재학습): 정확도 손실"),
        ("bios.bme.gap_reduction", reduction(rm["bme"], rm["none"]), "%", "Bias in Bios, BME (재학습): 보정 GAP 감소율"),
        ("bios.bme.gap", num(rm["bme"]["gap_rms_bc"]), "", "Bias in Bios, BME (사교 편집, 재학습): 보정 GAP"),
        ("bios.bme_orth.gap", num(rm["bme-orth-28"]["gap_rms_bc"]), "", "Bias in Bios, BME (직교 편집, 재학습): 보정 GAP"),
        ("bios.bme_orth.frozen_acc", pct(rm["bme-orth-28"]["frozen_acc"]), "%", "Bias in Bios, BME (직교 편집, 모델 고정): 정확도"),
        ("bios.bme_soft.gap", num(rm["bme-soft-0.25"]["gap_rms_bc"]), "", "Bias in Bios, BME (soft editing λ=0.25, 재학습): 보정 GAP"),
        ("bios.bme_frozen.acc", pct(fz["bme-frozen"]["acc"]), "%", "Bias in Bios, BME (모델 고정): 정확도"),
        ("bios.bme_frozen.acc_drop", acc_drop(fz["bme-frozen"], fz["none"]), "%p", "Bias in Bios, BME (모델 고정): 정확도 손실"),
        ("bios.bme_frozen.gap_reduction", reduction(fz["bme-frozen"], fz["none"]), "%", "Bias in Bios, BME (모델 고정): 보정 GAP 감소율"),
        ("bios.x_top28.acc_drop", acc_drop(rm["X-top-28"], rm["none"]), "%p", "Bias in Bios, 데이터 행렬 X의 상위 28개 제거: 정확도 손실"),
        ("bios.x_top28.gap", num(rm["X-top-28"]["gap_rms_bc"]), "", "Bias in Bios, 데이터 행렬 X의 상위 28개 제거: 보정 GAP"),
        ("bios.k_selected", str(spec["k"]), "개", "Bias in Bios, 순차 순열검정이 선택한 방향의 수 (28개 중)"),
        ("bios.bias_share", pct(spec["bias_share_of_variance"], 2), "%", "Bias in Bios, 임베딩 분산 가운데 클래스 안의 그룹 차이가 차지하는 비율"),
        ("bios.bma.added", num(au["bma"]["added_over_n"]), "N", "Bias in Bios, BMA: 추가한 표본의 양 (원본 대비)"),
        ("bios.bma.acc_drop", acc_drop(au["bma"], au["none"]), "%p", "Bias in Bios, BMA: 정확도 손실"),
        ("bios.bma.gap_reduction", reduction(au["bma"], au["none"]), "%", "Bias in Bios, BMA: 보정 GAP 감소율"),
        ("bios.cda.gap_reduction", reduction(au["cda-all"], au["none"]), "%", "Bias in Bios, CDA (all): 보정 GAP 감소율"),
    ]
    for d, name in (("bios", "Bias in Bios"), ("hazard", "위해위험 상담")):
        numbers += [(f"dose.{d}.none.a0.gap", num(draws(d, "none", 0.0).mean()), "", f"{name}, 편향 강도 α=0, 원본 모델: 보정 GAP"),
                    (f"dose.{d}.none.a1.gap", num(draws(d, "none", 1.0).mean()), "", f"{name}, 편향 강도 α=1, 원본 모델: 보정 GAP")]
        numbers += [(f"dose.{d}.{m}.a1.gap", num(draws(d, m, 1.0).mean()), "", f"{name}, 편향 강도 α=1, {lab}: 보정 GAP")
                    for m, lab in (("leace", "LEACE"), ("inlp", "INLP"), ("bme", "BME"), ("bma", "BMA"))]
        numbers.append((f"dose.{d}.bma.removed", removed(d, "bma"), "%", f"{name}, BMA: 데이터의 의존 때문에 늘어난 보정 GAP 가운데 줄인 비율"))
    numbers += [
        ("dose.bios.none.a0.tpr_minority", pct(draws("bios", "none", 0.0, "tpr_minority").mean()), "%", "Bias in Bios, 편향 강도 α=0, 원본 모델: 소수 그룹 재현율"),
        ("dose.bios.none.a1.tpr_minority", pct(draws("bios", "none", 1.0, "tpr_minority").mean()), "%", "Bias in Bios, 편향 강도 α=1, 원본 모델: 소수 그룹 재현율"),
        ("hazard.n", f"{hazard['n_train'] + hazard['n_test']:,}", "건", "위해위험 상담: 실험에 쓴 상담 건수 (학습 + 테스트)"),
        ("hazard.n_classes", str(hazard["n_classes"]), "종", "위해위험 상담: 품목 대분류의 수"),
        ("hazard.remote_share", pct(hazard["group_share"]), "%", "위해위험 상담: 비대면 판매의 비율 (전체)"),
        ("hazard.remote_share_min", pct(min(hazard["share"]), 0), "%", "위해위험 상담: 품목별 비대면 판매 비율의 최솟값"),
        ("hazard.remote_share_max", pct(max(hazard["share"]), 0), "%", "위해위험 상담: 품목별 비대면 판매 비율의 최댓값"),
        ("hazard.bias_share", pct(hazard["bias_share_of_variance"], 2), "%", "위해위험 상담: 임베딩 분산 가운데 클래스 안의 그룹 차이가 차지하는 비율"),
        ("hazard.none.gap", num(hz["none"]["gap_rms_bc"]), "", "위해위험 상담, 원본 모델: 보정 GAP"),
        ("hazard.none.tpr_minority", pct(hz["none"]["tpr_minority"]), "%", "위해위험 상담, 원본 모델: 비대면 판매 상담의 재현율"),
        ("hazard.none.tpr_majority", pct(hz["none"]["tpr_majority"]), "%", "위해위험 상담, 원본 모델: 일반 판매 상담의 재현율"),
        ("hazard.bme.gap_reduction", reduction(hz["bme"], hz["none"]), "%", "위해위험 상담, BME (재학습): 보정 GAP 감소율"),
        ("hazard.bme_frozen.gap_reduction", reduction(hz["bme-frozen"], hz["none"]), "%", "위해위험 상담, BME (모델 고정): 보정 GAP 감소율"),
        ("hazard.bma.acc_drop", acc_drop(hz["bma"], hz["none"]), "%p", "위해위험 상담, BMA: 정확도 손실"),
        ("hazard.bma.gap_reduction", reduction(hz["bma"], hz["none"]), "%", "위해위험 상담, BMA: 보정 GAP 감소율"),
        ("hazard.bma.tpr_minority_gain", f"{100 * (hz['bma']['tpr_minority'] - hz['none']['tpr_minority']):+.1f}", "%p",
         "위해위험 상담, BMA: 비대면 판매 상담의 재현율 변화"),
        ("hazard.rare.none", pct(rc["none"]["recall_rare"]), "%", "위해위험 상담, 비대면 판매가 드문 품목(학습 데이터에서 10% 미만), 원본 모델: 비대면 판매 상담의 재현율"),
        ("hazard.rare.bme", pct(rc["bme"]["recall_rare"]), "%", "위해위험 상담, 비대면 판매가 드문 품목, BME (재학습): 비대면 판매 상담의 재현율"),
        ("hazard.rare.bma", pct(rc["bma"]["recall_rare"]), "%", "위해위험 상담, 비대면 판매가 드문 품목, BMA: 비대면 판매 상담의 재현율"),
        ("multigroup.none.gap", num(mg["none"]["gap_rms_bc"]), "", "위해위험 상담 (3개 그룹), 원본 모델: 보정 GAP"),
        ("multigroup.bme.gap", num(mg["bme"]["gap_rms_bc"]), "", "위해위험 상담 (3개 그룹), BME (재학습): 보정 GAP"),
        ("multigroup.bma.gap", num(mg["bma"]["gap_rms_bc"]), "", "위해위험 상담 (3개 그룹), BMA: 보정 GAP"),
    ]

    md = ["# 본문의 표와 수치", "", "`results/*.json`에서 `python -m experiments.tables`로 생성함.", "",
          markdown(table2, "표 2. Bias in Bios"), "", markdown(table3, "표 3. 위해위험 소비자상담 내역"), "",
          "## 본문 수치", "", "| 항목 | 값 | 키 |", "|---|---|---|"]
    md += [f"| {text} | {value}{unit} | `{key}` |" for key, value, unit, text in numbers]
    (RESULTS_DIR / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    out = {"table2": table2, "table3": table3, "numbers": {key: {"value": value, "unit": unit, "text": text} for key, value, unit, text in numbers}}
    (RESULTS_DIR / "tables.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
