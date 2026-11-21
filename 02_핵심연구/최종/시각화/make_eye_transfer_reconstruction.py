"""Render both Eye figures directly from the reconstructed model's saved OOF data."""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, matthews_corrcoef, balanced_accuracy_score

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
PACKAGE = FINAL/'외부 검증'/'최종 모델'
OUTPUT = FINAL/'ㅎ.최종정리파일'
NAVY, BLUE, GREEN = '#24547f', '#287bb8', '#2f7c3b'
INK, MUTED, GRID = '#1c252c', '#59636c', '#d5dce2'
COLORS = ['#ba3b36', NAVY, '#bd691d', '#7850a4', GREEN]

for font in ['Malgun Gothic', '맑은 고딕', 'NanumGothic', 'Apple SD Gothic Neo']:
    try:
        fm.findfont(fm.FontProperties(family=font), fallback_to_default=False)
        plt.rcParams['font.family'] = [font, 'DejaVu Sans']
        break
    except ValueError:
        pass
plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10,
                     'savefig.facecolor': 'white', 'figure.facecolor': 'white'})


def read(name):
    return json.loads((PACKAGE/name).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_evidence():
    frame = pd.read_csv(PACKAGE/'oof_predictions.csv')
    stored, metadata, bootstrap = read('metrics.json'), read('metadata.json'), read('bootstrap_ci.json')
    methods = {
        'source_centered_transfer': ('source_centered_transfer_score', .5),
        'zero_centered_control': ('zero_centered_control_score', .5),
        'frozen_source': ('frozen_source_score', stored['frozen_source']['threshold']),
    }
    y = frame.label.to_numpy(dtype=int)
    actual = {}
    for name, (column, threshold) in methods.items():
        scores = frame[column].to_numpy()
        labels = (scores >= threshold).astype(int)
        cm = confusion_matrix(y, labels, labels=[0, 1])
        actual[name] = {
            'n': len(y), 'correct': int(np.sum(labels == y)), 'accuracy': float(np.mean(labels == y)),
            'MCC': float(matthews_corrcoef(y, labels)), 'ROC_AUC': float(roc_auc_score(y, scores)),
            'balanced_accuracy': float(balanced_accuracy_score(y, labels)),
            'confusion_matrix': cm.tolist(), 'threshold': float(threshold),
        }
        for key, value in actual[name].items():
            if key == 'confusion_matrix':
                assert value == stored[name][key]
            else:
                assert np.isclose(value, stored[name][key], rtol=0, atol=1e-12), (name, key)
    fpr, tpr, _ = roc_curve(y, frame.source_centered_transfer_score, drop_intermediate=False)
    candidate = pd.read_csv(PACKAGE/'data/candidate_audit.csv')
    assert metadata['original_mapping_recovered'] is False
    source_metadata = read('source_model/metadata.json')
    assert source_metadata['evaluation_unique_rows'] == source_metadata['training_rows'] == 673
    assert metadata['source_model_evaluation_unique_rows'] == 673
    assert sha(PACKAGE/'source_model/eye_A_final.joblib') == metadata['source_model_sha256']
    assert actual['source_centered_transfer']['confusion_matrix'] == [[3, 2], [0, 11]]
    return {'metrics': actual, 'metadata': metadata, 'bootstrap': bootstrap,
            'fpr': fpr, 'tpr': tpr, 'groups': frame.feature_group.nunique(),
            'candidate_count': len(candidate), 'mapping_count': len(read('mapping.json')['entries']),
            'negative': int(np.sum(y == 0)), 'positive': int(np.sum(y == 1))}


def draw_cm(ax, evidence, small=False):
    cm = np.array(evidence['metrics']['source_centered_transfer']['confusion_matrix'])
    ax.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max())
    tags = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max()/2 else INK
            ax.text(j, i-.07, str(cm[i, j]), ha='center', va='center',
                    fontsize=23 if small else 27, color=color, weight='bold')
            ax.text(j, i+.22, tags[i][j], ha='center', va='center',
                    fontsize=9 if small else 11, color=color)
    ax.set_xticks([0, 1], ['예측 음성\n(EPA III/IV)', '예측 양성\n(EPA I/II)'])
    ax.set_yticks([0, 1], ['실제 음성\n(EPA III/IV)', '실제 양성\n(EPA I/II)'])
    ax.tick_params(axis='both', length=0, labelsize=9 if small else 11, pad=7)
    ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='minor', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"혼동행렬 · n={evidence['metrics']['source_centered_transfer']['n']} · {evidence['groups']}그룹",
                 fontsize=11 if small else 13, color=NAVY, pad=11)


def draw_roc(ax, evidence, small=False):
    auc = evidence['metrics']['source_centered_transfer']['ROC_AUC']
    ax.plot([0, 1], [0, 1], '--', color='#b7bfc5', linewidth=1)
    ax.plot(evidence['fpr'], evidence['tpr'], color=BLUE, linewidth=2.4,
            label=f'ROC (AUC={auc:.3f})')
    ax.set(xlim=(-.03, 1.04), ylim=(-.03, 1.05),
           xlabel='FPR (1−Specificity)', ylabel='TPR (Sensitivity)')
    ax.set_title('ROC curve · 저장된 OOF 점수', fontsize=11 if small else 13, color=NAVY, pad=11)
    ax.set_xticks([0, .5, 1] if small else np.linspace(0, 1, 6))
    ax.set_yticks([0, .5, 1] if small else np.linspace(0, 1, 6))
    ax.tick_params(labelsize=9 if small else 10)
    ax.grid(color=GRID, linewidth=.7)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', fontsize=8.5 if small else 10, framealpha=.95)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)


def detail_figure(evidence, path):
    m = evidence['metrics']['source_centered_transfer']
    fig = plt.figure(figsize=(16, 6.5), dpi=180)
    fig.text(.5, .955, 'Gettings 1996 — 673행 평가 Eye A 기반 전이학습 재구성',
             ha='center', va='top', fontsize=17, weight='bold', color=NAVY)
    fig.text(.5, .902,
             f"15그룹 교차검증(OOF) · 일치 {m['correct']}/{m['n']} ({m['accuracy']*100:.1f}%) · MCC {m['MCC']:.3f} · AUC {m['ROC_AUC']:.3f}",
             ha='center', va='top', fontsize=13, color=INK)
    draw_cm(fig.add_axes([.13, .245, .31, .55]), evidence)
    draw_roc(fig.add_axes([.565, .245, .35, .55]), evidence)
    footnotes = [
        '기반: 673행 전체 평가 Eye A(eye_A_final.joblib) · 새 13성분 매핑 · 기존 전처리 고정 · 앵커 L2(λ=1) · 판정 임계값 0.5',
        '그룹을 하나씩 제외해 얻은 예측 성적이며, 이후 16행 전체로 학습한 예측용 모델의 독립 시험 성적은 아님 · 기존 673행과의 중복 미확인',
        '근거: 외부 검증/최종 모델/oof_predictions.csv · metrics.json · protocol.json · 독립 외부검증 아님',
    ]
    for y, text in zip([.105, .066, .027], footnotes):
        fig.text(.5, y, text, ha='center', va='center', fontsize=9.3, color=MUTED)
    fig.savefig(path)
    plt.close(fig)


def overview_figure(evidence, path):
    metrics = evidence['metrics']
    transfer, frozen, control = (metrics[k] for k in ['source_centered_transfer', 'frozen_source', 'zero_centered_control'])
    fig = plt.figure(figsize=(16, 11.7), dpi=180)
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set(xlim=(0, 1), ylim=(0, 1))
    canvas.axis('off')

    def text(x, y, value, size=11, color=INK, weight='normal', ha='left', va='baseline', **kwargs):
        return canvas.text(x, y, value, fontsize=size, color=color, weight=weight,
                           ha=ha, va=va, **kwargs)

    text(.5, .969, 'Eye 전이학습 — 673행 평가 모델 기반 재구성',
         size=23, color=NAVY, weight='bold', ha='center', va='center')
    text(.5, .932, '기반 모델: 673행 전체 평가 Eye A · Gettings 1996 · 새 13성분 매핑 · 연구용 재구성 모델',
         size=11.5, color=MUTED, ha='center', va='center')
    stages = [
        ('01. 원자료 확인', 'B4 배합표 + D5 EPA 등급\n같은 연구·제형코드 연결', '25개 후보 → 16개 선택'),
        ('02. 성분 매핑 재구성', '후보 분자구조 13개 고정\n매핑되지 않은 성분 제외', '함량 가중평균 특징'),
        ('03. 673행 모델 기반 학습', 'Eye A 전처리 고정\n균형 logistic 손실 + L2', '기존 계수 앵커 · λ=1'),
        ('04. 그룹 교차검증', '15그룹을 하나씩 제외\n임계값 0.5 · OOF 16행', f"14/16 · MCC {transfer['MCC']:.3f}"),
        ('05. 예측용 모델 저장', '16행 전체로 다시 학습\n모델·예측 코드·검증 저장', 'OOF 평가와 별도 모델'),
    ]
    box_y, box_h, gap = .804, .102, .012
    width = (.93 - 4*gap)/5
    for i, (title, body, result) in enumerate(stages):
        x = .035 + i*(width+gap)
        canvas.add_patch(FancyBboxPatch((x, box_y), width, box_h,
                         boxstyle='round,pad=0,rounding_size=.011',
                         facecolor='#f1f5f8', edgecolor=COLORS[i], linewidth=2.1))
        text(x+width/2, box_y+.081, title, size=11.5, color=COLORS[i], weight='bold', ha='center', va='center')
        text(x+width/2, box_y+.050, body, size=10, ha='center', va='center', linespacing=1.55)
        text(x+width/2, box_y+.015, result, size=10.6, color=COLORS[i], weight='bold', ha='center', va='center')
        if i < 4:
            canvas.annotate('', xy=(x+width+gap-.001, box_y+box_h/2), xytext=(x+width+.001, box_y+box_h/2),
                            arrowprops={'arrowstyle': '-|>', 'color': '#8b969f', 'lw': 1.5})

    text(.035, .763, '이번 분석에 사용한 자료', size=15, weight='bold')
    text(.035, .737, '함량이 모두 숫자이고 합계가 100±0.11%인 제형만 포함', size=10.5, color=MUTED)
    categories = [
        ('원자료 후보', evidence['candidate_count'], '제형', NAVY),
        ('함량 조건으로 제외', evidence['candidate_count']-transfer['n'], '제형', '#9caab6'),
        ('학습·평가 포함', transfer['n'], '제형', GREEN),
        ('양성: EPA I/II', evidence['positive'], '제형', NAVY),
        ('음성: EPA III/IV', evidence['negative'], '제형', NAVY),
        ('동일 부분특징 그룹', evidence['groups'], '그룹', '#7850a4'),
        ('새 성분 매핑표', evidence['mapping_count'], '구조', GREEN),
    ]
    ax = fig.add_axes([.171, .487, .291, .216])
    positions = np.arange(len(categories))
    ax.barh(positions, [r[1] for r in categories], height=.57, color=[r[3] for r in categories])
    ax.set_yticks(positions, [r[0] for r in categories], fontsize=10.1)
    ax.invert_yaxis()
    ax.set_xlim(0, 30)
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', colors=MUTED, labelsize=9)
    ax.grid(axis='x', color=GRID, linewidth=.6)
    ax.set_axisbelow(True)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    for position, (_, number, unit, _) in zip(positions, categories):
        ax.text(number+.45, position, f'{number}{unit}', va='center', fontsize=10.2, weight='bold')

    text(.035, .439, '이번 재구성에서 수행한 것', size=15, weight='bold')
    steps = [
        ('원문 라벨과 제형 대조', 'D5 원문 25개 EPA 라벨 재추출·대조\n16제형: 양성 11개 · 음성 5개'),
        ('새 성분 매핑표 고정', '원본 mapping.json 미확보 · PubChem 후보 구조 조회\n누락 성분과 구조로 표현한 함량을 별도 기록'),
        ('673행 평가 모델 기반 전이학습', 'Morgan 2,048 + descriptor 217 = 2,265개 특징\n전처리 고정 · 계수·절편에 기존 값과의 거리 벌점'),
        ('15그룹 평가와 0 기준 대조군', '동일 부분특징이 학습·시험에 동시에 들어가지 않게 분리\n임계값 0.5 고정 · 대조군도 같은 분할로 평가'),
        ('전체 재학습과 실행 검사', '16행 전체로 예측용 모델 저장 · fold 모델 30개 재로딩\n특징 일치·예측 재계산·CLI·파일 무결성 검사 통과'),
    ]
    for i, (title, body) in enumerate(steps):
        y = .401-i*.072
        canvas.add_patch(Rectangle((.035, y-.044), .004, .055, color=NAVY if i < 4 else GREEN, linewidth=0))
        text(.050, y, title, size=11.6, color=NAVY if i < 4 else GREEN, weight='bold', va='center')
        text(.050, y-.019, body, size=10.2, color=MUTED, va='top', linespacing=1.5)

    right = .532
    text(right, .763, '재구성 모델의 그룹 교차검증 결과', size=15, weight='bold')
    text(right, .737, '같은 16제형·부분 입력에서 비교 · 전이학습과 대조군은 OOF', size=10.4, color=MUTED)
    draw_cm(fig.add_axes([.607, .536, .158, .163]), evidence, small=True)
    draw_roc(fig.add_axes([.833, .536, .131, .163]), evidence, small=True)

    x, y, w = right, .271, .435
    widths = [.122, .102, .111, .100]
    starts = np.r_[x, x+np.cumsum(widths)[:-1]]
    row_height, header_height = .041, .050
    rows = [
        ('MCC', frozen['MCC'], transfer['MCC'], control['MCC']),
        ('ROC-AUC', frozen['ROC_AUC'], transfer['ROC_AUC'], control['ROC_AUC']),
        ('균형정확도', frozen['balanced_accuracy'], transfer['balanced_accuracy'], control['balanced_accuracy']),
        ('적중', f"{frozen['correct']} / 16", f"{transfer['correct']} / 16", f"{control['correct']} / 16"),
    ]
    canvas.add_patch(Rectangle((x, y+len(rows)*row_height), w, header_height, color=NAVY, linewidth=0))
    headers = ['지표', '기존 Eye A\n고정', '계수 기반\n전이학습', '0 기준\n대조군']
    for start, width, header in zip(starts, widths, headers):
        text(start+width/2, y+len(rows)*row_height+header_height/2, header,
             size=10.5, color='white', weight='bold', ha='center', va='center', linespacing=1.2)
    for i, row in enumerate(rows):
        ry = y+(len(rows)-1-i)*row_height
        canvas.add_patch(Rectangle((x, ry), w, row_height, facecolor='#f2f6fa' if i % 2 else 'white',
                                  edgecolor='#c8d2dc', linewidth=.7))
        for col, (start, width, value) in enumerate(zip(starts, widths, row)):
            label = f'{value:.3f}' if isinstance(value, (float, np.floating)) else value
            text(start+width/2, ry+row_height/2, label, size=11.2,
                 color=GREEN if col == 2 else (INK if col == 0 else MUTED),
                 weight='bold' if col in [0, 2] else 'normal', ha='center', va='center')
    canvas.add_patch(Rectangle((x, y), w, len(rows)*row_height+header_height,
                              fill=False, edgecolor='#96a9bc', linewidth=1.1))

    ci = evidence['bootstrap']['MCC_95_percentile_interval']
    notes = [
        f"· 적중 {transfer['correct']}/16 ({transfer['accuracy']*100:.1f}%) · 혼동행렬 TN3 / FP2 / FN0 / TP11",
        f"· 0 기준 대조군도 MCC {control['MCC']:.3f} → 기존 계수의 추가 이점 미입증",
        f"· 그룹 재표본 5,000회 중 유효 {evidence['bootstrap']['valid_resamples']:,}회",
        f"  MCC 95% 백분위 구간 [{ci[0]:.3f}, {ci[1]:.3f}] — 불확실성이 큼",
        '· 원본 매핑표 미확보에 따른 재구성본 · 원래 그림의 동일 모델 아님',
        '· 기존 673행과의 중복 미확인 · 독립 외부검증 아님',
        '· 전체 16행 재학습 모델의 예측값과 이 OOF 성적은 구분',
    ]
    text(right, .241, '\n'.join(notes), size=10.25, color=MUTED, va='top', linespacing=1.72)
    text(.5, .027, '근거: 외부 검증/최종 모델/의 protocol.json · mapping.json · oof_predictions.csv · metrics.json · bootstrap_ci.json',
         size=9.2, color=MUTED, ha='center', va='center')
    fig.savefig(path)
    plt.close(fig)


def render(detail=None, overview=None):
    evidence = load_evidence()
    outputs = {}
    if detail:
        detail = Path(detail)
        detail.parent.mkdir(parents=True, exist_ok=True)
        detail_figure(evidence, detail)
        outputs[str(detail)] = sha(detail)
    if overview:
        overview = Path(overview)
        overview.parent.mkdir(parents=True, exist_ok=True)
        overview_figure(evidence, overview)
        outputs[str(overview)] = sha(overview)
    audit = {
        'scope': 'figures from reconstructed OOF results; not original-model recovery',
        'metrics_recalculated_from_predictions': evidence['metrics'],
        'roc_fpr': evidence['fpr'].tolist(), 'roc_tpr': evidence['tpr'].tolist(),
        'source_hashes': {name: sha(PACKAGE/name) for name in
                          ['oof_predictions.csv', 'metrics.json', 'metadata.json', 'bootstrap_ci.json', 'protocol.json', 'mapping.json', 'source_model/eye_A_final.joblib']},
        'outputs': outputs,
    }
    (HERE/'eye_transfer_figures_verification.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    render(args.output_dir/'eye 전이학습 시각화.png', args.output_dir/'eye 전이학습.png')
