# A/B Test Plan

Сравниваем две версии модели прогнозирования дефолта в условиях реального трафика.

| | v1 (control) | v2 (treatment) |
|-|--------------|----------------|
| Алгоритм | LogisticRegression | RandomForestClassifier |
| F1 | 0.46 | 0.52 |
| Precision | 0.37 | 0.55 |
| Recall | 0.62 | 0.49 |

v1 ловит больше дефолтников, v2 реже ошибается когда говорит "дефолт". Какое поведение выгоднее банку — это и есть вопрос теста.

---

## Разделение трафика

Реализовано на уровне NGINX через `split_clients`:

```nginx
split_clients "${remote_addr}${time_iso8601}" $ml_backend {
    50%   ml_v1;
    *     ml_v2;
}
```

50% запросов идёт на v1, 50% на v2. Версия модели логируется в каждом ответе через поле `model_version`.

Rollback — изменить env var `AB_SPLIT` и перезапустить контейнер.

---

## Продолжительность теста

Минимум 2 недели. Меньше не даст достаточной выборки и не покроет недельную сезонность.

Расчёт минимальной выборки:

```
p1 = 0.22
delta = 0.03
alpha = 0.05, power = 0.80

n = 2 * (z_alpha/2 + z_beta)^2 * p*(1-p) / delta^2
  = 2 * (1.96 + 0.84)^2 * 0.22 * 0.78 / 0.03^2
  ~ 3000 запросов на группу
```

---

## Метрики

**F1-score** — первичная метрика. Датасет несбалансированный (22% дефолтов), accuracy здесь бессмысленна. F1 учитывает оба типа ошибок одновременно.

**Precision** — важна банку, чтобы не отказывать добросовестным клиентам. Каждый ложный отказ — потерянный процентный доход.

**Recall** — важна, чтобы не пропускать реальных дефолтников. Каждый пропущенный — прямые финансовые потери.

### Бизнес-метрики

**Expected Loss** — ожидаемые финансовые потери по портфелю:

```python
EL = sum(P_default_i * LGD * limit_bal_i)
LGD = 0.45
```

**Approval Rate** — доля одобренных заявок при пороге `P(default) < 0.3`:

```python
approval_rate = sum(prob < 0.3) / total_requests
```

Цель: v2 одобряет не меньше заявок при EL не хуже v1.

---

## Статистический анализ

F1, Precision, Recall — z-тест для пропорций:

```python
from statsmodels.stats.proportion import proportions_ztest

count = [tp_v1, tp_v2]
nobs  = [tp_v1 + fp_v1, tp_v2 + fp_v2]
stat, p_value = proportions_ztest(count, nobs)
```

Expected Loss — t-тест Welch, так как метрика непрерывная:

```python
from scipy.stats import ttest_ind

stat, p_value = ttest_ind(el_v1, el_v2, equal_var=False)
```

Доверительные интервалы — bootstrap, 1000 итераций:

```python
import numpy as np

def bootstrap_ci(metric_fn, y_true, y_pred, n=1000):
    scores = []
    for _ in range(n):
        idx = np.random.randint(0, len(y_true), len(y_true))
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    return np.percentile(scores, [2.5, 97.5])
```

---

## Критерий успеха v2

Все три условия должны выполняться:

1. `p_value < 0.05` по F1
2. `F1(v2) >= F1(v1)`
3. `EL(v2) <= EL(v1)`

Если хотя бы одно не выполнено — оставляем v1.

---

## Связь с архитектурой

- Два отдельных контейнера изолируют версии и не перемешивают логи
- NGINX делит трафик без изменений в коде моделей
- JSON-логирование с полем `model_version` позволяет разделить выборки для анализа
- `AB_SPLIT` как env var даёт быстрый rollback без пересборки образа