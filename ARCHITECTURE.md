# ARCHITECTURE.md

Обоснование ключевых архитектурных решений проекта.

---

## Монолит vs микросервисы

Выбрал **монолит**: один Flask-процесс, один Docker-образ.

Одна модель, один тип запроса, один разработчик. Разбивать на отдельные сервисы (feature preprocessing, inference, response formatting) здесь незачем: каждый межсервисный вызов добавляет latency и точку отказа, а выигрыша в независимом масштабировании при таком объёме нет.

Когда монолит перестаёт работать:

- Feature engineering обновляется независимо от модели
- Нужно масштабировать только inference-слой
- Разные команды владеют разными частями пайплайна
- Нагрузка в сотни RPS

Сейчас ничего из этого нет, монолит оправдан.

---

## NGINX + uWSGI в production

Flask-сервер (`app.run()`) однопоточный, без буферизации, без управления воркерами. Для production не годится.

Стандартная связка:

```
Клиент -> NGINX (80/443) -> uWSGI (unix socket) -> Flask
```

NGINX берёт на себя буферизацию медленных клиентов, SSL termination, статические файлы и A/B-роутинг через `split_clients`.

uWSGI берёт на себя параллельные запросы через несколько воркеров, graceful reload при деплое и перезапуск упавших процессов.

В этом проекте вместо uWSGI использую **Gunicorn**: проще в конфигурации, решает ту же задачу.

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app.api:app
```

---

## ONNX-ML

ONNX — платформонезависимый формат для ML-моделей. Sklearn-пайплайн экспортируется так:

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [("float_input", FloatTensorType([None, 23]))]
onnx_model = convert_sklearn(pipeline, initial_types=initial_type)

with open("models/model_v1.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

Инференс через `onnxruntime` работает в 2-5x быстрее sklearn и не зависит от Python-окружения. Актуально когда latency критична или нужен serving на платформах типа Triton.

---

## RabbitMQ

Сейчас у нас синхронный HTTP: клиент ждёт пока модель считает предсказание. При росте нагрузки это узкое место.

С брокером сообщений клиент отправляет задачу в очередь и получает `task_id`, worker забирает задачу и делает инференс, клиент забирает результат по `task_id`.

Где это нужно:

- **Батч-предсказания**: ночная обработка тысяч заявок
- **Логирование**: запросы публикуются в отдельную очередь, consumer пишет асинхронно
- **Отказоустойчивость**: если сервис лёг, задачи ждут в очереди

---

## Логирование

Сервис пишет каждый запрос в stdout в JSON-формате:

```json
{
  "timestamp": "2026-06-05T23:24:09Z",
  "level": "INFO",
  "message": "Prediction made",
  "model_version": "v1",
  "prediction": 0,
  "probability": 0.1234
}
```

В production логи собираются через ELK-стек:

```
Docker stdout -> Filebeat -> Logstash -> Elasticsearch -> Kibana
```

Метрики для Prometheus + Grafana: `predict_latency_seconds`, `predict_total{model_version, prediction}`, `predict_errors_total`.

---

## DVC и MLflow

**DVC** решает воспроизводимость данных и пайплайна. Git не хранит большие файлы — DVC хранит их в S3/GCS и версионирует указатели в Git. `dvc repro` воспроизводит весь пайплайн: данные, обучение, сохранение модели.

**MLflow** решает сравнение экспериментов:

```python
import mlflow

with mlflow.start_run():
    mlflow.log_params({"n_estimators": 100, "max_depth": 5})
    mlflow.log_metrics({"f1": 0.52, "precision": 0.55, "recall": 0.49})
    mlflow.sklearn.log_model(pipeline, "model")
```

Без этих инструментов через месяц экспериментов не вспомнишь с какими параметрами получилась нужная модель.

---

## Бизнес-метрики

Технические метрики говорят о качестве модели. Бизнес-метрики говорят о том, зарабатывает ли банк.

**Expected Loss** — ожидаемые финансовые потери:

```
EL = sum(P(default_i) * LGD * EAD_i)
```

- `P(default_i)` — вероятность дефолта из модели
- `LGD` = 0.45 — стандартная доля потерь при дефолте
- `EAD_i` = `limit_bal` — сумма под риском

**Approval Rate** — доля одобренных заявок при пороге `P(default) < 0.3`:

```
Approval Rate = одобренных / всего заявок
```

Если v2 одобряет больше клиентов при том же или меньшем EL — больше выданных кредитов, больше процентного дохода, без роста риска.