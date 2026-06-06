# credit-card-ml-deployment

Сервис машинного обучения для прогнозирования дефолта по кредитным картам. Flask, Docker, A/B-тестирование двух версий модели через NGINX.

Датасет: [Default of Credit Card Clients, UCI](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) — 30 000 клиентов, Тайвань, таргет — дефолт в следующем месяце.

---

## Структура

```
credit-card-ml-deployment/
├── app/
│   ├── api.py
│   └── model_handler.py
├── models/
│   ├── train_model.py
│   ├── model_v1.pkl
│   └── model_v2.pkl
├── tests/
│   └── test_api.py
├── docker/
│   └── nginx.conf
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── ab_test_plan.md
└── ARCHITECTURE.md
```

---

## Запуск локально

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python models/train_model.py

python app/api.py
```

Сервис поднимается на `http://localhost:5000`.

---

## Docker

Docker Hub: [k0stylev/credit-default-ml](https://hub.docker.com/r/k0stylev/credit-default-ml)

```bash
docker pull k0stylev/credit-default-ml:latest
docker run -p 5000:5000 k0stylev/credit-default-ml:latest
```

Собрать локально:

```bash
docker build -t k0stylev/credit-default-ml:latest .
docker run -p 5000:5000 k0stylev/credit-default-ml:latest
```

### Docker Compose

```bash
docker compose up --build
```

Поднимает три контейнера: v1 на порту 5001, v2 на 5002, NGINX на 80.

---

## API

### GET /health

```bash
curl http://localhost:5000/health
```

```json
{"status": "healthy", "timestamp": "2026-06-05T23:24:09.083043Z"}
```

### POST /predict

Принимает JSON с 23 признаками клиента, возвращает предсказание и вероятность дефолта.

Параметр `?model=v1` или `?model=v2` выбирает версию явно. Без параметра — случайный сплит 50/50.

```bash
curl -X POST http://localhost:5000/predict?model=v1 \
  -H "Content-Type: application/json" \
  -d '{
    "limit_bal": 20000, "sex": 2, "education": 2, "marriage": 1, "age": 24,
    "pay_0": 2, "pay_2": 2, "pay_3": -1, "pay_4": -1, "pay_5": -1, "pay_6": -1,
    "bill_amt1": 3913, "bill_amt2": 3102, "bill_amt3": 689,
    "bill_amt4": 0, "bill_amt5": 0, "bill_amt6": 0,
    "pay_amt1": 0, "pay_amt2": 689, "pay_amt3": 0,
    "pay_amt4": 0, "pay_amt5": 0, "pay_amt6": 0
  }'
```

```json
{"model_version": "v1", "prediction": 1, "probability": 0.7735}
```

```bash
curl -X POST http://localhost:5000/predict?model=v2 \
  -H "Content-Type: application/json" \
  -d '{
    "limit_bal": 20000, "sex": 2, "education": 2, "marriage": 1, "age": 24,
    "pay_0": 2, "pay_2": 2, "pay_3": -1, "pay_4": -1, "pay_5": -1, "pay_6": -1,
    "bill_amt1": 3913, "bill_amt2": 3102, "bill_amt3": 689,
    "bill_amt4": 0, "bill_amt5": 0, "bill_amt6": 0,
    "pay_amt1": 0, "pay_amt2": 689, "pay_amt3": 0,
    "pay_amt4": 0, "pay_amt5": 0, "pay_amt6": 0
  }'
```

```json
{"model_version": "v2", "prediction": 1, "probability": 0.87}
```

| поле | тип | описание |
|------|-----|----------|
| `prediction` | int | 0 нет дефолта, 1 дефолт |
| `probability` | float | вероятность дефолта [0.0, 1.0] |
| `model_version` | string | версия модели |

```bash
curl http://localhost:5000/features?model=v1
```

### Тесты

```bash
pytest tests/test_api.py -v
```

---

## Модели

| версия | алгоритм | F1 | Precision | Recall |
|--------|----------|----|-----------|--------|
| v1 | LogisticRegression | 0.46 | 0.37 | 0.62 |
| v2 | RandomForestClassifier | 0.52 | 0.55 | 0.49 |

v1 ловит больше дефолтников (Recall 0.62), v2 точнее когда говорит "дефолт" (Precision 0.55). Разное поведение при схожем F1 — основа для A/B-теста.

---

## A/B-тестирование

Подробный план: [ab_test_plan.md](ab_test_plan.md)

```bash
curl -X POST http://localhost:5000/predict?model=v1 -H "Content-Type: application/json" -d '{...}'
curl -X POST http://localhost:5000/predict?model=v2 -H "Content-Type: application/json" -d '{...}'
```

---

## Архитектура

Монолит vs микросервисы, NGINX+uWSGI, ONNX, RabbitMQ, DVC, MLflow: [ARCHITECTURE.md](ARCHITECTURE.md)