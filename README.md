# Predictive Maintenance ML Pipeline — EMR Serverless

Binary classification on industrial motor sensor data. Predicts whether a machine will fail within 24 hours using Spark MLlib on AWS EMR Serverless.

## What it does

Reads sensor readings from S3, trains a Logistic Regression model, scores the test split, and writes predictions + AUC metric + saved model back to S3 — all running serverless, no cluster to manage.

---

## Architecture

```
S3 (raw/)
    │
    │  sensor_readings.json
    │  (temperature, vibration, rpm,
    │   pressure, age_days, hours_since_maintenance,
    │   failure_within_24h)
    │
    ▼
┌─────────────────────────────────────────────────┐
│              EMR Serverless                      │
│                                                  │
│   SparkSession                                   │
│       │                                          │
│       ▼                                          │
│   Read JSON ──► 80/20 split (seed=42)            │
│                    │            │                │
│                  train        test               │
│                    │                             │
│                    ▼                             │
│   ┌─────────────────────────────┐               │
│   │      Spark ML Pipeline      │               │
│   │                             │               │
│   │  VectorAssembler            │               │
│   │  (6 features → raw_features)│               │
│   │          │                  │               │
│   │  StandardScaler             │               │
│   │  (zero mean, unit std)      │               │
│   │          │                  │               │
│   │  LogisticRegression         │               │
│   │  (maxIter=100, reg=0.01)    │               │
│   └─────────────────────────────┘               │
│                    │                             │
│               PipelineModel                      │
│                    │                             │
│             transform(test)                      │
│                    │                             │
│       BinaryClassificationEvaluator              │
│              (AUC metric)                        │
│                                                  │
└─────────────────────────────────────────────────┘
    │
    ├── S3 (ml-output/predictions/)   ← scored test rows (Parquet)
    ├── S3 (ml-output/metrics/)       ← AUC value (JSON)
    └── S3 (ml-output/model/)         ← saved PipelineModel
```

---

## ML Pipeline Flow

```
Raw JSON record
──────────────────────────────────────────────────────────
{
  "sensor_id":               "s036",
  "machine_id":              "machine_008",
  "temperature":             98.5,      ─┐
  "vibration":               3.8,        │
  "rpm":                     920.0,      │  6 numeric features
  "pressure":                168.0,      │
  "age_days":                720,        │
  "hours_since_maintenance": 380,       ─┘
  "failure_within_24h":      1.0        ← label
}

Step 1 — VectorAssembler
  [98.5, 3.8, 920.0, 168.0, 720, 380]  →  raw_features (DenseVector)

Step 2 — StandardScaler
  Subtract mean, divide by stddev per feature  →  features (normalised)

Step 3 — LogisticRegression
  w · features + b  →  rawPrediction (log-odds)
                    →  probability   [P(0), P(1)]
                    →  prediction    0.0 or 1.0

Output record
──────────────────────────────────────────────────────────
sensor_id | machine_id | timestamp | failure_within_24h | prediction | probability
```

---

## Features

| Feature | Unit | Normal range | Failure range |
|---|---|---|---|
| temperature | °C | 55–82 | 90–118 |
| vibration | g | 0.3–1.9 | 2.8–5.8 |
| rpm | rev/min | 1200–1580 | 680–1050 |
| pressure | bar | 95–142 | 150–195 |
| age_days | days | 30–730 | 500–730 |
| hours_since_maintenance | hours | 8–120 | 200–530 |

**Label:** `failure_within_24h` — `1.0` if the machine fails within 24 hours, `0.0` otherwise.

---

## Project Layout

```
emr-ml/
├── pom.xml                                        # Maven build — Spark + MLlib + shade plugin
├── src/main/java/com/emr/ml/
│   ├── PredictiveMaintenancePipeline.java         # Train job: reads data, trains model, writes output
│   └── SensorScorer.java                         # Inference job: loads saved model, scores new data
├── infra/
│   └── emr-serverless.yaml                        # CloudFormation: EMR app + IAM role
├── sample-data/
│   └── sensor_readings.json                       # 50 labelled sensor records
├── scripts/
│   └── build_and_submit.sh                        # Build + upload + submit in one command
└── .github/workflows/
    └── deploy.yml                                 # CI/CD: build → validate CFN → deploy
```

---

## Output

| Path | Format | Contents |
|---|---|---|
| `ml-output/predictions/` | Parquet | sensor_id, machine_id, timestamp, actual label, prediction, probability |
| `ml-output/metrics/` | JSON | `{ "auc": 0.9750 }` |
| `ml-output/model/` | Spark MLlib | Saved PipelineModel — reload with `PipelineModel.load(path)` |

---

## Inference — Scoring New Data

Once the training job has run and saved a model to `ml-output/model/`, use `SensorScorer` to score new sensor readings without retraining.

```
SensorScorer takes 3 arguments:
  <model_path>   s3://bucket/ml-output/model/      ← saved PipelineModel from training run
  <input_path>   s3://bucket/raw/new/              ← new unlabelled sensor JSON
  <output_path>  s3://bucket/ml-output/scored/     ← predictions written here
```

Submit via AWS console — Script arguments:
```
["s3://amazon-emr-724921138194/ml-output/model/","s3://amazon-emr-724921138194/raw/new/","s3://amazon-emr-724921138194/ml-output/scored/"]
```

Spark submit parameters:
```
--class com.emr.ml.SensorScorer
```

Output columns: `sensor_id`, `machine_id`, `timestamp`, `prediction` (0.0 or 1.0), `probability` ([P(normal), P(failure)]).

**Train vs Score:**

| | `PredictiveMaintenancePipeline` | `SensorScorer` |
|---|---|---|
| Reads labelled data | Yes | No (label not required) |
| Trains model | Yes | No |
| Saves model to S3 | Yes | No |
| Loads model from S3 | No | Yes |
| Writes predictions | Yes (test split) | Yes (all input rows) |
| Writes AUC metric | Yes | No |

---

## End-to-End Run

### 1. Build
```bash
mvn clean package -DskipTests
```

### 2. Deploy infrastructure
```bash
aws cloudformation deploy \
  --stack-name predictive-maintenance-emr-serverless \
  --template-file infra/emr-serverless.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides Environment=prod DataBucket=your-bucket
```

### 3. Build, upload, and submit
```bash
chmod +x scripts/build_and_submit.sh
./scripts/build_and_submit.sh your-bucket <app-id> <role-arn>
```

### 4. Check results
```bash
aws s3 ls s3://your-bucket/ml-output/ --recursive
```

---

## CI/CD

Push to `main` → GitHub Actions runs:
1. **build-and-test** — `mvn test` + `mvn package`
2. **validate-cfn** — `cfn-lint` on the CloudFormation template
3. **deploy** — deploys CFN stack, uploads JAR + data, submits EMR Serverless job

Uses OIDC federation via `tiny-transformer-github-actions-role` — no long-lived AWS keys.

Required GitHub secret: `DATA_BUCKET`

---

## GitHub Actions IAM Setup

The workflow assumes role `tiny-transformer-github-actions-role` (account `724921138194`) via OIDC — no static credentials stored in GitHub.

### Trust policy
The role's trust policy allows `yonatanbeyn/emr-ml` to assume it:
```json
"StringLike": {
  "token.actions.githubusercontent.com:sub": [
    "repo:yonatanbeyn/emr-ml:ref:refs/heads/main",
    "repo:yonatanbeyn/emr-ml:environment:*"
  ]
}
```

### Permissions granted (inline policy `EMRServerlessPipelineAccess`)

| Sid | Actions | Resource |
|---|---|---|
| CloudFormation | CreateStack, UpdateStack, CreateChangeSet, ExecuteChangeSet, DescribeStacks, etc. | `arn:aws:cloudformation:us-east-1:724921138194:stack/*/*` |
| S3 | PutObject, GetObject, ListBucket | `arn:aws:s3:::amazon-emr-724921138194/*` |
| EMRServerless | CreateApplication, StartJobRun, GetJobRun, TagResource, etc. | `*` |
| IAMForCFN | CreateRole, DeleteRole, AttachRolePolicy, PassRole, etc. | `arn:aws:iam::724921138194:role/*-emr-*` |

The IAM resource pattern `*-emr-*` covers both `prod-emr-serverless-job-role` (transaction analytics project) and `prod-emr-ml-job-role` (this project).

### Add the OIDC provider (one-time, already done)
```bash
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com
```
