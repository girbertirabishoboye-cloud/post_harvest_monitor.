# Post-Harvest Loss Monitoring Dashboard

A responsive, dark-themed Streamlit dashboard for the project **Machine Learning-Based System for Predicting and Reducing Post-Harvest Losses Among Farmers**.

It reads sensor data from Firebase Realtime Database through its REST API. The application is read-only: it uses HTTP GET requests and never writes to Firebase.

## Project structure

```text
post_harvest_dashboard/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── secrets.toml.example
```

## 1. Firebase data contract

The app reads these paths:

- `sensor_readings/latest`
- `sensor_readings/history`
- `system_status`
- `sensor_status`

Recommended latest record:

```json
{
  "temperature": 26.4,
  "humidity": 68.2,
  "soil_moisture_raw": 612,
  "soil_moisture_percent": 43.5,
  "soil_condition": "NORMAL",
  "environmental_risk": "WARNING",
  "recommendation": "Check storage ventilation and recheck conditions.",
  "timestamp_ms": 1790000000000
}
```

The values above are a **schema illustration only**, not live readings. Your actual database may contain different values or omit fields.

History can be a Firebase object keyed by push IDs or a JSON array. Each history record should contain the same sensor fields and `timestamp_ms`.

### Device heartbeat

To verify the ESP8266 status, publish a timestamp to one of these fields under `sensor_status` or `system_status`:

- `last_seen_ms`
- `heartbeat_ms`
- `last_seen`
- `heartbeat`

A Unix timestamp in milliseconds is recommended. The dashboard marks the device ONLINE when the heartbeat is no more than 120 seconds old, OFFLINE when older, and UNVERIFIED if no usable heartbeat exists. Change the threshold in `device_status()` if your firmware sends data less frequently.

Firebase being reachable does **not** mean the ESP8266 is online.

## 2. Run locally

Python 3.10 or later is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and set the correct database URL and, if required, a Firebase database auth token.

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## 3. Firebase permissions and authentication

The configured database URL defaults to:

`https://postharvestmonitor-default-rtdb.firebaseio.com`

If your database rules allow public read access, no token may be required. For a protected database, provide a valid Firebase Realtime Database auth token using Streamlit Secrets. Do not commit tokens, service account JSON, or private keys.

The dashboard only needs read permission for the four paths listed above. If Firebase returns HTTP 401/403, check the token and database rules. Do not make the database publicly writable to fix a read error.

Example rules for a controlled test environment (adapt to your authentication model; do not blindly deploy public access):

```json
{
  "rules": {
    "sensor_readings": { ".read": "auth != null", ".write": false },
    "system_status": { ".read": "auth != null", ".write": false },
    "sensor_status": { ".read": "auth != null", ".write": false }
  }
}
```

The ESP8266 should authenticate separately with the permissions needed to write its sensor records. The dashboard does not need write access.

## 4. Streamlit Community Cloud deployment

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, `README.md`, `.gitignore`, and the `.streamlit` folder.
2. Ensure `.streamlit/secrets.toml` is **not** committed. Commit only `.streamlit/secrets.toml.example`.
3. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
4. Choose **Create app**, select the repository, branch, and `app.py`.
5. Open the app's **Settings → Secrets** section.
6. Paste the TOML configuration below, replacing values with your real configuration.
7. Deploy. Review the app logs if Firebase returns an error.

Example Secrets content:

```toml
[firebase]
database_url = "https://postharvestmonitor-default-rtdb.firebaseio.com"
# Include only if your database requires it. Never commit a real token.
# auth_token = "YOUR_FIREBASE_DATABASE_AUTH_TOKEN"
```

## 5. Access from phone or computer

After deployment, Streamlit provides a public app URL. Open that same URL in a desktop browser or smartphone browser. The layout uses Streamlit's responsive columns and mobile-friendly navigation. On a phone, use the sidebar menu to switch pages.

## 6. Refresh and data updates

The dashboard refreshes its Firebase data approximately every 8 seconds using Streamlit's timed fragment. New ESP8266 records become visible after the next refresh. This is periodic polling, not a Firebase push subscription.

## 7. Decision Tree model integration

The ML page intentionally does not claim a model is integrated. It accepts a `.joblib` model upload during the current browser session and runs `predict()` when all three latest sensor values are present.

Expected feature columns and order:

1. `temperature`
2. `humidity`
3. `soil_moisture_percent`

For best reliability, save a scikit-learn `Pipeline` containing all preprocessing and the fitted `DecisionTreeClassifier`. Train and validate the model in Kaggle, then verify the same feature names, units, preprocessing, and class labels before deployment. Only load model files you trust. Uploaded model files are not permanently stored by this dashboard.

The model prediction is shown separately from the device-reported `environmental_risk`.

## 8. Important interpretation notes

- Missing Firebase fields are shown as unavailable, not replaced with sample values.
- A missing history path results in empty-history messaging and no fabricated chart.
- Risk labels and recommendations are taken from Firebase. The generic farmer guidance is advisory, not a diagnosis of spoilage.
- The dashboard cannot calculate meaningful crop-specific safe thresholds without the crop type, storage method, and validated reference limits.
- If no heartbeat is supplied, device status remains **UNVERIFIED**.
- Timestamp parsing supports Unix milliseconds, Unix seconds, and parseable date strings.

## 9. Troubleshooting

| Symptom | What to check |
|---|---|
| Firebase connection failed / 404 | Verify database URL and path names. |
| 401 or 403 | Check auth token and read rules for the four paths. |
| Latest values show dashes | Confirm exact field names and that latest is a JSON object. |
| History is empty | Confirm `sensor_readings/history` contains child records. |
| Device says UNVERIFIED | Add `last_seen_ms` or `heartbeat_ms` to `sensor_status` or `system_status`. |
| Old readings | Check ESP8266 Wi-Fi, firmware upload logic, and `timestamp_ms`. |
| Model prediction fails | Check that the artifact is a fitted estimator/pipeline and uses the expected features. |

## 10. Security

Never put Firebase credentials in source code or GitHub. Keep `.streamlit/secrets.toml` local and use Streamlit Cloud Secrets in production. Rotate any credential that has accidentally been committed.
