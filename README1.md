# LIFTNWITTIM Holistic Wellness App

A mobile biomechanical assessment tool that uses a smartphone camera to analyze fascial health and deliver a real-time Fascial Integrity Score (0–100).

---

## 1. Project Overview

The app analyzes five biomechanical modules using computer vision:

- **M1 Shearing Force** — measures tissue layer gliding
- **M2 Foot-to-Glute Connection** — checks kinetic chain integrity
- **M3 Movement Bandwidth** — tracks upper/lower body synchronization
- **M4 Hydraulic Indicator** — warmup readiness safety gate
- **M5 Stability Map** — force distribution under load

Output: a Fascial Integrity Score with tier classification (ELITE / FUNCTIONAL / COMPENSATING / NO_BUENO) and plain-language alert flags.

---

## 2. Prerequisites and Dependencies

**Backend:**
- Python 3.11
- Flask
- Flask-Limiter
- OpenCV (opencv-python-headless)
- NumPy
- SciPy
- Gunicorn

**Mobile Frontend:**
- Flutter 3.41+
- Dart
- camera package
- http package

**Infrastructure:**
- Railway (cloud deployment)
- Docker

---

## 3. Installation Steps

**Backend:**
```bash
git clone https://github.com/liftnwittim/Fascia-around-n-find-out.git
cd Fascia-around-n-find-out
pip install -r requirements.txt
```

**Flutter app:**
```bash
cd fascia_app
flutter pub get
```

---

## 4. Environment Variable Configuration

Railway injects the following environment variable automatically:

| Variable | Description |
|---|---|
| `PORT` | Port the server binds to (default: 8080) |

No additional environment variables are required for basic operation.

---

## 5. How to Run the Flask Application Locally

```bash
python FasciaApp.py
```

The server will start at `http://0.0.0.0:8080`

To run with Gunicorn (production-style):
```bash
gunicorn --bind 0.0.0.0:8080 --workers 3 --timeout 120 FasciaApp:app
```

---

## 6. Railway Deployment

The app deploys automatically via the included `Dockerfile` when changes are pushed to the `main` branch on GitHub.

- **Platform:** Railway
- **Container:** Docker (python:3.11-slim base)
- **Server:** Gunicorn with 3 workers
- **Domain:** `https://fascia-around-n-find-out-production.up.railway.app`

To redeploy manually, go to Railway → your service → Deployments → Redeploy.

---

## 7. API Endpoints

### GET /health
Confirms the server is running.

**Response:**
```json
{"status": "ok"}
```

### POST /analyze
Accepts a camera frame and returns a Fascial Integrity Score.

**Request:**
```bash
curl -X POST https://fascia-around-n-find-out-production.up.railway.app/analyze \
  -F "frame=@your_image.jpg" \
  -F "arch_engaged=neutral"
```

**Parameters:**
| Field | Type | Values |
|---|---|---|
| `frame` | file | JPEG image (max 5MB) |
| `arch_engaged` | string | `true` / `neutral` / `false` |

**Response:**
```json
{
  "score": 63.0,
  "tier": "FUNCTIONAL",
  "frame_received": true,
  "flags": [],
  "resolution": "720x1280",
  "debug": {
    "m1_shear": 75.5,
    "m2_foot_glute": 60.0,
    "m3_tensegrity": 65.0,
    "m4_hydro": 50.0,
    "m5_stability": 65.0,
    "warmup_adequate": true,
    "spike_count": 0,
    "lag_ms": 0.0
  }
}
```

**Rate limit:** 30 requests per minute per IP. Max file size: 5MB.

---

## 8. Testing Instructions

**Test backend is live:**
```bash
curl https://fascia-around-n-find-out-production.up.railway.app/health
```

**Run Flutter app on iPhone:**
```bash
cd fascia_app
flutter run -d YOUR_DEVICE_ID
```

Find your device ID with:
```bash
flutter devices
```

**Standard assessment position:**
- Tripod at 62 inches
- Subject standing 7 feet from camera
- Full body visible in frame
- Tap Analyze 3–5 times for stable readings

---

## 9. Known Limitations and Future Improvements

**Current limitations:**
- App requires USB connection to Mac without a paid Apple Developer account
- M2 Foot-to-Glute uses user-reported arch engagement — not fully automated
- M4 Hydraulic relies on RGB skin color shift which is affected by lighting conditions
- All five modules share a single camera frame — optimal framing varies by module
- Server state (frame history) resets if Railway restarts the container

**Planned improvements:**
- LiDAR integration for automated arch height detection
- Session history and longitudinal score tracking
- On-screen movement guides per module
- Apple Watch integration for HRV and heart rate data in M4
- Multi-user support with per-session state isolation
- App Store public release
