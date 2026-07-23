# book-genre-classifier-mlops

An end-to-end machine learning pipeline that classifies book covers into genre categories, built as part of an ML Engineering summative assignment. This project demonstrates the full ML lifecycle: data acquisition, model training, evaluation, deployment, monitoring, and retraining — all wrapped in a cloud-hosted API with a live dashboard.

This project is built with an eye toward a broader mission: helping power genre/age-appropriate book recommendations for a reading-engagement platform for young readers in Africa.

---

##  Video Demo

**[Watch the demo on YouTube →](YOUR_YOUTUBE_LINK_HERE)**

##  Live Deployment

- **API (Swagger docs):** https://book-genre-classifier-api.onrender.com/docs
- **Health check:** https://book-genre-classifier-api.onrender.com/health

> **Note:** This is hosted on Render's free tier, which spins down after 15 minutes of inactivity. The first request after idle time may take 30–50 seconds to respond while the service wakes up — this is expected behavior, not a bug.

---

##  Project Description

The classifier predicts a **genre category** from a book cover image, using **transfer learning with MobileNetV2**. It's designed to support automatic tagging and genre-based recommendations for a reading platform aimed at young readers.

### Genre categories
- Children's Books
- Comics & Graphic Novels
- Mystery, Thriller & Suspense
- Science Fiction & Fantasy
- Teen & Young Adult

### Dataset
A curated subset of the [BookCover30 dataset](https://github.com/uchidalab/book-dataset), narrowed to the 5 categories above:
- **650 training images** (130 per category)
- **100 test images** (20 per category)

Images were downloaded programmatically from the dataset's listing CSVs (see the notebook's data acquisition section).

### Model
- **Architecture:** MobileNetV2 (ImageNet-pretrained) + custom classification head (GlobalAveragePooling → Dense(128) → Dropout → Dense(5, softmax))
- **Training approach:** Initial training with a frozen base, followed by fine-tuning of the last 30 layers at a lower learning rate
- **Final test accuracy:** ~44–45% (against a 20% random baseline for 5 classes)
- **Best-performing class:** Comics & Graphic Novels (65% F1-score) — visually the most distinct genre in this dataset

Full training, evaluation, and metric breakdowns (precision, recall, F1, confusion matrix) are in the notebook.

---

##  Architecture

```
Project_name/
│
├── README.md
├── notebook/
│   └── book_genre_classifier.ipynb      # Data acquisition, preprocessing, training, evaluation
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                  # Image loading & normalization
│   ├── model.py                          # Model architecture, training, retraining logic
│   └── prediction.py                     # Inference logic
├── api/
│   ├── main.py                           # FastAPI app (predict, retrain, health endpoints)
│   └── requirements.txt
├── ui/
│   └── index.html                        # Monitoring dashboard (status, charts, predict, retrain)
├── data/
│   ├── train/
│   └── test/
├── models/
│   ├── book_genre_classifier.keras
│   └── class_names.json
├── locustfile.py                         # Load testing script
├── locust_sample_images/                 # Sample images used in load tests
├── locust_results/                       # Load test results (1/2/3 container comparisons)
├── docker-compose.yml                    # Multi-container orchestration + nginx load balancer
├── nginx.conf                            # Load balancer config
├── Dockerfile
└── .dockerignore
```

---

##  Setup Instructions

### Prerequisites
- Python 3.11 (TensorFlow does not yet support 3.13/3.14)
- Docker Desktop
- Git

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Run the API locally (without Docker)
```bash
py -3.11 -m venv venv
venv\Scripts\Activate.ps1      # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r api/requirements.txt
uvicorn api.main:app --reload
```
Visit `http://127.0.0.1:8000/docs` for the interactive API docs.

### 3. Run the API with Docker (recommended)
```bash
docker build -t book-genre-classifier .
docker run -p 8000:8000 book-genre-classifier
```

### 4. Open the dashboard
Open `ui/index.html` directly in your browser. Update the **API Endpoint** field at the top to point to either:
- Your local instance: `http://127.0.0.1:8000`
- The live deployment: `https://book-genre-classifier-api.onrender.com`

### 5. Run the training notebook
Open `notebook/book_genre_classifier.ipynb` in Google Colab or Jupyter. It walks through:
1. Data acquisition (downloading book covers from BookCover30 listing CSVs)
2. Preprocessing (cleaning corrupted images, resizing, normalization, augmentation)
3. Model training (frozen base → fine-tuning)
4. Evaluation (classification report, confusion matrix)

---

##  API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Model status, uptime, loaded classes |
| POST | `/predict` | Upload a single book cover image, get predicted genre |
| POST | `/upload-retrain-data` | Upload multiple images for a given category (for retraining) |
| POST | `/retrain` | Trigger retraining using all uploaded images |

---

##  Multi-Container Deployment & Load Testing

The API can be scaled horizontally using Docker Compose + nginx as a load balancer.

### Running multiple containers
```bash
# 1 container
docker-compose up --build --scale api=1

# 2 containers
docker-compose up --build --scale api=2

# 3 containers
docker-compose up --build --scale api=3
```
Once running, the API is reachable through nginx at `http://127.0.0.1:8080` (not 8000).

### Load testing with Locust
```bash
pip install locust
python -m locust -f locustfile.py --host=http://127.0.0.1:8080
```
Then open `http://localhost:8089`, configure the test (20 users, ramp-up rate 5), and start.

### Results: Flood Request Simulation

Tested with **20 concurrent simulated users** hitting `/predict` and `/health`, comparing 1, 2, and 3 API containers behind the nginx load balancer:

| Containers | Median Response (ms) | 95th Percentile (ms) | Failure Rate | Requests/sec |
|---|---|---|---|---|
| 1 | 30,000 | 60,000 | 53% | 0.2 |
| 2 | 14,000 | 54,000 | 32% | 0.8 |
| 3 | 14,000 | 36,000 | **2%** | 1.2 |

**Key findings:**
- A single container cannot reliably handle 20 concurrent users running TensorFlow inference — over half of requests failed with timeouts (504) or gateway errors (502).
- Scaling from 1→2 containers roughly **halved median response time** and meaningfully reduced failures.
- Scaling from 2→3 containers had a smaller effect on median latency, but had a large impact on **tail latency (95th percentile) and reliability** — failure rate dropped from 32% to just 2%, and throughput continued climbing.
- **Takeaway:** horizontal scaling primarily improves reliability and tail latency under load, rather than uniformly speeding up every request — an important nuance for capacity planning in a real deployment.

Full Locust reports (HTML + CSV) for each configuration are available in `locust_results/`.

---

##  Known Limitation: Retraining on Render Free Tier

The `/retrain` endpoint is fully implemented and works correctly — verified locally via Docker (see video demo).

However, retraining (loading MobileNetV2 + running training epochs) requires more memory than Render's free tier provides (512MB RAM limit). Attempting to trigger retraining on the **live Render deployment** results in an out-of-memory crash — this is a hosting-tier constraint, not an application bug.

In a production setting, this would be addressed by:
- Running retraining as a separate, higher-memory background job rather than inline in the same process as the prediction API
- Upgrading to a paid Render tier with more RAM
- Using a managed cloud training service triggered by the API

**Prediction, health-check, and data-upload endpoints all work correctly on the live Render deployment.** Only the training step itself is memory-constrained on the free tier — this is demonstrated working locally instead.

---

##  Tech Stack

- **Model:** TensorFlow / Keras, MobileNetV2 transfer learning
- **API:** FastAPI, Uvicorn
- **Containerization:** Docker, Docker Compose
- **Load Balancing:** nginx
- **Load Testing:** Locust
- **Deployment:** Render
- **Dashboard:** HTML/CSS/JavaScript, Chart.js

---

##  Model Evaluation Summary

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Children's Books | 0.38 | 0.55 | 0.45 |
| Comics & Graphic Novels | 0.57 | 0.65 | 0.61 |
| Mystery, Thriller & Suspense | 0.47 | 0.45 | 0.46 |
| Science Fiction & Fantasy | 0.50 | 0.30 | 0.38 |
| Teen & Young Adult | 0.29 | 0.25 | 0.27 |
| **Overall Accuracy** | | | **44%** |

Full confusion matrix and training curves are in the notebook.


## License

This project was built for educational purposes as part of an ALU ML Engineering summative assignment.
