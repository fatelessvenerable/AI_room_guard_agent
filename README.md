# Intelligent Guard AI System – Project Overview

## 📌 Description
This repository implements an **Intelligent Guard AI System** that can:
- Recognize **trusted users** using facial and optional voice features
- Detect and **flag unrecognized individuals**
- Provide **alerts** (e.g., WhatsApp notifications)

It is designed to work in real time using pre-generated embeddings and live detection loops.

---

## 📁 Project Structure

### 🗂 Folders

- **`trusted_users/`**  
  Contains images of trusted users (owners) used to generate facial embeddings.

- **`flagged_unrecognized_guys/`**  
  Stores images of unrecognized individuals captured by the system during testing.

---

## 📄 Files

- **`a2_ee782_hardik__archi_report.pdf`**  
  Full project report with methodology, results, and analysis.

- **`facial_feature_embedding.py`**  
  Python script to generate facial embeddings from images in `trusted_users/`.  
  These embeddings are saved and later used for recognition.

- **`hardik_archi_a2.py`**  
  Main project script that integrates:
  - Face recognition
  - Detection of unrecognized individuals
  - Alerting (e.g., messaging notifications, depending on setup)

- **`trusted_embeddings.pkl`**  
  Pre-generated facial embeddings file used for faster setup/demos.

- **`video_and_github_repo_links.txt`**  
  Contains links to the project demo video(s) and this GitHub repository.

---

## 📌 Usage Instructions

1. Populate the `trusted_users/` folder with images of recognized users.
2. Run:
   ```bash
   python facial_feature_embedding.py
  to generate embeddings in trusted_embeddings.pkl
3. Run the main AI guard script:
   ```bash
   python hardik_archi_a2.py
