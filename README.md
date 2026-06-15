# Tech Stack Recommender — Decode Labs Internship Project 3

A content-based recommendation system that maps your skills to the most mathematically aligned tech career paths using **TF-IDF vectorization** and **Cosine Similarity**.

---

## 📁 Project Structure

```
Decode Labs Project 3/
├── index.html                  # Frontend — Neo-Brutalist cyber terminal UI
├── tech_stack_recommender.py   # Backend — Python recommendation engine (Google Colab)
├── raw_skills.csv              # Dataset — 20 job roles with required skills
└── README.md                   # This file
```

---

## 🎯 Goal

> Create a simple recommendation system based on user preferences.

- Take user input (skills / interests)
- Match preferences using **content-based filtering**
- Display ranked recommended career paths

---

## 🧠 Algorithm

### Why Content-Based Filtering?
We chose content-based filtering over collaborative filtering because we match a user's own skill profile directly against job role skill sets — no user history or ratings needed.

### Pipeline (4 Steps)

```
User Skills Input (min 3)
        ↓
  [STEP 1] INGEST
  Capture skills, normalize text
        ↓
  [STEP 2] VECTORIZE — TF-IDF
  Build shared vocabulary (104 dimensions)
  TF  = skill_count / total_skills
  IDF = log((N+1) / (df+1)) + 1
  Vector = TF × IDF per dimension
        ↓
  [STEP 3] SCORE — Cosine Similarity
  cos(θ) = (A · B) / (‖A‖ × ‖B‖)
  Computed for all 20 job roles
        ↓
  [STEP 4] FILTER — Top-N Output
  Sort descending → return Top 3
```

### Cosine Similarity Interpretation

| Score | Meaning |
|-------|---------|
| `1.0` | Perfect alignment — identical skill orientation |
| `0.5` | Strong match — significant overlap |
| `0.0` | No overlap — orthogonal vectors |
| `-1.0`| Opposite (rare in TF-IDF space since values ≥ 0) |

---

## 📊 Dataset (`raw_skills.csv`)

20 job roles including:

| Role | Key Skills |
|------|-----------|
| Data Scientist | Python, SQL, Machine Learning, Statistics, TensorFlow |
| DevOps Engineer | AWS, Docker, Kubernetes, CI/CD, Linux |
| Frontend Developer | HTML, CSS, JavaScript, React JS, TypeScript |
| AI Engineer | Python, Deep Learning, NLP, PyTorch, Transformers |
| Cloud Architect | AWS, Azure, Kubernetes, Infrastructure, Security |
| Cybersecurity Analyst | Security, Linux, Cryptography, Ethical Hacking |
| Blockchain Developer | Solidity, Web3, Ethereum, Smart Contracts |
| ... | ... |

---

## 🖥️ How to Run

### Option 1 — Python Script (Google Colab)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload both `raw_skills.csv` and `tech_stack_recommender.py` via the 📁 Files panel
3. In a cell, run:
   ```python
   exec(open("tech_stack_recommender.py").read())
   ```
4. Enter at least **3 skills** one by one when prompted, then type `done`
5. View your **Top 3 career path recommendations**

> ✅ No `pip install` needed — uses only `pandas`, `math`, and `collections` (all pre-installed in Colab)

### Option 2 — Frontend UI (Browser)

1. Open `index.html` directly in any modern browser
2. Type skills into the terminal input (or click quick-add buttons)
3. Click **EXECUTE_RECOMMENDER.SYS**
4. Hover over result cards to see the **Spec Breakdown** with matched/required skills and cosine formula

> ✅ Fully self-contained — the entire TF-IDF + Cosine Similarity engine runs in JavaScript, no server needed

---

## 🔬 Example Run

**Input:** `Python`, `SQL`, `Machine_Learning`

| Rank | Career Path | Cosine Score | Why |
|------|-------------|-------------|-----|
| 🥇 1st | Data Scientist | `0.4262` | All 3 skills matched |
| 🥈 2nd | ML Engineer | `0.3116` | 2/3 matched (no SQL) |
| 🥉 3rd | AI Engineer | `0.2970` | 2/3 matched, larger role vector |

**Input:** `AWS`, `Docker`, `Kubernetes`

| Rank | Career Path | Cosine Score |
|------|-------------|-------------|
| 🥇 1st | DevOps Engineer | `0.5540` |
| 🥈 2nd | Cloud Architect | `0.5401` |
| 🥉 3rd | Data Engineer | `0.1554` |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Algorithm | TF-IDF + Cosine Similarity (Content-Based Filtering) |
| Backend | Python 3 (`pandas`, `math`, `collections`) |
| Frontend | Vanilla HTML5 + CSS3 + JavaScript (ES6+) |
| Fonts | Orbitron · Share Tech Mono · Rajdhani (Google Fonts) |
| Platform | Google Colab / Any modern browser |

---

## 📐 Key Formulas

**TF (Term Frequency)**
```
TF(skill, role) = count(skill in role) / total_skills_in_role
```

**IDF (Inverse Document Frequency)**
```
IDF(skill) = log((N + 1) / (df + 1)) + 1
```
where `N` = total roles, `df` = roles containing this skill

**TF-IDF**
```
TF-IDF(skill, role) = TF × IDF
```

**Cosine Similarity**
```
cos(θ) = (A · B) / (‖A‖ × ‖B‖)
```

---

## 👨‍💻 Author

**Decode Labs Internship — Project 3**  
*Simple Recommendation System Based on User Preferences*

---

> 💡 **Tip:** Skills are case-insensitive. `python`, `Python`, and `PYTHON` all work the same way.
