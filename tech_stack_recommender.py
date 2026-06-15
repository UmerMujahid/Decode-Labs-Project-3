
# ============================================================
#   PROJECT 3 — TECH STACK RECOMMENDER
#   Decode Labs Internship
#   Algorithm: Content-Based Filtering | TF-IDF + Cosine Similarity
#   Platform : Google Colab (no extra installs needed)
# ============================================================

# ------------------------------------------------------------------
# STEP 0 ▸ Import Libraries
#   All libraries below come pre-installed in Google Colab.
#   No pip install needed!
# ------------------------------------------------------------------
import pandas as pd                          # to load and handle our CSV dataset
import math                                  # for square-root in cosine similarity
from collections import Counter              # to count term frequencies


# ==================================================================
# SECTION 1 ▸ LOAD THE DATASET
# ==================================================================
# Make sure raw_skills.csv is uploaded to your Colab session.
# In Colab: click the folder icon → Upload → choose raw_skills.csv
# ------------------------------------------------------------------

def load_dataset(filepath):
    df = pd.read_csv(filepath)

    dataset = []
    for _, row in df.iterrows():
        role   = row["Job Role"].strip()
        skills = row["Skills"].strip().split()   # skills separated by spaces
        dataset.append({"role": role, "skills": skills})

    return dataset


# ==================================================================
# SECTION 2 ▸ BUILD THE VOCABULARY (Unique Words Across All Roles)
# ==================================================================

def build_vocabulary(dataset):
    vocab = set()
    for item in dataset:
        for skill in item["skills"]:
            vocab.add(skill.lower())        # lowercase to avoid "Python" != "python"
    return sorted(list(vocab))              # sorted list → consistent ordering


# ==================================================================
# SECTION 3 ▸ TF-IDF WEIGHTING
# ==================================================================
# TF-IDF = Term Frequency x Inverse Document Frequency
#
# • TF  → how often a skill appears in THIS role's skill list
# • IDF → how rare that skill is across ALL roles
#          (rare skills carry more weight → more discriminating)
# ------------------------------------------------------------------

def compute_idf(dataset, vocab):
    total_roles = len(dataset)
    idf = {}

    for skill in vocab:
        # count how many roles contain this skill
        count = sum(1 for item in dataset if skill in [s.lower() for s in item["skills"]])
        # add +1 to avoid division by zero (smoothing)
        idf[skill] = math.log((total_roles + 1) / (count + 1)) + 1

    return idf


def compute_tfidf_vector(skills_list, vocab, idf):
    # Step 1 — Term Frequency (how often each skill appears)
    skill_counts = Counter([s.lower() for s in skills_list])
    total_skills  = len(skills_list)

    # Step 2 — Build the TF-IDF vector
    vector = []
    for word in vocab:
        tf    = skill_counts.get(word, 0) / total_skills   # TF for this word
        tfidf = tf * idf.get(word, 1)                      # multiply by IDF
        vector.append(tfidf)

    return vector


# ==================================================================
# SECTION 4 ▸ COSINE SIMILARITY
# ==================================================================
# Measures the ANGLE between two vectors.
# The closer the angle to 0 degrees, the more similar → score closer to 1.
#
# Formula:
#   cos(theta) = (A · B) / (||A|| x ||B||)
#
#   Where:
#       A · B  = dot product (sum of element-wise products)
#       ||A||  = magnitude of vector A (square root of sum of squares)
# ------------------------------------------------------------------

def cosine_similarity(vec_a, vec_b):
    # Dot product: multiply matching positions and sum them up
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))

    # Magnitudes (lengths) of each vector
    magnitude_a = math.sqrt(sum(a ** 2 for a in vec_a))
    magnitude_b = math.sqrt(sum(b ** 2 for b in vec_b))

    # Avoid division by zero (if a vector is all zeros)
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# ==================================================================
# SECTION 5 ▸ THE RECOMMENDATION PIPELINE
# ==================================================================
# Brings together all 4 pipeline steps:
#   Step 1: Ingest  → capture user skills
#   Step 2: Score   → compute cosine similarity for every job role
#   Step 3: Sort    → rank roles from highest to lowest score
#   Step 4: Filter  → keep only Top-N results
# ------------------------------------------------------------------

def recommend(user_skills, dataset, vocab, idf, top_n=3):
    # ── STEP 1: INGESTION ─────────────────────────────────────────
    # Convert user's skills into a TF-IDF vector
    user_vector = compute_tfidf_vector(user_skills, vocab, idf)

    # ── STEP 2: SCORING ───────────────────────────────────────────
    # Compare user vector to every job role's vector
    scored_roles = []

    for item in dataset:
        role_vector = compute_tfidf_vector(item["skills"], vocab, idf)
        score = cosine_similarity(user_vector, role_vector)
        scored_roles.append((item["role"], score))

    # ── STEP 3: SORTING ───────────────────────────────────────────
    # Sort by score in descending order (highest match first)
    scored_roles.sort(key=lambda x: x[1], reverse=True)

    # ── STEP 4: FILTERING ─────────────────────────────────────────
    # Return only the top-N results
    return scored_roles[:top_n]


# ==================================================================
# SECTION 6 ▸ USER INTERFACE — INPUT & OUTPUT
# ==================================================================

def get_user_skills():
    print("=" * 55)
    print("   TECH STACK RECOMMENDER — Decode Labs Project 3")
    print("=" * 55)
    print("\nEnter your skills one at a time.")
    print("   You must enter AT LEAST 3 skills.")
    print("   Type  'done'  when you are finished.\n")

    skills = []
    while True:
        skill = input(f"   Skill #{len(skills) + 1}: ").strip()

        if skill.lower() == "done":
            if len(skills) < 3:
                print(f"\n   You need at least 3 skills. You entered {len(skills)}. Keep going!\n")
            else:
                break
        elif skill == "":
            print("   Empty input. Please type a skill or 'done'.")
        else:
            skills.append(skill)
            print(f"   Added: {skill}")

    return skills


def display_results(user_skills, recommendations, dataset):
    # Build a lookup dict: role name → skills list from CSV
    role_skills_map = {item["role"]: item["skills"] for item in dataset}

    print("\n" + "=" * 55)
    print("   YOUR PROFILE")
    print("=" * 55)
    print(f"   Skills entered: {', '.join(user_skills)}")

    print("\n" + "=" * 55)
    print("   TOP RECOMMENDED CAREER PATHS")
    print("=" * 55)

    medals = ["1st", "2nd", "3rd"]

    for rank, (role, score) in enumerate(recommendations):
        medal      = medals[rank] if rank < 3 else f"#{rank + 1}"
        bar_length = int(score * 30)               # visual progress bar
        bar        = "#" * bar_length + "-" * (30 - bar_length)

        # Fetch required skills for this role from the dataset
        required_skills = role_skills_map.get(role, [])
        skills_display  = ", ".join(required_skills)

        print(f"\n   [{medal}]  {role}")
        print(f"       Match Score    : {score:.4f}")
        print(f"       Similarity     : [{bar}]")
        print(f"       Skills Needed  : {skills_display}")

    print("\n" + "=" * 55)
    print("   TIP: Add more skills to refine your results!")
    print("=" * 55 + "\n")


# ==================================================================
# SECTION 7 ▸  MAIN — RUN THE PROGRAM
# ==================================================================

def main():
    filepath = "raw_skills.csv"

    print("\nLoading dataset...")
    dataset = load_dataset(filepath)
    print(f"Loaded {len(dataset)} job roles from dataset.\n")

    # Build shared vocabulary
    vocab = build_vocabulary(dataset)
    print(f"Vocabulary built: {len(vocab)} unique skills found.\n")

    # Pre-compute IDF weights
    idf = compute_idf(dataset, vocab)

    # Get user input
    user_skills = get_user_skills()

    # Run the recommendation pipeline
    print("\nRunning recommendation engine...")
    recommendations = recommend(user_skills, dataset, vocab, idf, top_n=3)

    # Display results — pass dataset so skills can be fetched per role
    display_results(user_skills, recommendations, dataset)


# Entry point
if __name__ == "__main__":
    main()
