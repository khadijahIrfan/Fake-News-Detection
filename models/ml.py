"""
Fake News Classification - Full ML Pipeline
============================================
Fake.csv  →  23,481 fake news articles  (label = 1)
True.csv  →  21,417 real news articles  (label = 0)
Total     →  44,898 articles

Fake subjects : News, politics, Government News, left-news, US_News, Middle-east
True subjects : politicsNews, worldnews

Task   : Binary classification — predict FAKE (1) vs REAL (0)
Models : Logistic Regression, Random Forest, Naive Bayes
"""

import os
import re
import string
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
FAKE_PATH  = "Fake.csv"
TRUE_PATH  = "True.csv"
MODEL_OUT  = "fake_news_best_model.pkl"


# ─────────────────────────────────────────────────────────────
# 1. LOAD & COMBINE DATA
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Data")
print("=" * 60)

fake_df          = pd.read_csv(FAKE_PATH)
true_df          = pd.read_csv(TRUE_PATH)

fake_df['label'] = 1   # FAKE
true_df['label'] = 0   # REAL

df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.dropna(subset=['title', 'text'], inplace=True)

print(f"Fake articles : {len(fake_df):,}")
print(f"Real articles : {len(true_df):,}")
print(f"Total         : {len(df):,}")
print(f"\nFake subjects : {fake_df['subject'].unique().tolist()}")
print(f"True subjects : {true_df['subject'].unique().tolist()}")
print(f"\nNull values:\n{df[['title','text','subject','label']].isnull().sum()}")


# ─────────────────────────────────────────────────────────────
# 2. LABEL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Label Distribution")
print("=" * 60)

label_names = ["Real", "Fake"]
print(df['label'].value_counts().rename({0: "Real (0)", 1: "Fake (1)"}))


# ─────────────────────────────────────────────────────────────
# 3. TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Text Preprocessing")
print("=" * 60)

def clean_text(text: str) -> str:
    """Lowercase, strip URLs, mentions, digits, punctuation."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)       # URLs
    text = re.sub(r"@\w+|#\w+", " ", text)             # mentions/hashtags
    text = re.sub(r"\d+", " ", text)                    # digits
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['title_clean'] = df['title'].apply(clean_text)
df['text_clean']  = df['text'].apply(clean_text)
df['combined']    = df['title_clean'] + " " + df['text_clean']

print("Sample cleaned fake title :")
print(" ", df[df['label']==1]['title_clean'].iloc[0][:100])
print("Sample cleaned real title :")
print(" ", df[df['label']==0]['title_clean'].iloc[0][:100])


# ─────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Feature Engineering")
print("=" * 60)

df['title_len']         = df['title'].str.len()
df['text_len']          = df['text'].str.len()
df['word_count']        = df['text_clean'].apply(lambda x: len(x.split()))
df['exclaim_count']     = df['text'].str.count(r'\!')
df['question_count']    = df['text'].str.count(r'\?')
df['caps_ratio']        = df['text'].apply(
    lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
)
df['avg_word_len']      = df['text_clean'].apply(
    lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
)
df['unique_word_ratio'] = df['text_clean'].apply(
    lambda x: len(set(x.split())) / max(len(x.split()), 1)
)
df['title_word_count']  = df['title_clean'].apply(lambda x: len(x.split()))
df['has_reuters']       = df['text'].str.lower().str.contains('reuters').astype(int)
df['has_quotes']        = df['text'].str.count(r'"').apply(lambda x: min(x, 10))

NUM_FEATURES = [
    'title_len', 'text_len', 'word_count', 'title_word_count',
    'exclaim_count', 'question_count', 'caps_ratio',
    'avg_word_len', 'unique_word_ratio', 'has_reuters', 'has_quotes'
]

print(f"Numeric features ({len(NUM_FEATURES)}):")
for f in NUM_FEATURES:
    fake_mean = df[df['label']==1][f].mean()
    real_mean = df[df['label']==0][f].mean()
    print(f"  {f:<22} | Fake avg: {fake_mean:6.2f}  | Real avg: {real_mean:6.2f}")


# ─────────────────────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Train/Test Split  (80% / 20%)")
print("=" * 60)

X_text = df['combined']
X_num  = df[NUM_FEATURES].values
y      = df['label']

(X_text_train, X_text_test,
 X_num_train,  X_num_test,
 y_train,      y_test) = train_test_split(
    X_text, X_num, y,
    test_size=0.2, random_state=42, stratify=y
)

print(f"Train : {len(y_train):,}  (Fake={y_train.sum():,} | Real={(y_train==0).sum():,})")
print(f"Test  : {len(y_test):,}   (Fake={y_test.sum():,}  | Real={(y_test==0).sum():,})")


# ─────────────────────────────────────────────────────────────
# 6. TF-IDF VECTORISATION
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: TF-IDF Vectorisation")
print("=" * 60)

tfidf = TfidfVectorizer(
    max_features=60_000,
    ngram_range=(1, 3),       # unigrams + bigrams + trigrams
    min_df=2,
    sublinear_tf=True,
    stop_words='english'
)

X_train_tfidf = tfidf.fit_transform(X_text_train)
X_test_tfidf  = tfidf.transform(X_text_test)

X_train_full  = hstack([X_train_tfidf, csr_matrix(X_num_train)])
X_test_full   = hstack([X_test_tfidf,  csr_matrix(X_num_test)])

print(f"TF-IDF vocabulary size : {len(tfidf.vocabulary_):,}")
print(f"Full feature matrix    : {X_train_full.shape}")


# ─────────────────────────────────────────────────────────────
# 7. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Model Training & Evaluation")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(
        C=5, max_iter=1000, solver='saga', n_jobs=-1, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=20, n_jobs=-1, random_state=42),
    "Naive Bayes": MultinomialNB(alpha=0.1),
}

results   = {}
best_name = None
best_acc  = 0.0

for name, model in models.items():
    print(f"\n{'─'*40}")
    print(f"  {name}")
    print(f"{'─'*40}")
    model.fit(X_train_full, y_train)
    y_pred = model.predict(X_test_full)
    proba  = model.predict_proba(X_test_full)[:, 1]
    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, proba)

    results[name] = {
        "model": model, "y_pred": y_pred,
        "proba": proba, "acc": acc, "auc": auc
    }

    print(f"  Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  ROC-AUC  : {auc:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=label_names))

    if acc > best_acc:
        best_acc  = acc
        best_name = name

print(f"\n{'='*60}")
print(f"✅ Best Model : {best_name}  —  Accuracy = {best_acc*100:.2f}%")
print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────
# 8. CROSS VALIDATION
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: 5-Fold Cross Validation  (best model)")
print("=" * 60)

cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    results[best_name]["model"],
    X_train_full, y_train,
    cv=cv, scoring='accuracy', n_jobs=-1
)
print(f"{best_name}")
print(f"  Fold scores : {[f'{s:.4f}' for s in scores]}")
print(f"  Mean        : {scores.mean():.4f}  ±  {scores.std():.4f}")


# ─────────────────────────────────────────────────────────────
# 9. VISUALISATIONS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Visualisations")
print("=" * 60)

# ── 9a. Main dashboard (2x2) ──────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Fake News Classifier — Results Dashboard", fontsize=15, fontweight='bold')

# Label distribution
ax = axes[0, 0]
bars = ax.bar(["Real News\n(True.csv)", "Fake News\n(Fake.csv)"],
              [len(true_df), len(fake_df)],
              color=['steelblue', 'salmon'], edgecolor='black', width=0.5)
for bar, val in zip(bars, [len(true_df), len(fake_df)]):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 200, f"{val:,}",
            ha='center', fontweight='bold')
ax.set_title("Dataset Composition")
ax.set_ylabel("Number of Articles")
ax.set_ylim(0, max(len(true_df), len(fake_df)) * 1.12)

# Model accuracy comparison
ax = axes[0, 1]
names_list = list(results.keys())
accs_list  = [results[n]['acc'] for n in names_list]
colors     = ['#2196F3', '#FF9800', '#4CAF50']
bars = ax.bar(names_list, accs_list, color=colors, edgecolor='black')
for bar, acc in zip(bars, accs_list):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() - 0.012,
            f"{acc*100:.1f}%", ha='center', va='top',
            color='white', fontweight='bold', fontsize=12)
ax.set_ylim(max(0, min(accs_list) - 0.08), 1.01)
ax.set_title("Model Accuracy Comparison")
ax.set_ylabel("Accuracy")
ax.set_xticklabels(names_list, rotation=10)

# Confusion matrix (best model)
ax = axes[1, 0]
cm = confusion_matrix(y_test, results[best_name]['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=label_names, yticklabels=label_names,
            annot_kws={"size": 13})
ax.set_title(f"Confusion Matrix  —  {best_name}")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

# ROC curves (all models)
ax = axes[1, 1]
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r['proba'])
    ax.plot(fpr, tpr, lw=2, label=f"{name}  (AUC = {r['auc']:.4f})")
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.fill_between(*roc_curve(y_test, results[best_name]['proba'])[:2],
                alpha=0.08, color='blue')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models")
ax.legend(fontsize=9, loc='lower right')

plt.tight_layout()
plt.savefig("fake_news_results.png", dpi=150, bbox_inches='tight')
print("Saved : fake_news_results.png")
plt.show()

# ── 9b. Top TF-IDF features ───────────────────────────────
if "Logistic Regression" in results:
    feature_names = tfidf.get_feature_names_out().tolist() + NUM_FEATURES
    coefs  = results["Logistic Regression"]["model"].coef_[0]
    top_n  = 25
    top_i  = np.argsort(coefs)[-top_n:][::-1]
    bot_i  = np.argsort(coefs)[:top_n]

    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 8))
    fig2.suptitle("Most Influential Words  —  Logistic Regression", fontsize=13)

    axes2[0].barh([feature_names[i] for i in top_i[::-1]],
                  [coefs[i] for i in top_i[::-1]], color='salmon', edgecolor='black')
    axes2[0].set_title("🔴 Top FAKE-leaning words")
    axes2[0].set_xlabel("Coefficient (higher = more fake)")

    axes2[1].barh([feature_names[i] for i in bot_i],
                  [coefs[i] for i in bot_i], color='steelblue', edgecolor='black')
    axes2[1].set_title("🔵 Top REAL-leaning words")
    axes2[1].set_xlabel("Coefficient (lower = more real)")

    plt.tight_layout()
    plt.savefig("fake_news_top_features.png", dpi=150, bbox_inches='tight')
    print("Saved : fake_news_top_features.png")
    plt.show()

# ── 9c. Subject breakdown ─────────────────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
fig3.suptitle("Subject Breakdown by Dataset", fontsize=13)

fake_subj = fake_df['subject'].value_counts()
axes3[0].barh(fake_subj.index, fake_subj.values, color='salmon', edgecolor='black')
axes3[0].set_title("Fake.csv  —  Subjects")
axes3[0].set_xlabel("Count")

true_subj = true_df['subject'].value_counts()
axes3[1].barh(true_subj.index, true_subj.values, color='steelblue', edgecolor='black')
axes3[1].set_title("True.csv  —  Subjects")
axes3[1].set_xlabel("Count")

plt.tight_layout()
plt.savefig("fake_news_subjects.png", dpi=150, bbox_inches='tight')
print("Saved : fake_news_subjects.png")
plt.show()


# ─────────────────────────────────────────────────────────────
# 10. SAVE BEST MODEL
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10: Saving Best Model")
print("=" * 60)

bundle = {
    "tfidf":        tfidf,
    "model":        results[best_name]["model"],
    "model_name":   best_name,
    "label_names":  label_names,
    "num_features": NUM_FEATURES,
}
joblib.dump(bundle, MODEL_OUT)
print(f"Saved → {MODEL_OUT}")
print(f"Model  : {best_name}")


# ─────────────────────────────────────────────────────────────
# 11. PREDICT FUNCTION  (use this after training)
# ─────────────────────────────────────────────────────────────
def predict_article(title: str, text: str,
                    model_path: str = MODEL_OUT) -> dict:
    """
    Predict whether an article is FAKE or REAL.

    Parameters:
        title      : Article headline
        text       : Full article body
        model_path : Path to saved .pkl file

    Returns dict:
        prediction  → "FAKE" or "REAL"
        fake_prob   → probability it is fake
        real_prob   → probability it is real
        confidence  → confidence of the top prediction

    Example:
        result = predict_article(
            title = "Shocking: Politician caught lying!",
            text  = "Sources say the senator misled voters..."
        )
        print(result)
    """
    bundle  = joblib.load(model_path)
    _tfidf  = bundle["tfidf"]
    _model  = bundle["model"]
    _labels = bundle["label_names"]

    combined  = clean_text(title) + " " + clean_text(text)
    tfidf_vec = _tfidf.transform([combined])

    t, tx = str(title), str(text)
    tc    = clean_text(tx)
    num_feats = np.array([[
        len(t),
        len(tx),
        len(tc.split()),
        len(t.split()),
        tx.count('!'),
        tx.count('?'),
        sum(1 for c in tx if c.isupper()) / max(len(tx), 1),
        np.mean([len(w) for w in tc.split()]) if tc.split() else 0,
        len(set(tc.split())) / max(len(tc.split()), 1),
        int('reuters' in tx.lower()),
        min(tx.count('"'), 10),
    ]])

    X     = hstack([tfidf_vec, csr_matrix(num_feats)])
    pred  = _model.predict(X)[0]
    proba = _model.predict_proba(X)[0]

    return {
        "prediction" : _labels[pred],
        "fake_prob"  : f"{proba[1]*100:.1f}%",
        "real_prob"  : f"{proba[0]*100:.1f}%",
        "confidence" : f"{max(proba)*100:.1f}%",
    }


# ─────────────────────────────────────────────────────────────
# 12. DEMO PREDICTIONS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 11: Demo Predictions")
print("=" * 60)

demo_articles = [
    {
        "title": "Donald Trump Sends Out Embarrassing New Year's Eve Message",
        "text" : "Trump couldn't wish all Americans a Happy New Year and leave it at that. "
                 "Instead, he had to give a shout out to his enemies, haters and the "
                 "very dishonest fake news media!!!"
    },
    {
        "title": "U.S. Senate reaches deal to reopen government, fund it through Feb. 8",
        "text" : "WASHINGTON (Reuters) - The U.S. Senate voted on Monday to advance "
                 "legislation to reopen the federal government after a three-day shutdown, "
                 "as Democrats agreed to end their blockade of a funding bill."
    },
]

for i, article in enumerate(demo_articles, 1):
    result = predict_article(article["title"], article["text"])
    print(f"\n📰 Article {i}: {article['title'][:60]}...")
    print(f"   Prediction : {result['prediction']}")
    print(f"   Fake prob  : {result['fake_prob']}")
    print(f"   Real prob  : {result['real_prob']}")
    print(f"   Confidence : {result['confidence']}")


# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")
print(f"  Fake articles   : {len(fake_df):,}")
print(f"  Real articles   : {len(true_df):,}")
print(f"  Total           : {len(df):,}")
print(f"  Best Model      : {best_name}")
print(f"  Test Accuracy   : {best_acc*100:.2f}%")
print(f"  CV Accuracy     : {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")
print(f"  ROC-AUC         : {results[best_name]['auc']:.4f}")
print(f"  Model saved to  : {MODEL_OUT}")
print(f"{'='*60}")
print("\n✅ Pipeline complete!")