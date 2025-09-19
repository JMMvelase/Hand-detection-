# train_compare.py
import os, joblib, math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

CSV_PATH = "sasl_dataset/sasl_landmarks.csv"
OUT_MODEL = "sasl_model_improved.pkl"
AUG_PER_SAMPLE = 5   # how many augmented versions per original row

# bones as (parent,child) for mediapipe indices
BONES = [
    (0,1),(1,2),(2,3),(3,4),      # thumb
    (0,5),(5,6),(6,7),(7,8),      # index
    (0,9),(9,10),(10,11),(11,12), # middle
    (0,13),(13,14),(14,15),(15,16),# ring
    (0,17),(17,18),(18,19),(19,20) # pinky
]
EPS = 1e-9

def get_xy_columns(df):
    x_cols = [c for c in df.columns if c.startswith('x')]
    y_cols = [c for c in df.columns if c.startswith('y')]
    x_cols = sorted(x_cols, key=lambda s: int(s[1:]))
    y_cols = sorted(y_cols, key=lambda s: int(s[1:]))
    return x_cols, y_cols

def row_to_points(row, x_cols, y_cols):
    pts = np.stack([row[x_cols].values.astype(float), row[y_cols].values.astype(float)], axis=1)
    return pts  # shape (21,2)

def normalize_and_canonicalize(pts, hand_label=None, do_mirror_left=True):
    center = pts[0].copy()
    pts_centered = pts - center
    if do_mirror_left and hand_label is not None and str(hand_label).lower().startswith('l'):
        pts_centered[:,0] *= -1
    max_dist = np.max(np.linalg.norm(pts_centered, axis=1))
    if max_dist < EPS:
        max_dist = 1.0
    pts_scaled = pts_centered / max_dist
    return pts_scaled

def compute_features_from_pts(pts):
    dists = np.linalg.norm(pts, axis=1)
    bone_vecs = []
    for (a,b) in BONES:
        v = pts[b] - pts[a]
        norm = np.linalg.norm(v)
        if norm < EPS:
            bone_vecs.extend([0.0, 0.0])
        else:
            u = v / norm
            bone_vecs.extend([u[0], u[1]])
    feats = np.concatenate([dists, np.array(bone_vecs, dtype=float)])
    return feats

def rotate_pts(pts, angle_deg):
    theta = np.deg2rad(angle_deg)
    R = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta),  math.cos(theta)]])
    return (pts @ R.T)

def augment_row(pts, hand_label):
    out = []
    out.append(normalize_and_canonicalize(pts, hand_label))
    for _ in range(AUG_PER_SAMPLE):
        ang = np.random.uniform(-30, 30)
        scale = np.random.uniform(0.9, 1.1)  # slightly reduced jitter
        noise = np.random.normal(scale=0.005, size=pts.shape)  # softer noise
        pts_aug = pts.copy()
        pts_aug = rotate_pts(pts_aug, ang)
        pts_aug = pts_aug * scale
        pts_aug = pts_aug + noise
        out.append(normalize_and_canonicalize(pts_aug, hand_label))
    return out

# --- Load CSV ---
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(CSV_PATH)
df = pd.read_csv(CSV_PATH)
x_cols, y_cols = get_xy_columns(df)
print(f"Detected {len(x_cols)} x-columns and {len(y_cols)} y-columns.")

X_list, y_list = [], []
for idx, row in df.iterrows():
    pts = row_to_points(row, x_cols, y_cols)
    hand_lbl = row.get('hand', None)
    augmented = augment_row(pts, hand_lbl)
    for pts_norm in augmented:
        feats = compute_features_from_pts(pts_norm)
        X_list.append(feats)
        y_list.append(row['label'])

X = np.vstack(X_list)
y = np.array(y_list)
print("Feature matrix:", X.shape)

# --- Define models ---
rf_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=42))
])

svm_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel="rbf", C=10, gamma="scale"))
])

# --- Cross-validation ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n🔎 Evaluating with 5-fold CV...")
rf_scores = cross_val_score(rf_pipe, X, y, cv=cv)
svm_scores = cross_val_score(svm_pipe, X, y, cv=cv)

print(f"RandomForest CV: {rf_scores} | Mean={rf_scores.mean():.3f}")
print(f"SVM CV        : {svm_scores} | Mean={svm_scores.mean():.3f}")

# --- Train final model (choose best) ---
best_pipe = rf_pipe if rf_scores.mean() >= svm_scores.mean() else svm_pipe
best_pipe.fit(X, y)

joblib.dump(best_pipe, OUT_MODEL)
print(f"\n✅ Saved best model ({'RF' if best_pipe==rf_pipe else 'SVM'}) to {OUT_MODEL}")
