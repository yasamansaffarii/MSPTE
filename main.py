# -*- coding: utf-8 -*-
"""
==============================================================================
 Fuzzy Cognitive Map (FCM) and Scenario Analysis for Factors Affecting the
 Effectiveness of Smart Educational Robots in STEM Learning
 Method: Combination of machine learning (TF-IDF + network analysis + inferential
 statistics) with data-driven weight extraction from a real meta-synthesis study
 (51 sources, 85 concepts, 11 sub-themes, 4 main themes)
==============================================================================

 When running the UPLOAD cell, upload the following two files:
        - themes.docx   (comprehensive meta-synthesis table)
        - concepts.docx         (full description of the 85 concepts - optional, documentation only)
 Run the remaining cells in order. All tables (CSV) and figures (PNG) will be
     saved in the current folder and available for download.
"""

# %% [1] Install libraries -----------------------------------------
!pip install python-docx scikit-learn networkx scipy pandas matplotlib -q

import json, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import docx

plt.rcParams["axes.unicode_minus"] = False

# %% [2] Load input file --------------------------------------------------
# In Colab: use the file upload widget, then set the path in THEMES_DOCX
#
#   from google.colab import files
#   uploaded = files.upload()
#
THEMES_DOCX = "themes.docx"   # comprehensive meta-synthesis table (primary and sole source of the numbers)

pdig = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
def to_num(s):
    return int(str(s).translate(pdig))

d = docx.Document(THEMES_DOCX)
assert len(d.tables) >= 2, "Input file structure does not match the reference version."
t = d.tables[1]  # second table = "Comprehensive meta-synthesis table: mapping concepts to themes"

rows_data = []
for ri, row in enumerate(t.rows):
    if ri == 0:
        continue
    cells = [c.text.strip() for c in row.cells]
    rows_data.append(cells)

last_main = None
records = []
for cells in rows_data:
    main_theme = cells[0] if cells[0] else last_main
    if cells[0]:
        last_main = cells[0]
    records.append({"main_theme": main_theme, "sub_theme": cells[1],
                     "concepts_text": cells[2], "sources": cells[3], "total_freq": cells[4]})

print(f"Number of sub-themes extracted from the file: {len(records)}")

# %% [3] Extract individual concepts with frequency and polarity (facilitator/barrier) --------------
def parse_concepts(records):
    all_concepts = []
    for r in records:
        text = r["concepts_text"]
        parts = re.split(r"\s*/\s*(?=[^()]*?\(\d+\):)", text)
        for p in parts:
            p = p.strip().strip(".").strip()
            m = re.match(r"^(.*?)\s*\((\d+)\):\s*(.*)$", p, re.DOTALL)
            if not m:
                continue
            name, freq, rest = m.groups()
            freq = int(freq)
            # NOTE: the checks below match the actual Persian polarity labels
            # ("تسهیل‌گر" = facilitator, "مانع" = barrier) as they appear verbatim
            # in the source meta-synthesis document, so they must remain in Persian.
            if ("تسهیل‌گر" in rest) and ("مانع" in rest):
                polarity = "both"
            elif "تسهیل‌گر" in rest:
                polarity = "facilitator"
            elif "مانع" in rest:
                polarity = "barrier"
            else:
                polarity = "unknown"
            all_concepts.append({"main_theme": r["main_theme"], "sub_theme": r["sub_theme"],
                                  "concept": name.strip(), "freq": freq, "polarity": polarity,
                                  "desc": rest.strip()})
    return all_concepts

concepts = parse_concepts(records)
#print(f"Number of individual concepts extracted: {len(concepts)}")

# %% [4] Real statistics per sub-theme (facilitator/barrier/net frequency) ----------------
sub_stats = {}
for r in records:
    st = r["sub_theme"]
    sub_stats[st] = {"main_theme": r["main_theme"], "total_freq_reported": to_num(r["total_freq"]),
                      "fac": 0.0, "bar": 0.0}
for c in concepts:
    st = c["sub_theme"]; f = c["freq"]
    if c["polarity"] == "facilitator":
        sub_stats[st]["fac"] += f
    elif c["polarity"] == "barrier":
        sub_stats[st]["bar"] += f
    elif c["polarity"] == "both":
        sub_stats[st]["fac"] += f * 0.5
        sub_stats[st]["bar"] += f * 0.5

for st, dct in sub_stats.items():
    dct["total_parsed"] = dct["fac"] + dct["bar"]
    dct["net_polarity"] = (dct["fac"] - dct["bar"]) / dct["total_parsed"] if dct["total_parsed"] > 0 else 0.0

df_sub = pd.DataFrame([
    {"sub_theme": st, "main_theme": dd["main_theme"], "fac_freq": dd["fac"], "bar_freq": dd["bar"],
     "total_freq": dd["total_freq_reported"], "net_polarity": round(dd["net_polarity"], 4)}
    for st, dd in sub_stats.items()])
print(df_sub.to_string(index=False))
df_sub.to_csv("table1_subtheme_stats.csv", index=False, encoding="utf-8-sig")

# %% [5] Define FCM nodes (4 main themes + 11 sub-themes + 1 outcome node) -------
main_themes = list(dict.fromkeys(r["main_theme"] for r in records))
sub_themes = [r["sub_theme"] for r in records]

node_labels, label_map_main, label_map_sub = {}, {}, {}
nid = 0
for m in main_themes:
    node_labels[f"M{nid}"] = m; label_map_main[m] = f"M{nid}"; nid += 1
for s in sub_themes:
    node_labels[f"S{nid}"] = s; label_map_sub[s] = f"S{nid}"; nid += 1
OUTCOME = "OUT"
node_labels[OUTCOME] = "STEM Learning Effectiveness with Smart Educational Robots (final outcome)"
all_nodes = list(node_labels.keys())
n = len(all_nodes)
idx = {k: i for i, k in enumerate(all_nodes)}
label2id = {v: k for k, v in node_labels.items()}
print(f"\nTotal number of FCM nodes: {n}")

# %% [6] Build the FCM weight matrix (statistical + TF-IDF text-based ML) --------
sub_row = {row.sub_theme: row for row in df_sub.itertuples()}
W = np.zeros((n, n))

# (a) Hierarchical edge: sub-theme -> main theme
max_freq = df_sub["total_freq"].max()
for st in sub_themes:
    row = sub_row[st]
    w = (row.total_freq / max_freq) * row.net_polarity
    W[idx[label_map_sub[st]], idx[label_map_main[row.main_theme]]] = round(w, 4)

# (b) Edge: main theme -> outcome
total_all_freq = df_sub["total_freq"].sum()
for m in main_themes:
    sub_rows = df_sub[df_sub["main_theme"] == m]
    theme_freq = sub_rows["total_freq"].sum()
    weighted_polarity = (sub_rows["total_freq"] * sub_rows["net_polarity"]).sum() / theme_freq
    w = (theme_freq / total_all_freq) * weighted_polarity
    W[idx[label_map_main[m]], idx[OUTCOME]] = round(w, 4)

# (c) Explicit, documented feedback loop: outcome -> sub-theme 3.1 "self-efficacy cycle"
#     (source: مفاهیم.docx, concept #10 - "the positive cycle between self-efficacy and
#      interest reinforces long-term learning effectiveness"; coefficient 0.5 to
#      moderate the indirect effect)
# NOTE: the string below matches the actual sub-theme label as extracted from the
# source document, so it must remain in Persian.
st_311 = [s for s in sub_themes if s.startswith("۳.۱")][0]
W[idx[OUTCOME], idx[label_map_sub[st_311]]] = round(sub_row[st_311].net_polarity * 0.5, 4)

# (d) Direct edge sub-theme -> outcome (since all 85 meta-synthesis concepts were
#     directly coded against "effectiveness")
for st in sub_themes:
    row = sub_row[st]
    w = round((row.total_freq / total_all_freq) * row.net_polarity, 4)
    i, j = idx[label_map_sub[st]], idx[OUTCOME]
    W[i, j] = W[i, j] + w if W[i, j] != 0 else w

# (e) Cross-theme lateral edges: TF-IDF semantic similarity (NLP/ML component)
def tokenize_fa(text):
    text = re.sub(r"[^\u0600-\u06FFA-Za-z\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]

sub_docs = [" ".join(c["concept"] + " " + c["desc"] for c in concepts if c["sub_theme"] == st)
            for st in sub_themes]
vec = TfidfVectorizer(tokenizer=tokenize_fa, lowercase=False, token_pattern=None)
X = vec.fit_transform(sub_docs)
sim = cosine_similarity(X)
sims_flat = sim[np.triu_indices(len(sub_themes), k=1)]
thr = sims_flat.mean() + 1.0 * sims_flat.std()
print(f"Semantic similarity (TF-IDF) threshold for accepting a lateral edge: {thr:.4f}")

cross_edges = []
for a in range(len(sub_themes)):
    for b in range(len(sub_themes)):
        if a == b:
            continue
        st_a, st_b = sub_themes[a], sub_themes[b]
        if sub_row[st_a].main_theme == sub_row[st_b].main_theme:
            continue
        s = sim[a, b]
        if s >= thr:
            pol_a, pol_b = sub_row[st_a].net_polarity, sub_row[st_b].net_polarity
            sign = 1 if (pol_a * pol_b) >= 0 else -1
            i, j = idx[label_map_sub[st_a]], idx[label_map_sub[st_b]]
            if W[i, j] == 0:
                W[i, j] = round(s * sign, 4)
                cross_edges.append((st_a, st_b, s, W[i, j]))
print(f"Number of lateral edges discovered by TF-IDF: {len(cross_edges)}")

pd.DataFrame(W, index=all_nodes, columns=all_nodes).to_csv("table2_weight_matrix.csv", encoding="utf-8-sig")
n_edges = int((W != 0).sum())
density = n_edges / (n * (n - 1))
print(f"Nodes={n} | Edges={n_edges} | Graph density={density:.4f}")

# %% [7] FCM inference engine (damped Kosko rule) and scenario analysis --------------
LAM, GAMMA = 0.7, 0.1  # selected from the sensitivity analysis below (section 8): unsaturated stable point

def sigmoid(x, lam=LAM):
    return 1.0 / (1.0 + np.exp(-lam * x))

def run_fcm(A0, clamp_idx=(), clamp_val=(), lam=LAM, gamma=GAMMA, max_iter=300, tol=1e-7):
    A = A0.copy()
    for ci, cv in zip(clamp_idx, clamp_val):
        A[ci] = cv
    history = [A.copy()]
    converged_at = max_iter
    for t in range(max_iter):
        raw = gamma * A + A @ W
        A_new = sigmoid(raw, lam)
        for ci, cv in zip(clamp_idx, clamp_val):
            A_new[ci] = cv
        history.append(A_new.copy())
        if np.linalg.norm(A_new - A) < tol:
            A = A_new; converged_at = t + 1; break
        A = A_new
    return A, np.array(history), converged_at

def clampnodes(labels, val):
    ids = [idx[label2id[l]] for l in labels]
    return ids, [val] * len(ids)

A0 = np.full(n, 0.5)
A_base, hist_base, it_base = run_fcm(A0.copy())

facilitator_subs = df_sub[df_sub.net_polarity > 0].sub_theme.tolist()
ci1, cv1 = clampnodes(facilitator_subs, 0.9)
A_s1, hist_s1, it_s1 = run_fcm(A0.copy(), ci1, cv1)

barrier_subs = df_sub[df_sub.net_polarity < 0].sub_theme.tolist()
ci_b, cv_b = clampnodes(barrier_subs, 0.9)
ci_f, cv_f = clampnodes(facilitator_subs, 0.2)
A_s2, hist_s2, it_s2 = run_fcm(A0.copy(), ci_b + ci_f, cv_b + cv_f)

# NOTE: these labels match actual sub-theme text extracted from the source
# document and must remain in Persian for the lookup to work.
s3_labels = ["۳.۱ چرخه خودکارآمدی، انگیزش درونی و درگیری عاطفی",
             "۲.۲ بازخورد بلادرنگ سیستم و چرخه‌های اشکال‌زدایی"]
ci3, cv3 = clampnodes(s3_labels, 0.9)
A_s3, hist_s3, it_s3 = run_fcm(A0.copy(), ci3, cv3)

s4_labels = ["۴.۲ محدودیت‌های لجستیکی، زمانی و اقتصادی"]
ci4, cv4 = clampnodes(s4_labels, 0.9)
A_s4, hist_s4, it_s4 = run_fcm(A0.copy(), ci4, cv4)

scenarios = {
    "Baseline (no intervention)": (A_base, it_base),
    "Best case (full facilitator activation)": (A_s1, it_s1),
    "Worst case (barrier dominance)": (A_s2, it_s2),
    "Targeted positive intervention (self-efficacy+feedback)": (A_s3, it_s3),
    "Targeted negative intervention (resource constraints)": (A_s4, it_s4),
}
rows = [{"Scenario": name, "Convergence iterations": it, "Final effectiveness": round(A[idx[OUTCOME]], 4),
         "Difference from baseline": round(A[idx[OUTCOME]] - A_base[idx[OUTCOME]], 4)}
        for name, (A, it) in scenarios.items()]
df_scn = pd.DataFrame(rows)
print(df_scn.to_string(index=False))
df_scn.to_csv("table3_scenarios.csv", index=False, encoding="utf-8-sig")

# %% [8] Sensitivity analysis of inference parameters (λ, γ) -------------------------------
sens_rows = []
for lam in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
    for gamma in [0.05, 0.1, 0.2, 0.3]:
        A_b, _, it_b = run_fcm(A0.copy(), lam=lam, gamma=gamma)
        sens_rows.append({"lambda": lam, "gamma": gamma, "iters": it_b,
                           "effectiveness_base": round(A_b[idx[OUTCOME]], 4)})
df_sens = pd.DataFrame(sens_rows)
df_sens.to_csv("table4_sensitivity.csv", index=False, encoding="utf-8-sig")
print(df_sens.to_string(index=False))

# %% [9] Network centrality metrics (transmitter/receiver/ordinary per Kosko's theory) -----------
out_degree = np.sum(np.abs(W), axis=1)
in_degree = np.sum(np.abs(W), axis=0)
rows = []
for k in all_nodes:
    i = idx[k]; od, idg = out_degree[i], in_degree[i]
    role = ("Transmitter" if od > 0 and idg == 0 else
             "Receiver" if od == 0 and idg > 0 else "Ordinary")
    rows.append({"node": k, "label": node_labels[k], "out_degree": round(od, 4),
                 "in_degree": round(idg, 4), "centrality": round(od + idg, 4), "role": role})
df_cent = pd.DataFrame(rows).sort_values("centrality", ascending=False)
df_cent.to_csv("table5_centrality.csv", index=False, encoding="utf-8-sig")
print(df_cent.to_string(index=False))

# %% [10] Statistical validation: correlation of FCM centrality with raw meta-synthesis frequency --------
sub_cent = df_cent[df_cent["node"].str.startswith("S")]
merged = sub_cent.merge(df_sub, left_on="label", right_on="sub_theme")
r_p, p_p = stats.pearsonr(merged["centrality"], merged["total_freq"])
r_s, p_s = stats.spearmanr(merged["centrality"], merged["total_freq"])
print(f"\nPearson correlation (centrality ~ raw frequency) = {r_p:.4f}  (p={p_p:.4f})")
print(f"Spearman correlation (centrality ~ raw frequency) = {r_s:.4f}  (p={p_s:.4f})")
with open("table6_validation.txt", "w", encoding="utf-8") as f:
    f.write(f"Pearson r={r_p:.4f}, p={p_p:.4f}\nSpearman rho={r_s:.4f}, p={p_s:.4f}\nn={len(merged)}\n")

# %% [11] Plot figures ---------------------------------------------------------
short_labels = {
    "M0": "M1: Curriculum Design", "M1": "M2: Human-Robot Ergonomics",
    "M2": "M3: Learner Cognitive/Affective", "M3": "M4: Ecosystem & Educator Competence",
    "S4": "S1.1: Interdisciplinary Integration", "S5": "S1.2: Cognitive Scaffolding",
    "S6": "S1.3: Collaborative Co-creation", "S7": "S2.1: Embodiment/Tangibility",
    "S8": "S2.2: Real-time Feedback", "S9": "S2.3: Technical Complexity Mgmt",
    "S10": "S3.1: Self-efficacy/Motivation Cycle", "S11": "S3.2: Gender Gap",
    "S12": "S3.3: Prior Knowledge/Readiness", "S13": "S4.1: TPACK/Teacher Self-efficacy",
    "S14": "S4.2: Logistic/Economic Constraints", "OUT": "OUTCOME: STEM Learning Effectiveness",
}

# Figure 1: FCM network
G = nx.DiGraph()
for k in all_nodes: G.add_node(k)
for i in range(n):
    for j in range(n):
        if W[i, j] != 0: G.add_edge(all_nodes[i], all_nodes[j], weight=W[i, j])
pos = {"M0": (-2, 2), "M1": (-2, 0.7), "M2": (-2, -0.7), "M3": (-2, -2),
       "S4": (-4.2, 2.7), "S5": (-4.2, 2.0), "S6": (-4.2, 1.3), "S7": (-4.2, 0.35),
       "S8": (-4.2, -0.35), "S9": (-4.2, -1.0), "S10": (-4.2, -1.6), "S11": (-4.2, -2.3),
       "S12": (-4.2, -3.0), "S13": (-4.2, -3.6), "S14": (-4.2, -4.3), "OUT": (2.2, -0.6)}
node_colors = ["#4C72B0" if k.startswith("M") else "#C44E52" if k == "OUT" else "#55A868" for k in all_nodes]
fig, ax = plt.subplots(figsize=(16, 11))
edge_colors = ["#2E8B57" if G[u][v]["weight"] > 0 else "#B22222" for u, v in G.edges()]
edge_widths = [1 + 4 * abs(G[u][v]["weight"]) for u, v in G.edges()]
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1800, ax=ax, edgecolors="black", linewidths=0.8)
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, arrows=True, arrowsize=14,
                        connectionstyle="arc3,rad=0.08", ax=ax, alpha=0.75)
nx.draw_networkx_labels(G, pos, labels={k: short_labels[k] for k in all_nodes}, font_size=8, ax=ax)
ax.set_title("Fuzzy Cognitive Map of Factors Affecting Educational Robot Effectiveness in STEM Learning")
ax.axis("off"); plt.tight_layout(); plt.savefig("fig1_fcm_network.png", dpi=200); plt.close()

# Figure 2: weight heatmap
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(W, cmap="RdYlGn", vmin=-1, vmax=1)
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels([short_labels[k] for k in all_nodes], rotation=90, fontsize=8)
ax.set_yticklabels([short_labels[k] for k in all_nodes], fontsize=8)
plt.colorbar(im, ax=ax, label="Edge weight")
ax.set_title("FCM Adjacency (Weight) Matrix"); plt.tight_layout()
plt.savefig("fig2_weight_heatmap.png", dpi=200); plt.close()

# Figure 3: scenario comparison
scn_en = ["Baseline", "Best-case", "Worst-case", "Targeted +", "Targeted -"]
vals = df_scn["Final effectiveness"].tolist()
colors_scn = ["#4C72B0", "#2E8B57", "#B22222", "#55A868", "#DD8452"]
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(scn_en, vals, color=colors_scn, edgecolor="black")
ax.axhline(vals[0], color="gray", linestyle="--", linewidth=1)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.4f}", ha="center")
ax.set_ylabel("Steady-state effectiveness"); ax.set_ylim(0, 0.8)
ax.set_title("FCM Scenario Analysis"); plt.tight_layout()
plt.savefig("fig3_scenario_comparison.png", dpi=200); plt.close()

# Figure 4: convergence curves
fig, ax = plt.subplots(figsize=(10, 6))
for h, lab, col in zip([hist_base, hist_s1, hist_s2, hist_s3, hist_s4], scn_en, colors_scn):
    ax.plot(range(len(h)), h[:, idx[OUTCOME]], marker="o", markersize=3, label=lab, color=col)
ax.set_xlabel("Iteration"); ax.set_ylabel("Activation of OUTCOME node")
ax.set_title("Convergence Trajectories across Scenarios"); ax.legend()
plt.tight_layout(); plt.savefig("fig4_convergence.png", dpi=200); plt.close()

# Figure 5: centrality
df_cent_sorted = df_cent.sort_values("centrality", ascending=True)
fig, ax = plt.subplots(figsize=(10, 8))
y = range(len(df_cent_sorted))
ax.barh(y, df_cent_sorted["out_degree"], color="#2E8B57", label="Out-degree")
ax.barh(y, df_cent_sorted["in_degree"], left=df_cent_sorted["out_degree"], color="#4C72B0", label="In-degree")
ax.set_yticks(y); ax.set_yticklabels([short_labels[k] for k in df_cent_sorted["node"]], fontsize=9)
ax.set_xlabel("Centrality"); ax.set_title("Node Centrality Decomposition"); ax.legend()
plt.tight_layout(); plt.savefig("fig5_centrality.png", dpi=200); plt.close()

# Figure 6: sensitivity
fig, ax = plt.subplots(figsize=(9, 6))
for gamma in sorted(df_sens["gamma"].unique()):
    sub = df_sens[df_sens["gamma"] == gamma]
    ax.plot(sub["lambda"], sub["effectiveness_base"], marker="o", label=f"γ={gamma}")
ax.set_xlabel("Sigmoid steepness (λ)"); ax.set_ylabel("Baseline steady-state effectiveness")
ax.set_title("Sensitivity Analysis of FCM Parameters"); ax.legend(title="Memory coeff.")
plt.tight_layout(); plt.savefig("fig6_sensitivity.png", dpi=200); plt.close()

print("\nAll six figures and six tables were successfully generated and saved.")

# %% [12] (Optional) LLM-assisted semantic edge-weight moderation module (Claude API) ----
# This section only activates if an API key is present, and serves purely as a
# secondary semantic validation layer (not a replacement for the statistical
# weights). Running it is optional.
"""
import os, requests
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if API_KEY:
    def llm_edge_check(concept_a, concept_b):
        prompt = (f"Rate the causal semantic relationship between these two educational-robotics "
                  f"themes on a scale from -1 (strong inhibiting) to +1 (strong facilitating), "
                  f"0 if unrelated. Respond with ONLY a number.\nA: {concept_a}\nB: {concept_b}")
        r = requests.post("https://api.anthropic.com/v1/messages",
                           headers={"x-api-key": API_KEY, "anthropic-version": "2023-06-01",
                                    "content-type": "application/json"},
                           json={"model": "claude-sonnet-4-6", "max_tokens": 10,
                                 "messages": [{"role": "user", "content": prompt}]})
        return r.json()
    # Example usage (disabled by default):
    # print(llm_edge_check(sub_themes[0], sub_themes[5]))
else:
    print("Note: ANTHROPIC_API_KEY is not set; the LLM semantic-moderation module was skipped "
          "(this study's main results rely solely on TF-IDF and real statistics).")
"""
