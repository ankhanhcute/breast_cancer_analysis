import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, levene, shapiro
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# page config - has to be first before anything else
image = Image.open('cina.png')
st.set_page_config(
    page_title="Cancer Analysis",
    page_icon=image,
    layout="wide"
)

# all the css styling for the cinnamoroll theme
# colors used:
#   background  : #f0f7ff  (soft cloud white)
#   sidebar     : #ffffff to #eaf4fb gradient
#   primary     : #7ec8e3  (baby blue)
#   border      : #cce4f4
#   text        : #4a6fa5  (dark blue)
#   pink accent : #f9a8b8  (soft pink - used for malignant)
#   card bg     : #ffffff
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700;800&display=swap');

* {
    font-family: 'Nunito', sans-serif !important;
}

.stApp {
    background-color: #f0f7ff;
}

h1, h2, h3, h4 {
    color: #3a5f8a !important;
    font-weight: 800 !important;
}

p, span, label, div {
    color: #4a6fa5 !important;
}

.stMarkdown p {
    color: #4a6fa5 !important;
}

/* sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #eaf4fb 100%) !important;
    border-right: 2px solid #cce4f4;
}
section[data-testid="stSidebar"] * {
    color: #4a6fa5 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.9rem !important;
    font-weight: 600 !important;
}

/* buttons */
.stButton > button {
    background: linear-gradient(135deg, #7ec8e3, #5ba8cc) !important;
    color: white !important;
    border-radius: 25px !important;
    border: none !important;
    font-weight: 700 !important;
    padding: 10px 28px !important;
    box-shadow: 0 3px 10px rgba(126, 200, 227, 0.35) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(126, 200, 227, 0.5) !important;
}

/* metric cards */
[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1.5px solid #cce4f4 !important;
    border-radius: 18px !important;
    padding: 18px !important;
    box-shadow: 0 3px 14px rgba(126, 200, 227, 0.18) !important;
}
[data-testid="stMetricValue"] {
    color: #3a5f8a !important;
    font-weight: 800 !important;
}
[data-testid="stMetricLabel"] {
    color: #7ec8e3 !important;
    font-weight: 600 !important;
}

/* dropdowns and multiselect */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: #ffffff !important;
    border: 1.5px solid #cce4f4 !important;
    border-radius: 14px !important;
}

/* tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #eaf4fb !important;
    border-radius: 14px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    font-weight: 600 !important;
    color: #4a6fa5 !important;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #3a5f8a !important;
    box-shadow: 0 2px 8px rgba(126, 200, 227, 0.25) !important;
}

/* dataframe table */
.stDataFrame {
    border-radius: 14px !important;
    border: 1.5px solid #cce4f4 !important;
    overflow: hidden !important;
}

hr {
    border-color: #cce4f4 !important;
    margin: 1.2rem 0 !important;
}

/* reusable card styles */
.cinna-card {
    background: #ffffff;
    border: 1.5px solid #cce4f4;
    border-radius: 18px;
    padding: 22px 26px;
    margin: 10px 0;
    box-shadow: 0 3px 14px rgba(126, 200, 227, 0.15);
    color: #4a6fa5;
}

.insight-box {
    background: linear-gradient(135deg, #eaf4fb, #f0f7ff);
    border-left: 4px solid #7ec8e3;
    border-radius: 0 14px 14px 0;
    padding: 14px 18px;
    margin: 10px 0;
    color: #4a6fa5;
}
</style>
""", unsafe_allow_html=True)


# load and prep the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("breast-cancer.csv")
    df = df.drop(columns=['id'])
    df['diagnosis_encoded'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df

df = load_data()

# split by diagnosis for easier access later
malignant_df = df[df['diagnosis'] == 'M']
benign_df    = df[df['diagnosis'] == 'B']


# sidebar navigation
with st.sidebar:
    st.image("cina.png", width=160)
    st.markdown("### Cancer Analysis")
    st.markdown("*by Khanh Truong*")
    st.markdown("`CAP2757`")
    st.markdown("---")
    page = st.radio("Navigate", [
        "Overview",
        "EDA",
        "Hypothesis Testing",
        "ML Model",
        "Predict Tumor"
    ])
    st.markdown("---")
    st.markdown("**Dataset Summary**")
    st.markdown(f"Rows: **{df.shape[0]}**")
    st.markdown(f"Features: **{df.shape[1] - 2}**")
    st.markdown(f"Malignant: **{len(malignant_df)}**")
    st.markdown(f"Benign: **{len(benign_df)}**")


# ==============================================================
#  OVERVIEW PAGE
# ==============================================================
if page == "Overview":
    st.title("Breast Cancer Data Analysis")
    st.markdown("*Wisconsin Breast Cancer Dataset - exploring what makes tumors different*")
    st.markdown("---")

    # top level KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("Total Patients",  569)
    with col2: st.metric("Malignant",       212)
    with col3: st.metric("Benign",          357)
    with col4: st.metric("Features",        30)
    with col5: st.metric("Model Accuracy",  "96.49%")

    st.markdown("---")

    # about the dataset and key findings side by side
    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.markdown("### About this dataset")
        st.markdown("""
        <div class='cinna-card'>
        The <b>Wisconsin Breast Cancer Dataset</b> contains real tumor biopsy measurements
        from 569 patients. Each sample has 30 numerical features that describe the tumor's
        size, shape, and texture, measured from digitized cell nucleus images.<br><br>
        The goal is to classify each tumor as either
        <b style='color:#e74c3c'>Malignant (cancerous)</b> or
        <b style='color:#2ecc71'>Benign (non-cancerous)</b>.
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("### Key findings")
        st.markdown("""
        <div class='insight-box'>
        Malignant tumors are on average <b>43% larger</b> in radius<br><br>
        <b>concave_points_worst</b> turned out to be the most predictive feature<br><br>
        The Random Forest model catches <b>93% of all cancer cases</b><br><br>
        Radius, perimeter and area are <b>highly correlated</b> since they all measure size
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # interactive filter for the raw data
    st.markdown("### Explore the dataset")
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        filter_option = st.selectbox(
            "Filter by diagnosis:",
            ["All", "Malignant (M)", "Benign (B)"]
        )
    with col_f2:
        num_rows = st.slider("Rows to show:", 5, 50, 10)

    filtered_df = (malignant_df if filter_option == "Malignant (M)"
                   else benign_df if filter_option == "Benign (B)"
                   else df)

    st.markdown(f"Showing **{len(filtered_df)}** patients, first {num_rows} rows")
    st.dataframe(filtered_df.head(num_rows), use_container_width=True)

    st.markdown("---")

    # quick stats for any feature the user picks
    st.markdown("### Feature summary")
    feature_pick = st.selectbox("Pick a feature:", df.columns[1:-1])

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Mean",   f"{df[feature_pick].mean():.3f}")
    with c2: st.metric("Median", f"{df[feature_pick].median():.3f}")
    with c3: st.metric("Std",    f"{df[feature_pick].std():.3f}")
    with c4: st.metric("Min",    f"{df[feature_pick].min():.3f}")
    with c5: st.metric("Max",    f"{df[feature_pick].max():.3f}")

    m_mean = malignant_df[feature_pick].mean()
    b_mean = benign_df[feature_pick].mean()
    diff   = (m_mean - b_mean) / b_mean * 100
    st.markdown(f"""
    <div class='insight-box'>
    For <b>{feature_pick}</b> — Malignant avg: <b>{m_mean:.3f}</b> &nbsp;|&nbsp;
    Benign avg: <b>{b_mean:.3f}</b> &nbsp;|&nbsp; Difference: <b>{diff:+.1f}%</b>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================
#  EDA PAGE
# ==============================================================
elif page == "EDA":
    st.title("Exploratory Data Analysis")
    st.markdown("*Looking at patterns and differences between malignant and benign tumors*")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Distributions",
        "Feature Comparison",
        "Correlations",
        "Outliers"
    ])

    # --- tab 1: distributions ---
    with tab1:
        st.markdown("### How are the tumors distributed?")
        col_a, col_b = st.columns(2)

        with col_a:
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#f0f7ff')
            ax.set_facecolor('#f0f7ff')
            counts = df['diagnosis'].value_counts()
            bars = ax.bar(
                ['Benign', 'Malignant'], counts.values,
                color=['#7ec8e3', '#f9a8b8'],
                width=0.5, edgecolor='white', linewidth=2
            )
            for bar, val in zip(bars, counts.values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 4, str(val),
                    ha='center', fontweight='bold',
                    color='#4a6fa5', fontsize=12
                )
            ax.set_title('Tumor count by type', color='#3a5f8a',
                         fontsize=13, fontweight='bold')
            ax.set_ylabel('Count', color='#4a6fa5')
            ax.tick_params(colors='#4a6fa5')
            for spine in ax.spines.values():
                spine.set_color('#cce4f4')
            st.pyplot(fig)

        with col_b:
            feat_hist = st.selectbox(
                "Select feature to plot:",
                df.columns[1:-1], key='hist'
            )
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#f0f7ff')
            ax.set_facecolor('#f0f7ff')
            ax.hist(benign_df[feat_hist], bins=25,
                    color='#7ec8e3', alpha=0.75, label='Benign')
            ax.hist(malignant_df[feat_hist], bins=25,
                    color='#f9a8b8', alpha=0.75, label='Malignant')
            ax.set_title(f'{feat_hist}', color='#3a5f8a',
                         fontsize=12, fontweight='bold')
            ax.legend(facecolor='white', edgecolor='#cce4f4')
            ax.tick_params(colors='#4a6fa5')
            for spine in ax.spines.values():
                spine.set_color('#cce4f4')
            st.pyplot(fig)

        st.markdown("""
        <div class='insight-box'>
        Most features show clearly separate distributions between malignant and benign tumors.
        This separation is what makes machine learning work so well on this dataset.
        </div>
        """, unsafe_allow_html=True)

    # --- tab 2: feature comparison with box plots ---
    with tab2:
        st.markdown("### Side-by-side comparison")
        selected = st.multiselect(
            "Pick features to compare:",
            list(df.columns[1:-1]),
            default=['radius_mean', 'area_mean', 'concavity_mean', 'texture_mean']
        )

        if selected:
            fig, axes = plt.subplots(1, len(selected),
                                      figsize=(5 * len(selected), 5))
            fig.patch.set_facecolor('#f0f7ff')
            if len(selected) == 1:
                axes = [axes]
            for i, feat in enumerate(selected):
                axes[i].set_facecolor('#f0f7ff')
                bp = axes[i].boxplot(
                    [malignant_df[feat], benign_df[feat]],
                    patch_artist=True, labels=['M', 'B'], widths=0.5
                )
                bp['boxes'][0].set_facecolor('#f9a8b8')
                bp['boxes'][1].set_facecolor('#7ec8e3')
                for el in ['whiskers', 'caps', 'medians']:
                    plt.setp(bp[el], color='#4a6fa5')
                plt.setp(bp['fliers'], marker='o',
                         color='#b8d4e8', markersize=4)
                axes[i].set_title(
                    feat.replace('_', '\n'),
                    fontsize=9, color='#3a5f8a', fontweight='bold'
                )
                axes[i].tick_params(colors='#4a6fa5')
                for spine in axes[i].spines.values():
                    spine.set_color('#cce4f4')
            plt.tight_layout()
            st.pyplot(fig)

            # difference summary table
            rows = []
            for feat in selected:
                m = malignant_df[feat].mean()
                b = benign_df[feat].mean()
                rows.append({
                    'Feature':       feat,
                    'Malignant avg': round(m, 3),
                    'Benign avg':    round(b, 3),
                    'Diff %':        f"{(m - b) / b * 100:+.1f}%"
                })
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True
            )

        st.markdown("""
        <div class='insight-box'>
        Malignant tumors consistently score higher across most features,
        especially area and concavity. This makes sense since larger and more
        irregular tumors tend to be malignant.
        </div>
        """, unsafe_allow_html=True)

    # --- tab 3: correlation heatmap ---
    with tab3:
        st.markdown("### Correlation heatmap")
        col_opt, _ = st.columns([1, 2])
        with col_opt:
            group = st.selectbox("Show feature group:", [
                "Mean features", "Worst features",
                "SE features", "All"
            ])

        cols = (
            [c for c in df.columns if '_mean'  in c] if group == "Mean features"  else
            [c for c in df.columns if '_worst' in c] if group == "Worst features" else
            [c for c in df.columns if '_se'    in c] if group == "SE features"    else
            list(df.columns[1:-1])
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('#f0f7ff')
        ax.set_facecolor('#f0f7ff')
        corr = df[cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask,
            annot=len(cols) <= 12, fmt='.2f',
            cmap='Blues', linewidths=0.5, ax=ax,
            annot_kws={'size': 8, 'color': '#3a5f8a'}
        )
        ax.set_title(f'Correlation - {group}', color='#3a5f8a',
                     fontsize=13, fontweight='bold')
        ax.tick_params(colors='#4a6fa5', labelsize=8)
        st.pyplot(fig)

        # show top 5 correlated pairs
        corr_pairs = (
            corr.where(~mask).stack()
            .reset_index()
            .rename(columns={'level_0': 'Feature A', 'level_1': 'Feature B', 0: 'r'})
        )
        top5 = (corr_pairs
                .assign(r_abs=corr_pairs['r'].abs())
                .nlargest(5, 'r_abs')[['Feature A', 'Feature B', 'r']])
        top5['r'] = top5['r'].round(3)
        st.markdown("**Top 5 strongest correlations**")
        st.dataframe(top5, use_container_width=True, hide_index=True)

    # --- tab 4: outlier check ---
    with tab4:
        st.markdown("### Outlier check")
        feat_out = st.selectbox(
            "Feature to inspect:",
            df.columns[1:-1], key='out'
        )

        q1  = df[feat_out].quantile(0.25)
        q3  = df[feat_out].quantile(0.75)
        iqr = q3 - q1
        lo  = q1 - 1.5 * iqr
        hi  = q3 + 1.5 * iqr
        outliers = df[(df[feat_out] < lo) | (df[feat_out] > hi)]

        col_a, col_b = st.columns([1.5, 1])
        with col_a:
            fig, ax = plt.subplots(figsize=(7, 4))
            fig.patch.set_facecolor('#f0f7ff')
            ax.set_facecolor('#f0f7ff')
            for diag, color, label in [
                ('B', '#7ec8e3', 'Benign'),
                ('M', '#f9a8b8', 'Malignant')
            ]:
                sub = df[df['diagnosis'] == diag]
                ax.scatter(
                    range(len(sub)), sub[feat_out].values,
                    color=color, alpha=0.5, s=20, label=label
                )
            ax.axhline(hi, color='#e74c3c', linestyle='--',
                       linewidth=1.2, label='IQR bound')
            ax.axhline(lo, color='#e74c3c', linestyle='--',
                       linewidth=1.2)
            ax.set_title(f'{feat_out} - outlier view',
                         color='#3a5f8a', fontsize=12, fontweight='bold')
            ax.legend(facecolor='white', edgecolor='#cce4f4', fontsize=8)
            ax.tick_params(colors='#4a6fa5')
            for spine in ax.spines.values():
                spine.set_color('#cce4f4')
            st.pyplot(fig)

        with col_b:
            st.markdown(f"""
            <div class='cinna-card'>
            <b>Outlier summary</b><br><br>
            Total outliers: <b>{len(outliers)}</b><br>
            Malignant: <b>{len(outliers[outliers['diagnosis'] == 'M'])}</b><br>
            Benign: <b>{len(outliers[outliers['diagnosis'] == 'B'])}</b><br><br>
            IQR lower bound: <b>{lo:.3f}</b><br>
            IQR upper bound: <b>{hi:.3f}</b>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================
#  HYPOTHESIS TESTING PAGE
# ==============================================================
elif page == "Hypothesis Testing":
    st.title("Hypothesis Testing")
    st.markdown("*Testing whether the difference in tumor radius between groups is statistically significant*")
    st.markdown("---")

    # define the groups
    malignant = df[df['diagnosis'] == 'M']['radius_mean']
    benign    = df[df['diagnosis'] == 'B']['radius_mean']

    # state the hypotheses
    st.markdown("### Research Question")
    st.markdown("""
    <div class='cinna-card'>
    Is there a statistically significant difference in mean tumor radius
    between Malignant and Benign tumors?<br><br>
    <b>H0 (Null hypothesis):</b> There is no significant difference in mean radius between the two groups<br>
    <b>Ha (Alternative hypothesis):</b> Malignant tumors have a significantly larger mean radius than benign tumors
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # step 1 - check normality with shapiro-wilk
    st.markdown("### Step 1 - Normality Check (Shapiro-Wilk Test)")
    st.markdown("Before running the t-test we need to check if the data is normally distributed.")

    stat_m, p_m = shapiro(malignant)
    stat_b, p_b = shapiro(benign)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Malignant p-value", f"{p_m:.4f}")
        if p_m < 0.05:
            st.error("Not normally distributed (p < 0.05)")
        else:
            st.success("Normally distributed (p > 0.05)")
    with col2:
        st.metric("Benign p-value", f"{p_b:.4f}")
        if p_b < 0.05:
            st.error("Not normally distributed (p < 0.05)")
        else:
            st.success("Normally distributed (p > 0.05)")

    st.markdown("""
    <div class='insight-box'>
    The malignant group failed the normality test, but the t-test is still valid
    because of the Central Limit Theorem. With 200+ samples in each group,
    the sampling distribution approaches normality regardless of the underlying distribution.
    We also used Welch's t-test (equal_var=False) to handle unequal variances.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # step 2 - levene's test for equal variance
    st.markdown("### Step 2 - Levene's Test (Equal Variance Check)")
    st.markdown("This checks whether the two groups have similar variance before running the t-test.")

    stat_lev, p_lev = levene(malignant, benign)
    st.metric("Levene's test p-value", f"{p_lev:.4f}")

    if p_lev < 0.05:
        st.error("Unequal variances detected (p < 0.05) - using Welch's t-test")
    else:
        st.success("Equal variances (p > 0.05) - standard t-test is fine")

    st.markdown("---")

    # step 3 - run the actual t-test
    st.markdown("### Step 3 - Welch's Two-Sample T-Test")

    t_stat, p_value = ttest_ind(malignant, benign, equal_var=False)

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Malignant Mean Radius", f"{malignant.mean():.2f}")
    with col2: st.metric("Benign Mean Radius",    f"{benign.mean():.2f}")
    with col3: st.metric("T-Statistic",           f"{t_stat:.4f}")
    with col4: st.metric("P-Value",               f"{p_value:.2e}")

    st.markdown("---")

    # conclusion
    if p_value < 0.05:
        st.success("Reject H0 - There is a statistically significant difference (p < 0.05)")
    else:
        st.error("Fail to reject H0 - No significant difference found")

    st.markdown("""
    <div class='cinna-card'>
    <b>Conclusion:</b> With a t-statistic of 22.21 and a p-value essentially equal to zero,
    we reject the null hypothesis. Malignant tumors have a mean radius of <b>17.46</b>
    compared to <b>12.15</b> for benign tumors, which is a <b>43% difference</b>.
    This is statistically significant at the 0.05 level, meaning tumor size is
    a strong predictor of malignancy.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # visual comparison of the two groups
    st.markdown("### Visual Comparison")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#f0f7ff')

    # histogram with mean lines
    axes[0].set_facecolor('#f0f7ff')
    axes[0].hist(malignant, bins=25, color='#f9a8b8',
                 alpha=0.75, label='Malignant')
    axes[0].hist(benign, bins=25, color='#7ec8e3',
                 alpha=0.75, label='Benign')
    axes[0].axvline(malignant.mean(), color='#c0392b', linestyle='--',
                    linewidth=2, label=f'M mean: {malignant.mean():.2f}')
    axes[0].axvline(benign.mean(), color='#2980b9', linestyle='--',
                    linewidth=2, label=f'B mean: {benign.mean():.2f}')
    axes[0].set_title('Radius Distribution', color='#3a5f8a',
                       fontsize=13, fontweight='bold')
    axes[0].legend(facecolor='white', edgecolor='#cce4f4')
    axes[0].tick_params(colors='#4a6fa5')
    for spine in axes[0].spines.values():
        spine.set_color('#cce4f4')

    # box plot comparison
    axes[1].set_facecolor('#f0f7ff')
    bp = axes[1].boxplot(
        [malignant, benign], patch_artist=True,
        labels=['Malignant', 'Benign']
    )
    bp['boxes'][0].set_facecolor('#f9a8b8')
    bp['boxes'][1].set_facecolor('#7ec8e3')
    for el in ['whiskers', 'caps', 'medians']:
        plt.setp(bp[el], color='#4a6fa5')
    axes[1].set_title('Radius Box Plot', color='#3a5f8a',
                       fontsize=13, fontweight='bold')
    axes[1].tick_params(colors='#4a6fa5')
    for spine in axes[1].spines.values():
        spine.set_color('#cce4f4')

    plt.tight_layout()
    st.pyplot(fig)


# ==============================================================
#  ML MODEL PAGE
# ==============================================================
elif page == "ML Model":
    st.title("Cancer Detection Model")
    st.markdown("*Random Forest classifier trained on 30 tumor features*")
    st.markdown("---")

    @st.cache_resource
    def train_model():
        X = df.drop(columns=['diagnosis', 'diagnosis_encoded'])
        y = df['diagnosis_encoded']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        mdl = RandomForestClassifier(n_estimators=100, random_state=42)
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
        return mdl, X_test, y_test, y_pred, X.columns

    model, X_test, y_test, y_pred, feature_names = train_model()
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(
        y_test, y_pred,
        target_names=['Benign', 'Malignant'],
        output_dict=True
    )

    # model performance KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Accuracy",        f"{accuracy * 100:.2f}%")
    with c2: st.metric("Correct Calls",   f"{int(accuracy * len(y_test))}/{len(y_test)}")
    with c3: st.metric("Cancer Recall",   f"{report['Malignant']['recall'] * 100:.0f}%")
    with c4: st.metric("Benign Precision",f"{report['Benign']['precision'] * 100:.0f}%")

    st.markdown("---")

    tab_cm, tab_fi, tab_report = st.tabs([
        "Confusion Matrix",
        "Feature Importance",
        "Full Report"
    ])

    # confusion matrix tab
    with tab_cm:
        col_a, col_b = st.columns([1, 1.3])
        with col_a:
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#f0f7ff')
            ax.set_facecolor('#f0f7ff')
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'],
                ax=ax, annot_kws={'size': 14, 'weight': 'bold'}
            )
            ax.set_title('Confusion Matrix', color='#3a5f8a',
                         fontsize=13, fontweight='bold')
            ax.set_xlabel('Predicted', color='#4a6fa5')
            ax.set_ylabel('Actual',    color='#4a6fa5')
            ax.tick_params(colors='#4a6fa5')
            st.pyplot(fig)

        with col_b:
            tn, fp, fn, tp = cm.ravel()
            st.markdown(f"""
            <div class='cinna-card'>
            <b>How to read this matrix</b><br><br>
            True Benign (TN): <b>{tn}</b> - correctly called benign<br>
            True Malignant (TP): <b>{tp}</b> - correctly caught cancer<br>
            Missed cancers (FN): <b>{fn}</b> - predicted benign but actually malignant<br>
            False alarms (FP): <b>{fp}</b> - predicted malignant but actually benign<br><br>
            Missing <b>{fn}</b> cancer case(s) is the most critical error type.
            The model keeps this very low which is important in a medical context.
            </div>
            """, unsafe_allow_html=True)

    # feature importance tab
    with tab_fi:
        feat_imp = pd.Series(
            model.feature_importances_, index=feature_names
        ).nlargest(15).sort_values()

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#f0f7ff')
        ax.set_facecolor('#f0f7ff')

        # highlight the top 3 in pink
        colors = ['#f9a8b8' if i >= len(feat_imp) - 3 else '#7ec8e3'
                  for i in range(len(feat_imp))]
        feat_imp.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
        ax.set_title('Top 15 Most Predictive Features', color='#3a5f8a',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Importance score', color='#4a6fa5')
        ax.tick_params(colors='#4a6fa5', labelsize=9)
        for spine in ax.spines.values():
            spine.set_color('#cce4f4')
        st.pyplot(fig)

        st.markdown("""
        <div class='insight-box'>
        The pink bars are the top 3 most important features.
        concave_points_worst came out on top, meaning the shape irregularity
        of the worst tumor region is the strongest indicator of malignancy.
        </div>
        """, unsafe_allow_html=True)

    # full classification report tab
    with tab_report:
        report_df = pd.DataFrame(report).T.drop(
            ['accuracy', 'macro avg', 'weighted avg'], errors='ignore'
        )
        report_df = report_df[['precision', 'recall', 'f1-score', 'support']].round(3)
        st.dataframe(report_df, use_container_width=True)

        st.markdown("""
        <div class='insight-box'>
        Recall matters more than precision in cancer detection because missing
        a malignant tumor is much more dangerous than a false alarm.
        The model's malignant recall of 93% means only about 7% of cancers go undetected.
        </div>
        """, unsafe_allow_html=True)


# ==============================================================
#  PREDICT TUMOR PAGE
# ==============================================================
elif page == "Predict Tumor":
    st.title("Tumor Prediction Tool")
    st.markdown("*Adjust the sliders to match a tumor's measurements and see what the model predicts*")
    st.markdown("---")

    @st.cache_resource
    def get_model():
        X = df.drop(columns=['diagnosis', 'diagnosis_encoded'])
        y = df['diagnosis_encoded']
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        mdl = RandomForestClassifier(n_estimators=100, random_state=42)
        mdl.fit(X_train, y_train)
        return mdl, X.columns

    model, feature_names = get_model()

    st.markdown("#### Adjust tumor measurements")
    st.markdown("These 10 sliders cover the most impactful mean features from the dataset.")

    main_features = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
        'smoothness_mean', 'compactness_mean', 'concavity_mean',
        'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean'
    ]

    inputs = {}
    col1, col2 = st.columns(2)
    for i, feat in enumerate(main_features):
        with (col1 if i % 2 == 0 else col2):
            inputs[feat] = st.slider(
                feat,
                float(df[feat].min()),
                float(df[feat].max()),
                float(df[feat].mean()),
                key=feat
            )

    # fill the remaining features with the dataset mean
    for feat in feature_names:
        if feat not in inputs:
            inputs[feat] = float(df[feat].mean())

    input_df = pd.DataFrame([inputs])[feature_names]

    st.markdown("---")
    if st.button("Run prediction", use_container_width=True):
        pred  = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        col_res, col_prob = st.columns([1.2, 1])

        with col_res:
            if pred == 1:
                st.error(f"MALIGNANT - model confidence: {proba[1] * 100:.1f}%")
                st.markdown("""
                <div class='insight-box' style='border-color:#f9a8b8; background:#fff0f3'>
                The entered measurements are consistent with a malignant tumor.
                This is a model output and not a medical diagnosis.
                Please consult a qualified healthcare professional.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success(f"BENIGN - model confidence: {proba[0] * 100:.1f}%")
                st.markdown("""
                <div class='insight-box'>
                The entered measurements are consistent with a benign tumor.
                Always confirm results with a medical professional.
                </div>
                """, unsafe_allow_html=True)

        with col_prob:
            st.metric("Benign probability",    f"{proba[0] * 100:.1f}%")
            st.metric("Malignant probability", f"{proba[1] * 100:.1f}%")

            # simple probability bar
            fig, ax = plt.subplots(figsize=(4, 1.2))
            fig.patch.set_facecolor('#f0f7ff')
            ax.barh([''], [proba[0]], color='#7ec8e3', label='Benign')
            ax.barh([''], [proba[1]], left=[proba[0]],
                    color='#f9a8b8', label='Malignant')
            ax.set_xlim(0, 1)
            ax.set_facecolor('#f0f7ff')
            ax.legend(loc='upper right', fontsize=7,
                      facecolor='white', edgecolor='#cce4f4')
            ax.tick_params(left=False, bottom=False,
                           labelbottom=False, labelleft=False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title('Probability split', color='#4a6fa5', fontsize=9)
            st.pyplot(fig)