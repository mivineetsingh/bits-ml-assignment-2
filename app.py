import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, confusion_matrix,
    roc_curve, precision_recall_curve
)

# 1. Page Configuration
st.set_page_config(
    page_title="Breast Cancer AI Diagnostics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Modern Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background with Gradient */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stFileUploader label {
        color: white !important;
        font-weight: 600;
    }
    
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: transform 0.3s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.25);
    }
    
    div[data-testid="stMetric"] * {
        color: #1e3c72 !important;
    }
    
    div[data-testid="stMetricLabel"] {
        font-weight: 600;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700;
    }
    
    div[data-testid="stMetricDelta"] {
        font-size: 0.85rem;
    }
    
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        font-size: 3rem !important;
        margin-bottom: 0.5rem;
    }
    
    h2, h3 {
        color: #2d3748;
        font-weight: 700;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 14px 32px;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.05rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 10px 10px 0 0;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 1.05rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 20px;
        border: 2px dashed rgba(255, 255, 255, 0.5);
    }
    
    /* Info/Warning Boxes */
    .stAlert {
        border-radius: 15px;
        border-left: 5px solid;
    }
    
    /* Dataframe Styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom Card Class */
    .custom-card {
        background: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin: 15px 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        margin: 10px 0;
    }
    
    .metric-card h3 {
        color: white;
        margin: 0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-card .value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 10px 0;
    }
    
    .metric-card .label {
        font-size: 0.85rem;
        opacity: 0.9;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Select box */
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Helper Functions
@st.cache_resource
def load_assets():
    try:
        models = {}
        model_list = [
            "Logistic_Regression", "Decision_Tree", "KNN", 
            "Naive_Bayes", "Random_Forest", "XGBoost"
        ]
        for name in model_list:
            models[name] = joblib.load(f'model/{name}.pkl')
        scaler = joblib.load('model/scaler.pkl')
        return models, scaler
    except FileNotFoundError as e:
        return None, None

def calculate_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }

def create_beautiful_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure with custom styling
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use a beautiful color palette
    sns.heatmap(cm, annot=False, fmt='d', cmap='RdPu', ax=ax,
                cbar_kws={'label': 'Count'}, square=True, linewidths=2, linecolor='white')
    
    # Add custom annotations with counts and percentages
    for i in range(2):
        for j in range(2):
            text = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
            ax.text(j + 0.5, i + 0.5, text,
                   ha="center", va="center",
                   color="white" if cm[i, j] > cm.max()/2 else "black",
                   fontsize=16, fontweight='bold')
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticklabels(['Malignant', 'Benign'], fontsize=11)
    ax.set_yticklabels(['Malignant', 'Benign'], fontsize=11)
    
    plt.tight_layout()
    return fig

def create_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot ROC curve with gradient-like effect
    ax.plot(fpr, tpr, color='#667eea', linewidth=3, label=f'ROC Curve (AUC = {auc:.3f})')
    ax.fill_between(fpr, tpr, alpha=0.3, color='#667eea')
    
    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier', alpha=0.5)
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve Analysis', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    return fig

def create_precision_recall_curve(y_true, y_prob):
    """Create a beautiful Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(recall, precision, color='#764ba2', linewidth=3, label='PR Curve')
    ax.fill_between(recall, precision, alpha=0.3, color='#764ba2')
    
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    return fig

def create_probability_distribution(y_true, y_prob):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separate probabilities by actual class
    malignant_probs = y_prob[y_true == 0]
    benign_probs = y_prob[y_true == 1]
    
    ax.hist(malignant_probs, bins=30, alpha=0.7, color='#EF4444', label='Malignant', edgecolor='black')
    ax.hist(benign_probs, bins=30, alpha=0.7, color='#10B981', label='Benign', edgecolor='black')
    
    ax.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def create_comparison_chart(comp_df, metric1='Accuracy', metric2='F1 Score'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Metric 1
    colors1 = plt.cm.Blues(np.linspace(0.4, 0.8, len(comp_df)))
    bars1 = ax1.bar(range(len(comp_df)), comp_df[metric1], color=colors1, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(comp_df)))
    ax1.set_xticklabels(comp_df.index, rotation=45, ha='right')
    ax1.set_ylabel(metric1, fontsize=11, fontweight='bold')
    ax1.set_title(f'{metric1} by Model', fontsize=13, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([comp_df[metric1].min() * 0.95, comp_df[metric1].max() * 1.05])
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Metric 2
    colors2 = plt.cm.Purples(np.linspace(0.4, 0.8, len(comp_df)))
    bars2 = ax2.bar(range(len(comp_df)), comp_df[metric2], color=colors2, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(comp_df)))
    ax2.set_xticklabels(comp_df.index, rotation=45, ha='right')
    ax2.set_ylabel(metric2, fontsize=11, fontweight='bold')
    ax2.set_title(f'{metric2} by Model', fontsize=13, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([comp_df[metric2].min() * 0.95, comp_df[metric2].max() * 1.05])
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    return fig

def create_radar_chart(comp_df):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    
    # Number of variables
    num_vars = len(metrics)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot each model
    colors = plt.cm.Set3(np.linspace(0, 1, len(comp_df)))
    
    for idx, (model_name, row) in enumerate(comp_df.iterrows()):
        values = [row[m] for m in metrics]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    # Fix axis to go in the right order
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.title('Multi-Metric Model Comparison', fontsize=14, fontweight='bold', pad=30)
    
    plt.tight_layout()
    return fig

# 4. Load Resources
models, scaler = load_assets()

if models is None:
    st.error("Critical Error: Models not found!")
    st.warning("Please ensure the 'model/' folder contains all .pkl files.")
    st.stop()

# 5. Sidebar
with st.sidebar:
    st.markdown("### **AI Diagnostics Control Panel**")
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "Upload Test Dataset",
        type=["csv"],
        help="Upload a CSV file with breast cancer features"
    )
    
    st.markdown("---")
    
    # Model info
    with st.expander("Available Models"):
        for model_name in models.keys():
            st.markdown(f"• {model_name.replace('_', ' ')}")
    
    st.markdown("---")
    st.caption("Built for BITS Pilani M.Tech (AIML)")
    st.caption("Assignment 2 - ML Model Evaluation")

# 6. Main App
# Header with custom styling
st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1>Breast Cancer AI Diagnostics</h1>
        <p style='font-size: 1.2rem; color: #4a5568;'>
            Advanced Machine Learning Model Evaluation Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

if uploaded_file is not None:
    # Load Data
    data = pd.read_csv(uploaded_file)
    
    # Handle Target
    if 'target' in data.columns:
        y_true = data['target']
        X = data.drop(columns=['target'])
        has_labels = True
    else:
        y_true = None
        X = data
        has_labels = False
    
    # Dataset Overview
    with st.expander("Dataset Overview", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Samples", len(data))
        col2.metric("Features", len(X.columns))
        if has_labels:
            col3.metric("Malignant Cases", int((y_true == 0).sum()))
            col4.metric("Benign Cases", int((y_true == 1).sum()))
    
    # Create Tabs
    tab1, tab2, tab3 = st.tabs(["Model Analysis", "Model Comparison", "Advanced Metrics"])
    
    # --- TAB 1: INDIVIDUAL ANALYSIS ---
    with tab1:
        st.markdown("### Single Model Deep Dive")
        st.markdown("---")
        
        selected_model_name = st.selectbox(
            "Select Model for Evaluation",
            list(models.keys()),
            format_func=lambda x: x.replace('_', ' ')
        )
        model = models[selected_model_name]
        
        # Preprocessing
        if selected_model_name in ["Logistic_Regression", "KNN"]:
            X_processed = scaler.transform(X)
        else:
            X_processed = X
        
        # Predictions
        y_pred = model.predict(X_processed)
        y_prob = model.predict_proba(X_processed)[:, 1]
        
        if has_labels:
            metrics = calculate_metrics(y_true, y_pred, y_prob)
            
            # Primary Metrics - Beautiful Cards
            st.markdown("#### Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Accuracy",
                    f"{metrics['Accuracy']:.2%}",
                    delta=f"{(metrics['Accuracy'] - 0.5):.1%} vs random"
                )
            
            with col2:
                st.metric(
                    "F1 Score",
                    f"{metrics['F1 Score']:.4f}",
                    delta="Harmonic mean"
                )
            
            with col3:
                st.metric(
                    "Precision",
                    f"{metrics['Precision']:.4f}",
                    delta="TP / (TP + FP)"
                )
            
            with col4:
                st.metric(
                    "Recall",
                    f"{metrics['Recall']:.4f}",
                    delta="TP / (TP + FN)"
                )
            
            st.markdown("---")
            
            # Additional Metrics in colored cards
            st.markdown("#### Advanced Metrics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>AUC-ROC Score</h3>
                    <div class='value'>{metrics['AUC']:.4f}</div>
                    <div class='label'>Area Under ROC Curve - Model's ability to distinguish between classes</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>Matthews Correlation</h3>
                    <div class='value'>{metrics['MCC']:.4f}</div>
                    <div class='label'>Balanced quality metric even with class imbalance</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Confusion Matrix and ROC Curve
            st.markdown("#### Visual Analysis")
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.pyplot(create_beautiful_confusion_matrix(y_true, y_pred))
            
            with col_right:
                st.pyplot(create_roc_curve(y_true, y_prob))
        
        else:
            st.warning("No 'target' column found. Showing predictions only.")
        
        # Predictions Table
        st.markdown("---")
        st.markdown("#### Detailed Predictions")
        results = X.copy()
        results['Predicted_Class'] = ['Malignant' if p == 0 else 'Benign' for p in y_pred]
        results['Confidence'] = [f"{p:.1%}" for p in y_prob]
        
        if has_labels:
            results['Actual'] = ['Malignant' if p == 0 else 'Benign' for p in y_true]
            results['Correct'] = ['Yes' if pred == actual else 'No' 
                                 for pred, actual in zip(y_pred, y_true)]
        
        st.dataframe(results.head(20), use_container_width=True, height=400)
    
    # --- TAB 2: MODEL COMPARISON ---
    with tab2:
        st.markdown("### Model Performance Comparison")
        st.markdown("---")
        
        if not has_labels:
            st.error("Comparison requires labeled data with 'target' column")
        else:
            if st.button("Launch Complete Evaluation"):
                comparison_results = []
                
                # Progress tracking
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                for idx, (name, model) in enumerate(models.items()):
                    progress_text.markdown(f"**Evaluating {name.replace('_', ' ')}...**")
                    
                    # Preprocess
                    if name in ["Logistic_Regression", "KNN"]:
                        X_curr = scaler.transform(X)
                    else:
                        X_curr = X
                    
                    # Predict
                    curr_pred = model.predict(X_curr)
                    curr_prob = model.predict_proba(X_curr)[:, 1]
                    
                    # Metrics
                    m = calculate_metrics(y_true, curr_pred, curr_prob)
                    m["Model"] = name.replace('_', ' ')
                    comparison_results.append(m)
                    
                    progress_bar.progress((idx + 1) / len(models))
                
                progress_text.markdown("**Evaluation Complete**")
                
                # Results DataFrame
                comp_df = pd.DataFrame(comparison_results)
                comp_df = comp_df.set_index("Model")
                
                # Winner announcement
                best_model = comp_df['Accuracy'].idxmax()
                best_accuracy = comp_df['Accuracy'].max()
                
                st.success(f"**Best Performing Model:** {best_model} with {best_accuracy:.2%} accuracy")
                
                st.markdown("---")
                
                # Interactive Table
                st.markdown("#### Complete Performance Table")
                
                # Style the dataframe
                styled_df = comp_df.style.highlight_max(
                    axis=0, 
                    color='lightgreen',
                    subset=['Accuracy', 'F1 Score', 'Precision', 'Recall', 'AUC', 'MCC']
                ).highlight_min(
                    axis=0, 
                    color='#FFE5E5',
                    subset=['Accuracy', 'F1 Score', 'Precision', 'Recall', 'AUC', 'MCC']
                ).format("{:.4f}")
                
                st.dataframe(styled_df, use_container_width=True, height=300)
                
                st.markdown("---")
                
                # Comparison Charts
                st.markdown("#### Visual Comparison")
                st.pyplot(create_comparison_chart(comp_df, 'Accuracy', 'F1 Score'))
                
                st.markdown("---")
                
                # Radar Chart
                st.markdown("#### Multi-Metric Radar Comparison")
                st.pyplot(create_radar_chart(comp_df))
    
    # --- TAB 3: ADVANCED METRICS ---
    with tab3:
        st.markdown("### Advanced Diagnostic Curves")
        st.markdown("---")
        
        if not has_labels:
            st.error("Advanced metrics require labeled data")
        else:
            selected_adv_model = st.selectbox(
                "Select Model for Advanced Analysis",
                list(models.keys()),
                format_func=lambda x: x.replace('_', ' '),
                key="advanced_model_select"
            )
            
            model_adv = models[selected_adv_model]
            
            # Preprocess
            if selected_adv_model in ["Logistic_Regression", "KNN"]:
                X_adv = scaler.transform(X)
            else:
                X_adv = X
            
            # Predictions
            y_pred_adv = model_adv.predict(X_adv)
            y_prob_adv = model_adv.predict_proba(X_adv)[:, 1]
            
            # ROC and PR Curves
            col1, col2 = st.columns(2)
            
            with col1:
                st.pyplot(create_roc_curve(y_true, y_prob_adv))
            
            with col2:
                st.pyplot(create_precision_recall_curve(y_true, y_prob_adv))
            
            # Probability Distribution
            st.markdown("---")
            st.markdown("#### Prediction Probability Distribution")
            st.pyplot(create_probability_distribution(y_true, y_prob_adv))
            
            # Threshold Analysis
            st.markdown("---")
            st.markdown("#### Decision Threshold Analysis")
            
            threshold = st.slider(
                "Adjust Classification Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Change the threshold to see how it affects predictions"
            )
            
            # Recalculate predictions with new threshold
            y_pred_thresh = (y_prob_adv >= threshold).astype(int)
            metrics_thresh = calculate_metrics(y_true, y_pred_thresh, y_prob_adv)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics_thresh['Accuracy']:.2%}")
            col2.metric("Precision", f"{metrics_thresh['Precision']:.4f}")
            col3.metric("Recall", f"{metrics_thresh['Recall']:.4f}")
            col4.metric("F1 Score", f"{metrics_thresh['F1 Score']:.4f}")

else:
    # Welcome Screen
    st.markdown("""
        <div style='text-align: center; padding: 50px; background: white; border-radius: 20px; margin: 30px; box-shadow: 0 10px 40px rgba(0,0,0,0.1);'>
            <h2 style='color: #667eea;'>Welcome to the AI Diagnostics Platform</h2>
            <p style='font-size: 1.2rem; color: #4a5568; margin: 20px 0;'>
                Upload your breast cancer dataset to begin comprehensive model evaluation
            </p>
            <div style='text-align: left; max-width: 600px; margin: 30px auto;'>
                <h3 style='color: #2d3748;'>Getting Started:</h3>
                <ol style='font-size: 1.1rem; line-height: 2; color: #4a5568;'>
                    <li><strong>Upload Data:</strong> Use the sidebar to upload your CSV file</li>
                    <li><strong>Analyze Models:</strong> Explore individual model performance with detailed metrics</li>
                    <li><strong>Compare:</strong> Run all models and see which performs best</li>
                    <li><strong>Advanced Metrics:</strong> Dive deep into ROC curves, PR curves, and threshold analysis</li>
                </ol>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature showcase
    st.markdown("---")
    st.markdown("### Platform Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='custom-card'>
            <h3 style='color: #667eea;'>Model Analysis</h3>
            <p style='color: #4a5568;'>Deep dive into individual model performance with comprehensive metrics, confusion matrices, and ROC curves</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='custom-card'>
            <h3 style='color: #667eea;'>Model Comparison</h3>
            <p style='color: #4a5568;'>Compare multiple models side-by-side with beautiful visualizations and radar charts</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='custom-card'>
            <h3 style='color: #667eea;'>Advanced Metrics</h3>
            <p style='color: #4a5568;'>ROC curves, precision-recall analysis, probability distributions, and threshold tuning</p>
        </div>
        """, unsafe_allow_html=True)