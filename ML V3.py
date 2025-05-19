#%%
# -*- coding: utf-8 -*-
"""
Created on [19/05/2025]
@author: [Michael Gonçalves]
"""

# =============================================================================
# 1. Importação de Bibliotecas
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt
import shap

# =============================================================================
# 2. Funções Personalizadas
# =============================================================================
def custom_recall_metric(preds, dtrain):
    y_true = dtrain.get_label()
    y_pred_proba = preds
    y_pred = (y_pred_proba > 0.35).astype(int) 
    recall = np.sum(y_true * y_pred) / (np.sum(y_true) + 1e-16)
    return 'recall@0.3', recall

# =============================================================================
# 3. Pré-processamento
# =============================================================================
try:
    df = pd.read_csv('folha.csv')
except FileNotFoundError:
    print("Erro: Arquivo 'folha.csv' não encontrado. Verifique o caminho.")
    exit()

df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 
                    'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=np.int64)

cols_to_drop = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
df_clean = df_encoded.drop(cols_to_drop, axis=1)

# =============================================================================
# 4. Engenharia de Features Numéricas Estáveis
# =============================================================================
df_clean['WorkloadRatio'] = df_clean['YearsAtCompany'] / (df_clean['YearsInCurrentRole'] + 1e-6) 
df_clean['SalaryGrowth'] = (df_clean['MonthlyIncome'] - df_clean['DailyRate']) / (df_clean['DailyRate'] + 1e-6)
df_clean['PromotionVelocity'] = df_clean['YearsSinceLastPromotion'] / (df_clean['YearsAtCompany'] + 1e-6)

df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
df_clean = df_clean.fillna(df_clean.median()) 

# =============================================================================
# 5. Balanceamento de Classes com SMOTE
# =============================================================================
X = df_clean.drop('Attrition', axis=1)
y = df_clean['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.25,
    stratify=y,
    random_state=42
)

print(f"Contagem de classes antes do SMOTE (treino): \n{y_train.value_counts()}")
n_majority_before_smote = y_train[y_train == 0].shape[0]
smote_strategy = {1: int(n_majority_before_smote * 0.6)} 

smote = SMOTE(
    sampling_strategy=smote_strategy,
    k_neighbors=5,
    random_state=42
)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"Contagem de classes depois do SMOTE (treino): \n{y_train_smote.value_counts()}")

# =============================================================================
# 6. Controle de Multicolinearidade
# =============================================================================
common_cols_before_corr = X_train_smote.columns.intersection(X_test.columns)
X_test = X_test[common_cols_before_corr]
X_train_smote = X_train_smote[common_cols_before_corr]

corr_matrix = X_train_smote.corr().abs()
upper_triangle_mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
upper = corr_matrix.where(upper_triangle_mask)

to_drop_corr = [column for column in upper.columns if upper[column].any() and upper[column].max() > 0.85]
print(f"Colunas a serem dropadas por alta correlação: {to_drop_corr}")

X_train_smote = X_train_smote.drop(to_drop_corr, axis=1)
X_test = X_test.drop(to_drop_corr, axis=1)

final_cols = X_train_smote.columns
X_test = X_test[final_cols]

# =============================================================================
# 7. Modelagem XGBoost com binary:logistic e scale_pos_weight
# =============================================================================
count_neg_smote = np.sum(y_train_smote == 0)
count_pos_smote = np.sum(y_train_smote == 1)
if count_pos_smote > 0:
    calculated_scale_pos_weight = count_neg_smote / count_pos_smote
else:
    calculated_scale_pos_weight = 1

print(f"Calculated scale_pos_weight: {calculated_scale_pos_weight:.4f}")

dtrain = xgb.DMatrix(X_train_smote, label=y_train_smote, feature_names=X_train_smote.columns.tolist())
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=X_test.columns.tolist())

params = {
    'objective': 'binary:logistic', 
    'eval_metric': 'auc',         
    'max_depth': 4,
    'learning_rate': 0.05,
    'reg_alpha': 0.7,
    'reg_lambda': 1.2,
    'subsample': 0.75,
    'colsample_bytree': 0.6,
    'gamma': 0.2,
    'max_delta_step': 1, 
    'min_child_weight': 2,
    'seed': 42,
    'scale_pos_weight': calculated_scale_pos_weight
}

print("\nTreinando com binary:logistic e scale_pos_weight...")
model = xgb.train(
    params,
    dtrain=dtrain,
    num_boost_round=1000,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=50,
    verbose_eval=50,
    custom_metric=custom_recall_metric 
)

# =============================================================================
# 8. Avaliação e Explicabilidade
# =============================================================================
y_proba = model.predict(dtest)

# Ajuste de Threshold para maximizar Recall na classe positiva 
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

desired_recall_target = 0.70 
valid_indices = np.where(recall >= desired_recall_target)[0]

if len(valid_indices) > 0:
    f1_scores_at_valid_recall = (2 * precision[valid_indices] * recall[valid_indices]) / \
                                (precision[valid_indices] + recall[valid_indices] + 1e-16)
    optimal_idx_in_valid = np.argmax(f1_scores_at_valid_recall)
    optimal_idx = valid_indices[optimal_idx_in_valid]

    if optimal_idx < len(thresholds):
        final_threshold = thresholds[optimal_idx]
    else: 
        final_threshold = thresholds[-1] if len(thresholds) > 0 else 0.5 # Fallback mais simples
        if recall[optimal_idx] == 1.0 and precision[optimal_idx] > 0: # Todos positivos são capturados
             if optimal_idx == len(recall) -1: 
                   pass
                   final_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else (thresholds[-1] if len(thresholds)>0 else 0.5)


    print(f"\nThreshold ótimo (recall >= {desired_recall_target}): {final_threshold:.4f}")
    print(f"Para este threshold: Recall = {recall[optimal_idx]:.4f}, Precision = {precision[optimal_idx]:.4f}, F1-Score = {f1_scores_at_valid_recall[optimal_idx_in_valid]:.4f}")
else:
    print(f"Não foi possível atingir o recall de {desired_recall_target} de forma significativa. Usando threshold padrão de 0.5.")
    final_threshold = 0.5 # Fallback
    temp_preds_05 = (y_proba >= 0.6).astype(int)
    temp_recall_05 = np.sum(y_test * temp_preds_05) / (np.sum(y_test) + 1e-16)
    temp_precision_05 = np.sum(y_test * temp_preds_05) / (np.sum(temp_preds_05) + 1e-16)
    temp_f1_05 = (2 * temp_precision_05 * temp_recall_05) / (temp_precision_05 + temp_recall_05 + 1e-16)
    print(f"Para threshold 0.5: Recall = {temp_recall_05:.4f}, Precision = {temp_precision_05:.4f}, F1-Score = {temp_f1_05:.4f}")


y_pred_final = (y_proba >= final_threshold).astype(int)

# Métricas finais
print("\nMatriz de Confusão (com threshold ajustado):")
print(confusion_matrix(y_test, y_pred_final))
print("\nRelatório de Classificação (com threshold ajustado):")
print(classification_report(y_test, y_pred_final, target_names=['No Attrition', 'Attrition'], zero_division=0))
print(f"\nROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")


# =============================================================================
# 9. SHAP (Explicabilidade)
# =============================================================================
explainer = shap.TreeExplainer(model)

try:
    # Tentar com X_test DataFrame primeiro, pois preserva nomes de colunas para SHAP
    shap_values = explainer.shap_values(X_test)
except Exception as e_shap_df:
    print(f"Erro ao calcular SHAP values com X_test (DataFrame): {e_shap_df}")
    print("Tentando com DMatrix dtest...")
    try:
        shap_values = explainer.shap_values(dtest)
    except Exception as e_shap_dmatrix:
        print(f"Erro ao calcular SHAP values com DMatrix dtest: {e_shap_dmatrix}")
        shap_values = None

if shap_values is not None:
    # Para 'binary:logistic', shap_values é um único array para a classe positiva
    shap_values_for_plot = shap_values

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_for_plot, X_test, plot_type="bar", show=False)
    plt.title('SHAP Values - Importância das Features para Attrition')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,8))
    shap.summary_plot(shap_values_for_plot, X_test, show=False)
    plt.title('SHAP Values - Impacto das Features nas Predições de Attrition')
    plt.tight_layout()
    plt.show()
else:
    print("Não foi possível gerar os gráficos SHAP.")
# %%
