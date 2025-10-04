"""Autogradingscript - Tests para SelectKBest en regresión."""

import pytest


def load_data():
    """
    Carga y preprocesa el dataset Auto MPG.
    
    Returns:
        tuple: (X, y) donde X son las características e y es el target MPG.
    """
    import pandas as pd

    dataset = pd.read_csv("files/input/auto_mpg.csv")
    dataset = dataset.dropna()  # Elimina las 6 filas con Horsepower NaN
    dataset["Origin"] = dataset["Origin"].map(
        {1: "USA", 2: "Europe", 3: "Japan"},
    )
    y = dataset.pop("MPG")
    x = dataset.copy()

    return x, y


def load_estimator():
    """
    Carga el estimador entrenado desde el archivo pickle.
    
    Returns:
        sklearn estimator: El estimador entrenado o None si no existe.
    """
    import os
    import pickle

    if not os.path.exists("homework/estimator.pickle"):
        return None
    with open("homework/estimator.pickle", "rb") as file:
        estimator = pickle.load(file)

    return estimator


# ============================================================================
# TESTS DE CALIDAD DE DATOS
# ============================================================================

def test_data_loading():
    """Test: Verificar que la carga de datos funciona correctamente."""
    x, y = load_data()
    
    # Verificar estructura básica
    assert x.shape[0] == 392  # 398 originales - 6 NaN = 392
    assert x.shape[1] == 7   # 7 características después de quitar MPG
    assert len(y) == 392     # Target debería tener la misma longitud
    
    # Verificar que Origin se mapeó correctamente
    assert set(x["Origin"].unique()) <= {"USA", "Europe", "Japan"}
    
    # Verificar que no hay valores faltantes después del preprocesamiento
    assert x.isnull().sum().sum() == 0
    assert y.isnull().sum() == 0


def test_data_preprocessing():
    """Test: Verificar el preprocesamiento de datos."""
    x, y = load_data()
    
    # Verificar rangos de datos
    assert y.min() > 0      # MPG debería ser positivo
    assert y.max() < 50    # Valores realistas para MPG
    
    # Verificar características numéricas
    numeric_cols = ['Cylinders', 'Displacement', 'Weight', 'Acceleration', 'Model Year']
    for col in numeric_cols:
        assert x[col].dtype in ['int64', 'float64']
        assert x[col].min() > 0


# ============================================================================
# TESTS DE MODELO
# ============================================================================

def test_estimator_availability():
    """Test: Verificar que el modelo entrenado existe."""
    estimator = load_estimator()
    assert estimator is not None, "No se encontró el modelo entrenado"
    
    # Verificar que es un estimador sklearn
    assert hasattr(estimator, 'predict')
    assert hasattr(estimator, 'best_estimator_')  # Debería ser GridSearchCV


def test_model_performance_minimum():
    """
    Test: Rendimiento mínimo requerido.
    
    El modelo debe tener al menos un R² de 0.6 (60% de varianza explicada).
    """
    from sklearn.metrics import r2_score

    x, y = load_data()
    estimator = load_estimator()
    
    y_pred = estimator.predict(x)
    r2 = r2_score(y, y_pred)

    assert r2 > 0.6, f"Actor de correlación R² {r2:.3f} es menor al mínimo requerido 0.6"


def test_model_predictions_consistency():
    """Test: Verificar consistencia de las predicciones."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np

    x, y = load_data()
    estimator = load_estimator()
    
    y_pred = estimator.predict(x)
    
    # Verificar que las predicciones son valores numéricos válidos
    assert np.all(np.isfinite(y_pred))  # No NaN ni infinitos
    assert np.all(y_pred >= 0)         # MPG no puede ser negativo
    
    # Verificar que MAE y MSE son razonables
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    assert mae < 10.0, f"⚠️ MAE {mae:.2f} es muy alto (revisar modelo)"
    assert mse < 200.0, f"⚠️ MSE {mse:.2f} es muy alto (revisar modelo)"


# ============================================================================
# TESTS DE CARACTERÍSTICAS SELECCIONADAS (SelectKBest)
# ============================================================================

def test_feature_selection():
    """
    Test: Verificar que SelectKBest está configurado correctamente.
    """
    estimator = load_estimator()
    
    # Verificar que el pipeline contiene SelectKBest
    pipeline = estimator.best_estimator_
    assert 'selectkbest' in pipeline.named_steps
    
    selectkbest = pipeline.named_steps['selectkbest']
    
    # Verificar que k es razonable (entre 1 y numero de features)
    x, y = load_data()
    max_features = x.shape[1]
    k = selectkbest.k
    
    assert 1 <= k <= max_features, f"K={k} debe estar entre 1 y {max_features}"
    
    # Verificar que SelecKBest usando f_regression
    from sklearn.feature_selection import f_regression
    assert selectkbest.get_params()['score_func'] == f_regression


def test_preprocessing_pipeline():
    """
    Test: Verificar que el pipeline de preprocesamiento es correcto.
    """
    estimator = load_estimator()
    pipeline = estimator.best_estimator_
    
    # Verificar ColumnTransformer para OneHotEncoder en Origin
    transformer = pipeline.named_steps['tranformer']  # Nota: hay typo en 'tranformer'
    assert hasattr(transformer, 'transformers')
    
    # Verificar que StandardScaler está en remainder
    assert transformer.remainder.__class__.__name__ == 'StandardScaler'


# ============================================================================
# TESTS DE VALIDACIÓN CRUZADA Y OPTIMIZACIÓN
# ============================================================================

def test_grid_search_optimization():
    """Test: Verificar que GridSearchCV se ejecutó correctamente."""
    estimator = load_estimator()
    
    # Verificar que es GridSearchCV
    assert hasattr(estimator, 'best_estimator_')
    assert hasattr(estimator, 'best_params_')
    assert hasattr(estimator, 'cv_results_')
    
    # Verificar que se probó al menos diferentes valores de k
    best_params = estimator.best_params_
    assert 'selectkbest__k' in best_params
    
    # Verificar que se hizo validación cruzada
    assert estimator.cv == 5


def test_reproducibility():
    """
    Test: Verificar que los resultados son reproducibles.
    """
    from sklearn.metrics import r2_score
    
    x, y = load_data()
    estimator = load_estimator()
    
    # Hacer predicciones múltiples veces
    y_pred_1 = estimator.predict(x)
    y_pred_2 = estimator.predict(x)
    
    # Los resultados deben ser idénticos
    import numpy as np
    assert np.array_equal(y_pred_1, y_pred_2), "Predicciones inconsistentes"
    
    # Los scores también deben ser iguales
    r2_1 = r2_score(y, y_pred_1)
    r2_2 = r2_score(y, y_pred_2)
    assert r2_1 == r2_2, "Scores inconsistentes"


# ============================================================================
# TEST PRINCIPAL DE EVALUACIÓN (ORIGINAL)
# ============================================================================

def test_01():
    """
    Test principal: Evaluación del rendimiento del modelo.
    
    Este test verifica que el modelo implementado cumple con el rendimiento mínimo
    requerido para SelectKBest en regresión.
    """
    from sklearn.metrics import r2_score

    x, y = load_data()
    estimator = load_estimator()
    
    y_pred = estimator.predict(x)
    r2 = r2_score(y, y_pred)

    print(f"\n📊 RESULTADOS DEL MODELO:")
    print(f"🏷️  Dataset: Auto MPG ({x.shape[0]} muestras, {x.shape[1]} características)")
    print(f"🎯 R² Score: {r2:.4f} ({r2*100:.1f}% de varianza explicada)")
    print(f"✅ Test {'APROBADO' if r2 > 0.6 else 'REPROBADO'} - Umbral mínimo: 0.6")

    assert r2 > 0.6