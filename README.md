# Model Evaluator

Гибкий инструмент для сравнения моделей машинного обучения с поддержкой ансамблей, оптимизации гиперпараметров и валидации.

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.6.1-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Optuna](https://img.shields.io/badge/Optuna-4.2-2C5D92)](https://optuna.org)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2.7-FF6B4A)](https://catboost.ai)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.5.0-FFD700)](https://lightgbm.readthedocs.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.3-9B4F96)](https://xgboost.readthedocs.io)

#### Автор: [Ерофеев Олег](https://github.com/SomeBotMeOn)

---

## 🔥 Возможности

- **10+ моделей**: CatBoost, XGBoost, LightGBM, Логистическая регрессия, Случайный лес и другие
- **Ансамбли**: Блендинг, Бэггинг, Стекинг, Голосование (Hard/Soft)
- **Оптимизация**: Интеграция с Optuna для автоматического подбора параметров
- **Валидация**: K-Fold, Стратифицированный K-Fold, Leave-One-Out, Train-Test Split
- **Контроль качества**:
  - Предварительная проверка конфигурации
  - Соответствие PEP-8
  - Логирование ошибок
  - Автосохранение моделей

---

## 🆕 Новости:
* **Feb 13, 2025**: Запуск ModelEvaluator 0.0.1 для задач классификации с поддержкой ансамблей и Optuna.

---

## Установка

1. Создайте окружение Anaconda:
```bash
conda create -n model_eval python=3.10
conda activate model_eval
```

2. Скачайте файл `environment.yml`
3. Установите зависимости:
```bash
conda env update -f environment.yml
```

---

## Быстрый старт

```bash
from model_evaluator import ModelEvaluator

# Инициализация
evaluator = ModelEvaluator(
    data=df, # Ваш датасет
    target_column='Cluster', # Целевая переменная
)

link = 'models/' # Папка для сохранения моделей

# Пример оценки моделей
results = evaluator.evaluate_models(
    save_models_to=link
)
  
results # Вывод результатов
```

---

## Особенности

1. **Быстрая проверка введенных параметров и конфликтов в самом начале работы кода!**
2. Поддержка мультикласса и бинарной классификации
3. Автоматическое определение типа классификации
4. Поддержка метрик для много классовых задач
5. Возможность сохранения моделей в файл
6. Поддержка кастомных параметров для моделей
7. Возможность оценки ансамблей моделей
8. Оптимизация гиперпараметров с помощью Optuna
9. Возможность выбора метода валидации
10. Возможность использовать конкретные модели для оценки
11. Возможность исключить модели из оценки

---

## Подробнее о методам

### Метод `evaluate_models()`

| Параметр           | Тип   | Описание                                                                                     |
|--------------------|-------|----------------------------------------------------------------------------------------------|
| `selected_models`  | dict  | Словарь с моделями для оценки: `{'Модель': {параметры}}`                                     |
| `unselected_models`| dict  | Модели для исключения из оценки: `{'Модель': {}}`                                            |
| `custom_params`    | dict  | Пользовательские параметры для всех моделей: `{'Модель': {параметры}}`                       |
| `cv_method`        | str   | Метод валидации: `KFold`, `Stratified`, `LeaveOneOut`, `train_test_split` (default: `KFold`) |
| `cv_params`        | dict  | Параметры для выбранной валидации                                                            |
| `save_models_to`   | str   | Путь для сохранения обученных моделей                                                        |

#### Заметки: 
1. Передавать можно один из трех параметров: `selected_models`, `unselected_models`, `custom_params`.
2. Если `selected_models` или `custom_params` не переданы, то будут использованы все модели по умолчанию
(с использованием зафиксированного `seed`, где эт возможно)

#### Возвращает:
- Датасет с результатами оценки моделей
- Во время работы выводятся оформленные промежуточные результаты, а также информация о процессе обучения

#### Пример:
```bash
link = '../../models/'

custom_params = {
    'CatBoost': {'verbose': 0, 'random_state': 42},
    'XGBoost': {'verbosity': 0, 'random_state': 42},
    'LightGBM': {'verbosity': -1, 'random_state': 42},
}

results_df = evaluator.evaluate_models(
    custom_params=custom_params,
    cv_method='KFold',
    cv_params={'n_splits': 5},
    save_models_to=link
)

results_df
```

### Метод `evaluate_ensembles()`

| Ключ ансамбля | Параметры                                                                                                                                                                                                       |
|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Blending`    | `custom_name` (default: 'Blending'), `base_models`, `meta_model`, `test_size` (default: 0.3)                                                                                                                    |
| `Stacking`    | `custom_name` (default: 'Stacking'), `base_models`, `meta_model`, `cv_method` (default: 'KFold'), `cv_params` (default: {'n_splits': 5})                                                                        |
| `Bagging`     | `custom_name` (default: 'Bagging'), `base_model`, `n_estimators` (default: 10), `max_samples` (default: 1.0), `max_features` (default: 1.0), `bootstrap` (default: True), `bootstrap_features` (default: False) |
| `Voting`      | `custom_name` (default: 'Voting'), `base_models`, `voting_type` (default: 'hard')                                                                                                                               |


#### Описание параметров:
- `custom_name`: Пользовательское название ансамбля.
- `base_models`: Список базовых моделей для использования в ансамбле.
- `meta_model`: Модель для обучения на предсказаниях базовых моделей.
- `test_size`: Размер тестовой выборки для Blending.
- `cv_method`: Метод кросс-валидации для Stacking.
- `cv_params`: Параметры кросс-валидации для Stacking.
- `voting_type`: Тип голосования для Voting.
- `base_model`: Базовая модель для Bagging.
- `n_estimators`: Количество базовых моделей для Bagging.
- `max_samples`: Максимальное количество образцов для каждой базовой модели в Bagging.
- `max_features`: Максимальное количество признаков для каждой базовой модели в Bagging.
- `bootstrap`: Использовать ли бутстрап выборки для Bagging.
- `bootstrap_features`: Использовать ли бутстрап признаков для Bagging.

#### Возвращает:
- Датасет с результатами оценки ансамблей
- Во время работы выводятся оформленные промежуточные результаты, а также информация о процессе обучения

#### Пример:
```bash
ensemble_config = {
    'Blending': {
        'base_models': {
            'RandomForest': {'random_state': 42},
            'ExtraTrees': {'random_state': 42},
        },
        'meta_model': {
            'LogisticRegression': {'random_state': 42}
        },
        'n_splits': 5
    },
    'Stacking': {
        'cv_method': 'KFold',
        'cv_params': {
            'n_splits': 5
        },
        'base_models': {
            'RandomForest': {'random_state': 42},
            'LogisticRegression': {'random_state': 42},
            'ExtraTrees': {'random_state': 42},
        },
        'meta_model': {
            'LogisticRegression': {'random_state': 42}
        }
    },
    'Bagging': {
        'base_model': {
            'LogisticRegression': {'random_state': 42},
            'ExtraTrees': {'random_state': 42},
        }
    },
    'Voting': {
        'custom_name': 'Soft Voting',
        'voting_type': 'soft',
        'base_models': {
            'RandomForest': {'random_state': 42},
            'LogisticRegression': {'random_state': 42},
        }
    }
}

ensemble_results = evaluator.evaluate_ensembles(ensemble_config=ensemble_config)
ensemble_results
```

### Метод `tune_models_with_optuna()`

| Параметр            | Тип   | Описание                                                                  |
|---------------------|-------|---------------------------------------------------------------------------|
| `optuna_config`     | dict  | Конфигурация оптимизации: `{'Модель': {'параметр': 'распределение'}}`     |
| `n_trials`          | int   | Количество испытаний                                                      |
| `timeout`           | int   | Лимит времени оптимизации (секунды)                                       |
| `scoring`           | str   | Целевая метрика (`accuracy`, `f1`, `roc_auc` и др.) (default: `accuracy`) |
| `cv_method`         | str   | Метод кросс-валидации (default: `KFold`)                                  |
| `cv_params`         | dict  | Параметры кросс-валидации                                                 |
| `show_progress_bar` | bool  | Отображение прогресса оптимизации (default: `True`)                       |

##### Заметка: если `n_trials` и `timeout` не переданы, то оптимизация будет продолжаться до достижения лимита времени (`timeout=3600`)

#### Возвращает:
- Датасет с результатами оптимизации моделей
- Лучшие подобранные параметры для каждой модели
- Во время работы выводятся оформленные промежуточные результаты, а также информация о процессе обучения

#### Пример:
```bash
optuna_config = {
    'RandomForest': {
        'n_estimators': "trial.suggest_int('n_estimators', 100, 200)",
        'max_depth': "trial.suggest_int('max_depth', 3, 10)"
    },
    'LogisticRegression': {
        'C': "trial.suggest_float('C', 0.01, 10.0, log=True)"
    }
}

best_models, results_df = evaluator.tune_models_with_optuna(
    optuna_config=optuna_config, 
    timeout=3600 * 2,
    show_progress_bar=False
)
```