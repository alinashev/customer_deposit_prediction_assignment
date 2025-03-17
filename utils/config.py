"""
config.py

This module contains configuration settings for data preprocessing,
feature engineering, model training, and hyperparameter tuning.
"""

### --- Preprocessing Configuration --- ###
PREPROCESSING_CONFIG = {
    "handle_unknowns": True,  # Handle 'unknown' values in categorical features
    "apply_outliers": True,  # Apply outlier processing
    "apply_feature_engineering": True,  # Generate new features
    "apply_preprocessing": True  # Perform main preprocessing (encoding, scaling, etc.)
}

# Ordinal encoding mappings for categorical features with a specific order
ORDINAL_MAPPINGS = {
    "education": ['illiterate', 'basic.4y', 'basic.6y', 'basic.9y',
                  'high.school', 'professional.course', 'university.degree']
}

# Categorical features that require one-hot encoding
ONE_HOT_COLS = ["job", "marital", "housing", "loan", "default", "poutcome"]

# Direct categorical mappings for encoding
CATEGORY_MAPPINGS = {
    "month": {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    },
    "day_of_week": {
        'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5
    }
}

OUTLIER_CONFIG = {
    "clip_columns": {
        "age": {"upper_quantile": 0.99},  # Clip outliers for 'age'
        "campaign": {"upper_quantile": 0.95},  # Clip outliers for 'campaign'
        "previous": {"upper_quantile": 0.99},  # Clip outliers for 'previous'
        "cons.conf.idx": {"lower_quantile": 0.01, "upper_quantile": 0.99}  # Clip outliers for 'cons.conf.idx'
    },
    "drop_columns": ["duration"],  # Drop 'duration' column
    "log_transform": ["campaign"],  # Apply log transformation to 'campaign'
    "binary_features": {
        "pdays": {"threshold": 999, "new_col": "was_contacted_before"}  # Create binary feature from 'pdays'
    }
}

MISSING_VALUE_CONFIG = {
    "education": {"strategy": "mode"},
    "job": {"strategy": "skip"},
    "marital": {"strategy": "skip"},
    "default": {"strategy": "skip", "fill_value": "no"},
    "housing": {"strategy": "skip"},
    "loan": {"strategy": "skip"},
    "cons.conf.idx": {"strategy": "skip"},
    "emp.var.rate": {"strategy": "skip"}
}

### --- Feature Engineering Configuration --- ###
FEATURE_ENGINEERING_CONFIG = {
    # Feature interactions between categorical/numeric variables
    "INTERACTION_FEATURES": [],
    # Ratio-based features (division of numerical features)
    "RATIO_FEATURES": [
        ("campaign", "previous"),  # campaign_previous_ratio
    ],
    # Economic features based on macroeconomic indicators
    "ECONOMIC_FEATURES": [
        # ("cons.price.idx", "cons.conf.idx")  # price_conf_interaction
    ],
    # Time-based features
    "TIME_FEATURES": {
        "second_half_year": {
            "column": "month",
            "values": ["jul", "aug", "sep", "oct", "nov", "dec"]
        },
        "weekday_call": {
            "column": "day_of_week",
            "values": ["mon", "tue", "wed", "thu", "fri"]
        }
    },
    # Contact-related features
    "CONTACT_FEATURES": [
        "total_contacts",  # Sum of previous + campaign contacts
        "was_previously_contacted",  # Binary indicator: previous > 0
        "campaign_to_total_ratio"  # campaign / total_contacts
    ]
}

METRIC_CONFIG = {
    "roc_auc": "primary",
    "f1_score": "secondary",
    "precision": "disabled",
    "recall": "disabled",
    "pr_auc": "disabled",
    "g_mean": "disabled",
    "balanced_accuracy": "disabled"
}
