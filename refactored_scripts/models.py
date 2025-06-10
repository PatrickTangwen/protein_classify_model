import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

class ImprovedProteinClassifier(nn.Module):
    """
    The original deep neural network for protein classification.
    """
    def __init__(self, input_dim, num_classes, hidden_dims=[512, 256, 128]):
        super(ImprovedProteinClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.4)
            ])
            prev_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        features = self.feature_layers(x)
        output = self.classifier(features)
        return output

# Dictionary of models to be benchmarked
MODELS = {
    'random_forest': {
        'model': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'type': 'sklearn'
    },
    'svm': {
        'model': SVC(kernel='rbf', probability=True, random_state=42),
        'type': 'sklearn'
    },
    'neural_network': {
        'model': ImprovedProteinClassifier, # The class, not an instance
        'type': 'pytorch'
    },
    'logistic_regression': {
        'model': LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
        'type': 'sklearn'
    },
    'extra_trees': {
        'model': ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'type': 'sklearn'
    },
    'knn': {
        'model': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'type': 'sklearn'
    },
    'naive_bayes': {
        'model': GaussianNB(),
        'type': 'sklearn'
    },
    # 'decision_tree': {
    #     'model': DecisionTreeClassifier(random_state=42, max_depth=20),
    #     'type': 'sklearn'
    # },
    # 'adaboost': {
    #     'model': AdaBoostClassifier(random_state=42, n_estimators=100),
    #     'type': 'sklearn'
    # }
}

# Add XGBoost if available
if XGBOOST_AVAILABLE:
    MODELS['xgboost'] = {
        'model': xgb.XGBClassifier(
            random_state=42,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            n_jobs=-1,
            eval_metric='logloss'
        ),
        'type': 'sklearn'
    }

# Add LightGBM if available
# if LIGHTGBM_AVAILABLE:
#     MODELS['lightgbm'] = {
#         'model': lgb.LGBMClassifier(
#             random_state=42,
#             n_estimators=100,
#             max_depth=6,
#             learning_rate=0.1,
#             n_jobs=-1,
#             verbose=-1
#         ),
#         'type': 'sklearn'
#     } 