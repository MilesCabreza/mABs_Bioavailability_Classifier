import sklearn
import xgboost
import esm.pretrained
from transformers import AutoTokenizer, AutoModel
from antiberty import AntiBERTyRunner
from ablang import pretrained
import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.model_selection import GridSearchCV, cross_validate, LeaveOneOut
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# setting random seed to reproducing same results
torch.manual_seed(42)
np.random.seed(42)

# Creating Variables from dataset

csv = 'mAb_inputdata.csv'
df = pd.read_csv(csv)
heavy_seqs = df['Heavy_Var']
light_seqs = df['Light_Var']
labels = df['High/Low']

"""**3. Load Model Class**"""

class Classifier:
    def __init__(self):
        self.param_grid = None
        self.pipeline = None
        self.grid_search = None
        self.best_clf = None
        self.model_files = None
        self.voting_clf = None
        

    def create_inputs_antiBERTy(self, heavy_seqs, light_seqs, labels):
        # function creates the needed inputs using antiBERTy, a pytorch based model
        antiberty = AntiBERTyRunner()
            
        # creates tensors for both light and heavy sequences
        heavy_embeddings = antiberty.embed(heavy_seqs)
        light_embeddings = antiberty.embed(light_seqs)

        def tensors_to_numpy_average(tensors):
            # Convert each tensor to a numpy array
            np_arrays = [tensor.numpy() for tensor in tensors]
            # Sum over the 512 dimension and divide by 512 to get average of seqs lengths -> list of 2D arrays to list of 1D arrays
            avg_arrays = [np.sum(np_array, axis=0) / 512 for np_array in np_arrays]
            # Stack arrays to form the final array -> list of 1D arrays to 2D array
            final_array = np.vstack(avg_arrays)
            return final_array

        # creates arrays for both light and heavy sequences
        X1 = tensors_to_numpy_average(heavy_embeddings)
        X2 = tensors_to_numpy_average(light_embeddings)
        # Combines the 2 light and heavy arrays to form ***(dataset size, 1024)*** shape
        X = np.hstack((X1, X2))

        y = np.array(labels)
        return X, y


    def create_inputs_ABlang(self, heavy_seqs, light_seqs, labels):
        # loads in ABlangs light and heavy specified models
        heavy_ablang = pretrained('heavy')
        light_ablang = pretrained('light')

        # function that modifies the sequences, replacing any string value ABlang can't use with '*'
        def replace_except_specific_letters(lst, allowed_letters):
            return [
                ''.join(char if char in allowed_letters else '*' for char in item)
                for item in lst
            ]

        # modification of light and heavy sequences data
        allowed_letters = set('MRHKDESTNQCGPAVIFYWL')
        heavy_seqs_modified = replace_except_specific_letters(heavy_seqs, allowed_letters)
        light_seqs_modified = replace_except_specific_letters(light_seqs, allowed_letters)

        X1 = heavy_ablang(heavy_seqs_modified, mode='rescoding')
        X2 = light_ablang(light_seqs_modified, mode='rescoding')

        # the following performs the same array manipulation as in the antiBERTy method
        def avg_array(np_arrays):
            avg_arrays = [np.sum(np_array, axis=0) / 768 for np_array in np_arrays]
            final_array = np.vstack(avg_arrays)
            return final_array

        X1 = avg_array(X1)
        X2 = avg_array(X2)
        # Combines the 2 light and heavy arrays to form ***(dataset size, 1536)*** shape
        X = np.hstack((X1, X2))    

        y = np.array(labels)
        return X, y


    def create_inputs_ESM(self, heavy_seqs, light_seqs, labels):
        
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()

        def seqs_to_numpy_array(seqs):
            batch_converter_input = []
            for seq in seqs:
                batch_converter_input.append(("antibody", seq))

            model.eval()
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_converter_input)

            # calculates the length of each sequence by counting non-padding tokens.
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

            # Extract per-residue representations (on CPU), creates tensors with dimension
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[6], return_contacts=True)
            token_representations = results["representations"][6]

            # Sequence representation, this turns it from a list of dimension (Variable length, 1280) to
            # a list of tensors with dimension (1280, )
            sequence_representations = []
            for i, tokens_len in enumerate(batch_lens):
                sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))

            # This converts list of tensors into one numpy array
            np_arrays = [tensor.numpy() for tensor in sequence_representations]
            final_array = np.vstack(np_arrays)
            return final_array

        X1 = seqs_to_numpy_array(heavy_seqs)
        X2 = seqs_to_numpy_array(light_seqs)
        # Combines the 2 light and heavy arrays
        X = np.hstack((X1, X2))
        
        y = np.array(labels)
        return X, y

    def tune_hyperparameters_loo(self, X, y):
    # tunes the parameters of selected model using leave one out validation gridsearch
        loo = LeaveOneOut()
        self.grid_search = GridSearchCV(self.pipeline, self.param_grid, cv=loo)
        self.grid_search.fit(X, y)
        self.best_clf = self.grid_search.best_estimator_
        return self.grid_search.best_params_, self.grid_search.best_score_


    def loo_validate(self, X, y):
        if not self.best_clf:
            raise ValueError('Model not tuned. Call tune_hyperparameters() first.')
        loo = LeaveOneOut()

        scores = cross_validate(self.best_clf, X, y, cv=loo, return_train_score=True)
        
        # Extract training and test scores
        train_scores = scores['train_score']
        test_scores = scores['test_score']
        
        # Convert np.float64 values to regular Python floats and round to 4 decimal places
        rounded_train_scores = [round(float(score), 4) for score in train_scores]
        rounded_test_scores = [round(float(score), 4) for score in test_scores]


        # Create a DataFrame with group scores as lists and mean scores
        loo_df = pd.DataFrame({
            'Train Scores (All mAbs)': [rounded_train_scores],  # Store scores as lists
            'Test Scores (All mAbs)': [rounded_test_scores],
            'Mean Training Score': [f"{train_scores.mean():.6}"],
            'Mean Validation Score': [f"{test_scores.mean():.6}"],
            'Best Hyperparameters': [str(self.grid_search.best_params_)]
        })

        return loo_df      

        
    def alldata_test(self, X, y):
        
        def save_loo_models(X, y): 
            
            from os import makedirs
            makedirs("loo_models", exist_ok=True)
                       
            loo = LeaveOneOut()      
            model_files = []  # To store file paths of saved models
            
            from sklearn.base import clone
            
            # Loop through each mAb/fold in loo
            for i, (train_idx, test_idx) in enumerate(loo.split(X, y)):
                # Clone the best classifier (from hyperparameter tuning) to avoid modifying the original
                model = clone(self.best_clf)
                
                # Train the model on the training data of the current fold
                model.fit(X[train_idx], y[train_idx])
                
                # Save the trained model
                model_filename = f'loo_models/model_group_{i}.pkl'
                joblib.dump(model, model_filename)
                model_files.append(model_filename)
                
                self.model_files = model_files  # Store model paths for later use
        
        save_loo_models(X, y)
        
        # Function to load models and fit BaggingClassifier
        def fit_voting_ensemble(X, y):
            # Load the saved models and use them as base estimators for BaggingClassifier
            estimators = []
            
            for i, model_filename in enumerate(self.model_files):
                model = joblib.load(model_filename)
                estimators.append((f'model{i}', model))

            
            # Create a VotingClassifier using these models
            voting_clf = VotingClassifier(estimators=estimators, voting='hard')  # Use 'hard' for majority voting
            voting_clf.fit(X, y)
            
            
            self.voting_clf = voting_clf  # Save the voting ensemble for future predictions
    
        fit_voting_ensemble(X, y)
    
    
    def run_clf(self, heavy_seqs, light_seqs, labels, feature_model = None, clf_model = None, save_model = None):
    # method that can be run to excecute entire model process and allows user to choose which model
    # to generate features from, which classifier model to use, and whether to output "all data" or
    # loo and training data scores

        torch.manual_seed(42)
        np.random.seed(42)

        # if model specifications are not stated when running method, the following prompts will appear to specify model
        if not feature_model:
            feature_model = input('Which model to generate features from: \nantiBERTy\nABlang\nESM\nDeepSP\n')
        if not clf_model:
            clf_model = input('Which classifier to use')
            print('\n'.join(map(str, self.clf_list)))

        # creation of inputs
        if feature_model == 'antiBERTy':
            X, y = self.create_inputs_antiBERTy(heavy_seqs, light_seqs, labels)
        elif feature_model == 'ABlang':
            X, y = self.create_inputs_ABlang(heavy_seqs, light_seqs, labels)
        elif feature_model == 'ESM':
            X, y = self.create_inputs_ESM(heavy_seqs, light_seqs, labels)
        else:
            print(f'==============================================')
            print("This model does not yet exist within this class")
            print(f'==============================================')

        # preprocessing of features
        selector = SelectKBest(score_func=f_classif, k=50) #selects top 50
        X = selector.fit_transform(X, y)
        pca = PCA(5)
        X = pca.fit_transform(X)

        # running of models and tuning; all the hyperparameters for each model are listed below
        if clf_model == 'SVC':
            self.param_grid = {
              'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
              'svc__degree': [2, 3, 4, 5],
              'svc__gamma': ['scale', 'auto'],
              'svc__probability': [True, False],
              'svc__C': [0.001, 0.01, 0.1, 1, 10],
              'svc__tol': [0.01, 0.001, 0.0001, 0.00001]
            }
            self.pipeline = Pipeline(steps=[
                ('scaler', StandardScaler()),
                ('svc', SVC(random_state = 42, class_weight = 'balanced'))
            ])
            self.pipeline.fit(X, y)

        elif clf_model == 'XGBClassifier':
            self.param_grid = {
              'xgbclassifier__lambda': [0.01],
              'xgbclassifier__n_estimators': [1],
              'xgbclassifier__max_depth': [1, 3, 5, 7, 10],
              'xgbclassifier__learning_rate': [0.1, 0.2, 0.3],
              'xgbclassifier__subsample': [0.5],
              'xgbclassifier__gamma': [0],
              'xgbclassifier__colsample_bytree': [0.5, 0.8, 1.0],
              'xgbclassifier__min_child_weight': [1, 5, 10],
              'xgbclassifier__eval_metric': ['logloss'],
              'xgbclassifier__tree_method': ['auto'],
            }
            self.pipeline = Pipeline(steps=[
                ('scaler', StandardScaler()),
                ('xgbclassifier', XGBClassifier(random_state = 42))
            ])
            self.pipeline.fit(X, y)

        elif clf_model == 'NuSVC':
            self.param_grid = {
                'NuSVC__nu': [0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.9],
                'NuSVC__kernel': ['linear', 'sigmoid', 'poly', 'rbf'],
                'NuSVC__degree': [2, 3, 4, 5],
                'NuSVC__gamma': ['scale', 'auto'],
                'NuSVC__probability': [True],
                'NuSVC__tol': [0.01, 0.001, 0.0001, 0.00001]
            }
            self.pipeline = Pipeline(steps=[
                ('scaler', StandardScaler()),
                ('NuSVC', NuSVC(random_state = 42, class_weight = 'balanced'))
            ])
            self.pipeline.fit(X, y)

        elif clf_model == 'SGDClassifier':
            self.param_grid = {
                "sgdclassifier__penalty": ["l1", "elasticnet"],
                "sgdclassifier__alpha": [0.001, 0.01, 0.1],
                "sgdclassifier__l1_ratio": [0.15, 0.2, 0.25],
                "sgdclassifier__max_iter": [100, 1000, 10000],
                "sgdclassifier__tol": [0.01, 0.001, 0.0001],
                "sgdclassifier__loss": ["modified_huber", "perceptron"],
                "sgdclassifier__learning_rate": [
                    "constant",
                    "optimal",
                    "invscaling",
                    "adaptive",
                ],
            }
            self.pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("sgdclassifier", SGDClassifier(random_state=42)),
                ]
            )
            self.pipeline.fit(X, y)

        elif clf_model == 'LinearSVC':
          self.param_grid = {
              'linearsvc__penalty': ['l1', 'l2'],
              'linearsvc__loss': ['hinge', 'squared_hinge'],
              'linearsvc__tol': [0.01, 0.001, 0.0001, 0.00001],
              'linearsvc__C': [0.001, 0.01, 0.1, 1, 10, 100],
              'linearsvc__multi_class': ['ovr', 'crammer_singer'],
              'linearsvc__max_iter': [100, 1000, 10000],
          }
          self.pipeline = Pipeline(steps=[
              ('scaler', StandardScaler()),
              ('linearsvc', LinearSVC(random_state = 42, class_weight = 'balanced'))
          ])
          self.pipeline.fit(X, y)
         
        else:
            print(f'==============================================')
            print("This model does not yet exist within this class")
            print(f'==============================================')
          
        if save_model:
            self.tune_hyperparameters_loo(X, y)
            self.alldata_test(X, y) # comment out if you want to test just the base model
            
            joblib.dump(self.voting_clf, "saved_model.joblib")
            #Load the model from the file
            model = joblib.load("saved_model.joblib")

            # Make predictions on the test set for all data score
            predictions = model.predict(X)
            accuracy = accuracy_score(y, predictions)
            cm = confusion_matrix(y, predictions)
            auc = roc_auc_score(y, predictions)
            precision = precision_score(y, predictions)
            recall = recall_score(y, predictions)
            f1 = f1_score(y, predictions)
            logloss = log_loss(y, predictions)
            
            # saving model for exportation
            X, y = self.create_inputs_ESM(heavy_seqs, light_seqs, labels)
            saved_pipeline = Pipeline(
                steps=[
                    ("SelectKBest", SelectKBest(k=50)),
                    ("PCA", PCA(5)),
                    ("scaler", StandardScaler()),
                    ("SVC", model),
                ]
            )
            saved_pipeline.fit(X, y)
            joblib.dump(saved_pipeline, "SVC_Classifier.joblib")
            
            test_df = pd.DataFrame(
                {
                    "Accuracy Score": f"{accuracy:.4}",
                    "True Positives": [cm[1, 1]],
                    "True Negatives": [cm[0, 0]],
                    "False Positives": [cm[0, 1]],
                    "False Negatives": [cm[1, 0]],
                    "AUC-ROC": f"{auc:.4}",
                    "Precision": f"{precision:.4}",
                    "Recall": f"{recall:.4}",
                    "F1-Score": f"{f1:.4}",
                    "Log Loss": f"{logloss:.4}",
                }
            )
            return test_df
          
        else:
            # tuning and validation of model for loo validation and train scores
            self.tune_hyperparameters_loo(X, y)
            final_printout = self.loo_validate(X, y)
            return final_printout
                      

"""**4. Example of running one algorithms with selected parameters**"""

bioavail_classifier = Classifier()
df = bioavail_classifier.run_clf(heavy_seqs, light_seqs, labels, 'ESM', 'SVC', True)
print(df)
print('done')

"""**5. Running top 5 algorithms for each feature generating model**"""

def run_top_5_classifiers(heavy_seqs, light_seqs, labels):
    top_5_classifiers = ['LinearSVC', 'SVC', 'NuSVC', 'XGBClassifier', 'SGDClassifier']
    
    all_results_df = pd.DataFrame(
        columns=["Feature Model", "Classifier"]
    )

    for feature_model in ["antiBERTy", "ABlang", "ESM"]:
        for clf in top_5_classifiers:
            # Run the classifier and get results (assume clf_results_df is modified accordingly)
            clf_results_df = bioavail_classifier.run_clf(heavy_seqs, light_seqs, labels, feature_model, clf)

            # Add the classifier, feature model, and other details to the DataFrame
            clf_results_df["Feature Model"] = feature_model
            clf_results_df["Classifier"] = clf

            # Append results to the main DataFrame
            all_results_df = pd.concat(
                [all_results_df, clf_results_df], ignore_index=True
            )

        print("completed language model")

    return all_results_df

# EXAMPLE USAGE
# df = run_top_5_classifiers(heavy_seqs, light_seqs, labels)
# df.to_excel("Bioavailability_Classifier_Stats.xlsx", index=False, sheet_name="no_feature_engineering")
# print(df)
