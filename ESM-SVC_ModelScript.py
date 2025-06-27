import joblib
import numpy as np
import pandas as pd
import esm.pretrained
from torch import manual_seed, no_grad

class Classifier:
    def __init__(self, model):
        self.loaded_model = model

    def create_inputs_ESM(self, heavy_seqs, light_seqs):
        
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
        
        return X


    def run_clf(self, heavy_seqs, light_seqs):

        manual_seed(42)
        np.random.seed(42)
        
        X = self.create_inputs_ESM(heavy_seqs, light_seqs)

        # Load the model back into memory
        SVC_Classifier_model = self.loaded_model
        
        predictions = SVC_Classifier_model.predict(X)
        
        return predictions


# === EXAMPLE USAGE ===

# Load in sklearn model
model = joblib.load("SVC_Classifier.joblib")

clf = Classifier(model)

# Load input data
df = pd.read_csv("bioavailability-and-sequences.csv")
heavy = df["Heavy_Var"]
light = df["Light_Var"]
labels = df["High/Low"]

# Run and Test
predictions = clf.run_clf(heavy, light)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels, predictions)
print(f"Accuracy: {accuracy}")
