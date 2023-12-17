import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import scipy.stats as stats
import time
import seaborn as sns

from sklearn.model_selection import KFold
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_curve, auc


# Define a class for adaptive feature scaling
class AdaptiveFeatureScaler:
    def __init__(self):
        self.feature_ranges = None

    def fit(self, X):
        # Calculate the range (max - min) for each feature
        self.feature_ranges = X.max(axis=0) - X.min(axis=0)

    def transform(self, X):
        # Scale each feature based on the calculated range
        if self.feature_ranges is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        return X / self.feature_ranges

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# Function to preprocess data
def preprocessData(dataPath):
    df = pd.read_csv(dataPath)

    columns = ['RSRP', 'RSRQ', 'Light', 'Mag', 'Acc', 'Sound', 'Proximity', 'Daytime', 'New_Recording', 'IO']
    data = pd.DataFrame(data=df, columns=columns)

    # Convert the string formatted data into float
    data = data.astype('float')

    # AdaptiveFeatureScaler  
    afs = AdaptiveFeatureScaler()
    data_scaled = afs.fit_transform(data[['RSRP', 'RSRQ', 'Light', 'Mag', 'Acc', 'Sound', 'Proximity', 'Daytime']])

    # Combine the scaled features and the label
    data_scaled['IO'] = data['IO']

    # Continue with the rest of the data processing as before
    states = data_scaled['IO'].value_counts().index
    data_scaled.reset_index(drop=True, inplace=True)
    data_scaled.index = data_scaled.index + 1
    data_scaled.index.name = 'index'

    return df, data_scaled

# Function to balance data by selecting the same number of samples for each class
def balanceData(df, data):
    # Get the value counts of the 'IO' column
    value_counts = df['IO'].value_counts()

    # Find the minimum count of both labels
    min_count = min(value_counts)

    # Filter the DataFrame for 'Outdoor' and 'Indoor' categories
    Outdoor = df[df['IO'] == 0].head(min_count).copy()
    Indoor = df[df['IO'] == 1].head(min_count).copy()

    balanced_data = pd.concat([Outdoor, Indoor], ignore_index=True)

    return balanced_data

# Function to encode the data labels using LabelEncoder
def encodedData(balanced_data):
    # Encoding the Data with suitable labels
    label = LabelEncoder()
    balanced_data['label'] = label.fit_transform(balanced_data['IO'])
    return balanced_data

# Function to standardize the features
def standardizeData(encoded_data):
    X = encoded_data[['RSRP', 'RSRQ', 'Light', 'Mag', 'Acc', 'Sound', 'Proximity', 'Daytime']]
    y = encoded_data['label']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    scaled_X = pd.DataFrame(data=X, columns=['RSRP', 'RSRQ', 'Light', 'Mag', 'Acc', 'Sound', 'Proximity', 'Daytime'])
    scaled_X['label'] = y.values

    return scaled_X, X, y

# Function to adaptiveScaleData the features with AdaptiveFeatureScaler
def adaptiveScaleData(encoded_data):
    X = encoded_data[['RSRP', 'RSRQ', 'Light', 'Mag', 'Acc', 'Sound', 'Proximity', 'Daytime']]
    y = encoded_data['label']

    afs = AdaptiveFeatureScaler()  # Create an instance of AdaptiveFeatureScaler
    X = afs.fit_transform(X)  # Use fit_transform to transform the features

    scaled_X = pd.DataFrame(data=X, columns=['RSRP', 'RSRQ', 'Light', 'Mag', 'Acc', 'Sound', 'Proximity', 'Daytime'])
    scaled_X['label'] = y.values

    return scaled_X, X, y

# Function to create overlapping frames from the data
def get_frames(df, frame_size, hop_size, n_features=8):
    N_FEATURES = n_features

    frames = []
    labels = []
    for i in range(0, len(df) - frame_size, hop_size):
        # Get each feature for the current frame
        features = [df[feature].values[i: i + frame_size] for feature in df.columns[:-1]]

        # Retrieve the most often used label in this segment
        label = stats.mode(df['label'][i: i + frame_size])[0][0]
        frames.append(features)
        labels.append(label)

    # Bring the segments into a better shape
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)

    return frames, labels

# Function to create overlapping frames from the standardized data
def framedData(scaled_X, X, y, window_size=6, features_number=8):
    frame_size = window_size
    hop_size = int(frame_size / 2)
    n_features = features_number

    # Remove rows with NaN values and reset index
    scaled_X = scaled_X.dropna().reset_index(drop=True)
    X, y = get_frames(scaled_X, frame_size, hop_size, n_features)
    return scaled_X, X, y

# Function to split the data with cross-validation
def splitDataWithCrossValidation(X, y, fold_number=6):
    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Creating the K-fold cross-validation iterator
    kfold = StratifiedKFold(n_splits=fold_number, shuffle=True, random_state=0)

    return X_train, X_test, y_train, y_test, kfold