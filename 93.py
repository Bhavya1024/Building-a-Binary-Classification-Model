import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout 




####################
# TASK 1:         ##
####################

#-------------------
# DATASET 1:
#-------------------
def dataset1():

    # Step 1: read emoticon dataset
    train_emoticon_df = pd.read_csv("datasets/train/train_emoticon.csv")
    train_emoticon_X = train_emoticon_df['input_emoticon'].tolist()
    train_emoticon_X = [list(datapoint) for datapoint in train_emoticon_X]
    train_emoticon_Y = train_emoticon_df['label'].tolist()
    train_emoticon_Y = np.array(train_emoticon_Y)

    test_emoticon_X = pd.read_csv("datasets/test/test_emoticon.csv")['input_emoticon'].tolist()
    test_emoticon_X = [list(datapoint) for datapoint in test_emoticon_X]

    # Step 2: creating a list of all available emoticons
    emozi_set = set()
    for datapoint in train_emoticon_X+test_emoticon_X:
        for emozi in datapoint:
            emozi_set.add(emozi)

    ### This sorting is one of the key steps here. 
    ### The test set contains precisely 12 unique emoticon which are not found in the training set.
    ### The assumption is that - in the root data from which this emoticon dataset is derived, similar
    ### bit patterns with minimal variations would cause the same effect on the label, and those similar
    ### bit patterns would be converted to emoticons with adjacent unicode values. So sorting them according
    ### to unicode values would later allow the label encoder to assign them similar/adjacent encoding.
    ### So even if the model has not seen that exact emoticon during training, a similar emoticon with almost
    ### the same encoding will have the same effect on the label prediction.
    emozi_list = sorted(list(emozi_set),key=lambda ele:
                        ele.encode('unicode-escape').decode('ASCII'))

    ### upon data analysis we found that there are some emojis which exist in all of the emoji strings.
    ### they are: ('🙯', 14160), ('😣', 14160), ('😑', 14160), ('🛐', 7080), ('🚼', 7080), ('🙼', 7080), ('😛', 7080)
    ### the number associated to each of them denotes no. of times they exist in complete dataset.
    ### no. of datapoint in the dataset is 7080. 
    ### The top 3 emojis exist twice in every datapoint, later 4 exist once in every datapoint.

    ### further analysis revealed that these emozis don't actually contribute to the change in label
    ### and the rest 3 emozis in the 13-length sequence contribute much more to the labels changing from one datapoint to another

    # Step 3: removing these most frequent emozis from each datapoint
    most_freq_emozis = ['😛', '🚼', '🛐', '😑', '🙼', '🙯', '😣']

    train_emoticon_x_reduced = []
    test_emoticon_x_reduced = []

    for datapoint in train_emoticon_X:
        new_data = []
        for emozi in datapoint:
            if emozi in most_freq_emozis:
                # new_data.append('❌')
                continue
            else:
                new_data.append(emozi)
        train_emoticon_x_reduced.append(new_data)

    for datapoint in test_emoticon_X:
        new_data = []
        for emozi in datapoint:
            if emozi in most_freq_emozis:
                # new_data.append('❌')
                continue
            else:
                new_data.append(emozi)
        test_emoticon_x_reduced.append(new_data)

    # Step 4: using label encoder from sklearn to create labels for each emozi
    infrequent_emozi = []
    for emozi in emozi_list:
        if emozi not in most_freq_emozis:
            infrequent_emozi.append(emozi)
    le = LabelEncoder()
    le.fit(infrequent_emozi)

    # Step 5: transform training and test set
    train_emoticon_x_reduced = np.array([le.transform(datapoint) for datapoint in train_emoticon_x_reduced])
    test_emoticon_x_reduced = np.array([le.transform(datapoint) for datapoint in test_emoticon_x_reduced])


    ### Data analysis showed that even if all the 13 emoticons in two datapoints are same,
    ### their labels may differ (between 0 and 1) if the positions of emoticons are different.
    ### We use a Deep learning construct called Gated Recurrent Unit(GRU) which is quite effective
    ### at learning positional encoding and uses less parameters than LSTM and CNN.

    # Step 6: Create and train the model
    model = Sequential()
    model.add(Embedding(input_dim=len(infrequent_emozi), output_dim=16)) # input_dim=219 unique symbols, output_dim=16-dim embeddings
    model.add(GRU(32)) # GRU to learn positional embeddings. Can be replaced with LSTM or CNN
    model.add(Dropout(0.3)) # Dropout layer for regularization to prevent overfitting
    model.add(Dense(8, activation='relu')) # Dense layer with 'relu' activation function to learn a non-linear decision boundary
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_emoticon_x_reduced, train_emoticon_Y,
            epochs = 23,
            verbose = 0,
            shuffle = True
            )

    # Step 7: Do predictions on test data
    predictions = model.predict(test_emoticon_x_reduced) # predicts the probability of a datapoint having the label 1
    binary_predictions = (predictions > 0.5).astype(int) # if probability is greater than 0.5, label is predicted 1 else 0
    with open("pred_emoticon.txt", "w") as f:
        for pred in binary_predictions:
            f.write(f"{pred[0]}\n")

dataset1()

#-------------------
# DATASET 2:
#-------------------
def dataset2():
    # Step 1: Loading the dataset
    train = np.load("datasets/train/train_feature.npz")
    test = np.load("datasets/test/test_feature.npz")
    # Step 2 : Converting numpy arrays to list as lists allow for mixed data types, resizing, and other operations that are more complex or limited with NumPy arrays.
    X_train = np.array(train['features'].tolist())
    y_train = np.array(train['label'])
    X_test = np.array(test['features'].tolist())

    #  Step 3 :flattening
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Step 4: Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 5: Dimensionality Reduction using PCA
    pca = PCA(n_components=75)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Step 6: Classification using Support vector classifier
    svm_clf = SVC(C=0.6, kernel="rbf", random_state=42)
    svm_clf.fit(X_train_pca, y_train)

    # Step 7 :Predictions SVM using Rbf kernel
    predictions = svm_clf.predict(X_test_pca)
    with open("pred_deepfeat.txt", "w") as f:
        for pred in predictions:
            f.write(f"{pred}\n")

dataset2()



#-------------------
# DATASET 3:
#-------------------

### function to get a list of all possible substrings that are common
### between two strings str1 and str2
def get_common_substrings(str1, str2):
    substrings = []
    for start in range(len(str1)):
        for end in range(start+1, len(str1)+1):
            substr = str1[start:end]
            if str2.find(substr) != -1:
                substrings.append(substr)
    substrings = list(set(substrings))
    return substrings

def dataset3():

    ### For task 2 we are using dataset 1 and dataset 3
    ### Each emoticon in dataset 1 is given a text label made up of digits.
    ### Dataset 3 contains text sequence which is made by concatnation of those text labels
    ### corresponding to each datapoint in dataset1, which is further padded with leading zeros
    ### to make the text sequence of uniform length 50.
    ### The idea is to remove padding from the text sequence and break it into 13 parts such that 
    ### the broken parts correspond to each emoticon in the corresponding datapoint in emoticon
    ### dataset. Then learn the model based on that broken text sequence.


    # Step 1: read emoticon and text sequence dataset
    train_emoticon_df = pd.read_csv("datasets/train/train_emoticon.csv")
    train_emoticon_X = train_emoticon_df['input_emoticon'].tolist()
    train_emoticon_Y = train_emoticon_df['label'].tolist()
    train_emoticon_Y = np.array(train_emoticon_Y)

    test_emoticon_X = pd.read_csv("datasets/test/test_emoticon.csv")['input_emoticon'].tolist()

    train_text_seq_df = pd.read_csv("datasets/train/train_text_seq.csv")
    train_text_seq_X = train_text_seq_df['input_str'].tolist()
    train_text_seq_Y = train_text_seq_df['label'].tolist()
    train_text_seq_Y = np.array(train_text_seq_Y)

    test_text_seq_X = pd.read_csv("datasets/test/test_text_seq.csv")['input_str'].tolist()

    training_data_count = len(train_emoticon_X)
    test_data_count = len(test_emoticon_X)


    # Step 2: map corresponding datapoints from dataset 1 to dataset 3

    feature_mapping = dict()
    combined_emoticon_X = train_emoticon_X+test_emoticon_X
    combined_text_seq_X = train_text_seq_X+test_text_seq_X

    for i in range(training_data_count+test_data_count):
        feature_mapping[combined_emoticon_X[i]] = combined_text_seq_X[i]


    # Step 3: Create a list of all available emoticons

    emozi_set = set()
    for datapoint in combined_emoticon_X:
        for emozi in datapoint:
            emozi_set.add(emozi)
    emozi_list = sorted(list(emozi_set),key=lambda ele:
                            ele.encode('unicode-escape').decode('ASCII'))
    

    # Step 4: find the text encoding for each emoticon

    ### APPROACH: We manually found the encodings of 7 "frequent" emoticons
    ### (as discussed in methodology applied on dataset 1) which are present
    ### in all the datapoints of dataset1, that means their encodings would be 
    ### present in corresponding datapoints of dataset 3 as well. For finding 
    ### the encodings of the rest of the emoticons, we choose an emoticon and
    ### select all the datapoints from dataset 1 which contains that emoticon
    ### and corresponding text sequences from dataset 3. Then we find all the 
    ### substrings that are common in those chosen text sequences. Those common
    ### substrings correspond to those 7 "frequent" emoticons (whose encodings
    ### we already have) and one another emoticon whose encoding we wanted to
    ### find. Since we already know the encodings of frequent emoticons we
    ### remove them from the common substrings set and we are left with the
    ### required encoding of that emoticon

    toppers = ['😛', '🚼', '🙯', '🙼', '🛐', '😑', '😣'] # 7 "frequent" emoticons
    toppers_enc = ['15436', '422', '262', '284', '464', '1596', '614'] # their encodings present in dataset 3

    symbol_label = dict() # dictionary to store encoding of each emoticon

    for emozi in emozi_list: # selecting an emoticon from the available list
        e_seq_list = []  # list to store datapoints from dataset 1 that contains that emoticon
        t_seq_list = []  # list to store text sequence from dataset 3 corresponding to each datapoint in "e_seq_list"
        for seq in combined_emoticon_X:
            if emozi in seq:
                e_seq_list.append(seq)
                t_seq_list.append(feature_mapping[seq])

        # code to find common substrings in emoticon sequence and text sequence
        common_emozi = get_common_substrings(e_seq_list[0], e_seq_list[1])
        common_text = get_common_substrings(t_seq_list[0], t_seq_list[1])

        for i in range(len(e_seq_list)):
            curr_emozi_seq = e_seq_list[i]
            curr_text_seq = t_seq_list[i]
            
            for emozi_str in common_emozi.copy():
                if curr_emozi_seq.find(emozi_str) == -1:
                    common_emozi.remove(emozi_str)
            for text in common_text.copy():
                if curr_text_seq.find(text) == -1:
                    common_text.remove(text)
        
        for emozi_ in common_emozi.copy():
            if emozi_ in toppers:
                common_emozi.remove(emozi_)

        
        for enc in common_text.copy():
            for enc_ in toppers_enc:
                if enc_.find(enc) != -1:
                    common_text.remove(enc)
                    break

        # storing the (emoticon, encoding) pair in dictionary 
        symbol_label[emozi] = max([int(string) for string in common_text])

    # adding "frequent" emoticon's encoding to the dictionary as well 
    for i in range(len(toppers)):
        symbol_label[toppers[i]] = int(toppers_enc[i])   


    # Step 5: Rebuilding the text_sequence dataset but with each emoticon's
    # encoding separated from each other using Dataset 1 and emoji encoding
    # dictionary (without including encodings of "frequent" emoticons (because
    # as discussed in Dataset 1 methodology, they do not contribute to the 
    # variations in label of the dataset).

    train_text_X2 = []
    test_text_X2 = []

    for string in train_emoticon_X:
        features = []
        for emozi in string:
            if emozi in toppers:
                # features.append(0)
                continue
            else:
                features.append(symbol_label[emozi])
        features = np.array(features)
        train_text_X2.append(features)
    train_text_X2 = np.array(train_text_X2)

    for string in test_emoticon_X:
        features = []
        for emozi in string:
            if emozi in toppers:
                # features.append(0)
                continue
            else:
                features.append(symbol_label[emozi])
        features = np.array(features)
        test_text_X2.append(features)
    test_text_X2 = np.array(test_text_X2)   


    # Step 6: Create a set of all encodings and provide a label to each encoding

    code_set = set()
    for symbol in symbol_label:
        code_set.add(symbol_label[symbol])
    code_set = list(code_set) # contains list of all encodings

    ### since each encoding is still used as a catagorical data, and 226 unique
    ### encodings exist in dataset 3, each corresponding to an emoticon in dataset 1,
    ### each encoding is converted to yet another encoding which is just an ordinal
    ### number given to each encoding.

    code_to_label = {code:label for label,code in enumerate(code_set)} # dictionary to store ordinal no. of each encoding


    # Step 7: Transform training and test set datapoints according to new encodings
    
    train_X2_transformed = [] # final training set
    test_X2_transformed = [] # final test set

    for datapoint in train_text_X2:
        train_X2_transformed.append([code_to_label[code] for code in datapoint])
    for datapoint in test_text_X2:
        test_X2_transformed.append([code_to_label[code] for code in datapoint])
    train_X2_transformed = np.array(train_X2_transformed)
    test_X2_transformed = np.array(test_X2_transformed)

    # Step 8: Train the model on transformed training set
    model3 = Sequential()
    model3.add(Embedding(input_dim=len(code_set), output_dim=16)) # input_dim=219 unique symbols, output_dim=16-dim embeddings
    model3.add(GRU(32)) # GRU to learn positional embeddings. Can be replaced with LSTM or CNN
    model3.add(Dropout(0.3)) # Dropout layer for regularization to prevent overfitting
    model3.add(Dense(8, activation='relu')) # Dense layer with 'relu' activation function to learn a non-linear decision boundary
    model3.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

    model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model3.fit(train_X2_transformed, train_text_seq_Y,
           epochs = 13,
           verbose = 0,
           shuffle = True
        )
    
    # Step 9: Perform prediction on test set
    predictions = model3.predict(test_X2_transformed) # predicts the probability of a datapoint having the label 1
    binary_predictions = (predictions > 0.5).astype(int) # if probability is greater than 0.5, label is predicted 1 else 0
    with open("pred_textseq.txt", "w") as f:
        for pred in binary_predictions:
            f.write(f"{pred[0]}\n")

dataset3()



####################
# TASK 2:         ##
####################

### Applying stacking of models
### Training "lower" models on individual datasets to generate class probabilities and later
### training another "upper" model on those class probabilities generated by "lower" models.



## Training Model 1:-----------------------

# Step 1: read emoticon dataset
train_emoticon_df = pd.read_csv("datasets/train/train_emoticon.csv")
train_emoticon_X = train_emoticon_df['input_emoticon'].tolist()
train_emoticon_X = [list(datapoint) for datapoint in train_emoticon_X]
train_emoticon_Y = train_emoticon_df['label'].tolist()
train_emoticon_Y = np.array(train_emoticon_Y)

valid_emoticon_df = pd.read_csv("datasets/valid/valid_emoticon.csv")
valid_emoticon_X = valid_emoticon_df['input_emoticon'].tolist()
valid_emoticon_X = [list(datapoint) for datapoint in valid_emoticon_X]
valid_emoticon_Y = valid_emoticon_df['label'].tolist()
valid_emoticon_Y = np.array(valid_emoticon_Y)

test_emoticon_X = pd.read_csv("datasets/test/test_emoticon.csv")['input_emoticon'].tolist()
test_emoticon_X = [list(datapoint) for datapoint in test_emoticon_X]

# Step 2: creating a list of all available emoticons
emozi_set = set()
for datapoint in train_emoticon_X+test_emoticon_X:
    for emozi in datapoint:
        emozi_set.add(emozi)
emozi_list = sorted(list(emozi_set),key=lambda ele:
                        ele.encode('unicode-escape').decode('ASCII'))

# Step 3: removing these most frequent emozis from each datapoint
most_freq_emozis = ['😛', '🚼', '🛐', '😑', '🙼', '🙯', '😣']

train_emoticon_x_reduced = []
valid_emoticon_x_reduced = []
test_emoticon_x_reduced = []

for datapoint in train_emoticon_X:
    new_data = []
    for emozi in datapoint:
        if emozi in most_freq_emozis:
            # new_data.append('❌')
            continue
        else:
            new_data.append(emozi)
    train_emoticon_x_reduced.append(new_data)

for datapoint in valid_emoticon_X:
    new_data = []
    for emozi in datapoint:
        if emozi in most_freq_emozis:
            # new_data.append('❌')
            continue
        else:
            new_data.append(emozi)
    valid_emoticon_x_reduced.append(new_data)

for datapoint in test_emoticon_X:
    new_data = []
    for emozi in datapoint:
        if emozi in most_freq_emozis:
            # new_data.append('❌')
            continue
        else:
            new_data.append(emozi)
    test_emoticon_x_reduced.append(new_data)

# Step 4: using label encoder from sklearn to create labels for each emozi
infrequent_emozi = []
for emozi in emozi_list:
    if emozi not in most_freq_emozis:
        infrequent_emozi.append(emozi)
le = LabelEncoder()
le.fit(infrequent_emozi)

# Step 5: transform training and test set
train_emoticon_x_reduced = np.array([le.transform(datapoint) for datapoint in train_emoticon_x_reduced])
valid_emoticon_x_reduced = np.array([le.transform(datapoint) for datapoint in valid_emoticon_x_reduced])
test_emoticon_x_reduced = np.array([le.transform(datapoint) for datapoint in test_emoticon_x_reduced])

# Step 6: Create and train the model
model = Sequential()
model.add(Embedding(input_dim=len(infrequent_emozi), output_dim=12)) # input_dim=219 unique symbols, output_dim=16-dim embeddings
model.add(GRU(16)) # GRU to learn positional embeddings. Can be replaced with LSTM or CNN
model.add(Dropout(0.3)) # Dropout layer for regularization to prevent overfitting
model.add(Dense(8, activation='relu')) # Dense layer with 'relu' activation function to learn a non-linear decision boundary
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_emoticon_x_reduced, train_emoticon_Y,
        epochs = 23,
        verbose = 0,
        shuffle = True
        )

# probability data from model 1
emoticon_train_pred = model.predict(train_emoticon_x_reduced)
emoticon_valid_pred = model.predict(valid_emoticon_x_reduced)
emoticon_test_pred = model.predict(test_emoticon_x_reduced)


## Training Model 2:---------------------

# Step 1: reading deep features dataset
train_feat = np.load("datasets/train/train_feature.npz", allow_pickle=True)
train_feat_X = train_feat['features']
train_feat_Y = train_feat['label']

val_feat = np.load("datasets/valid/valid_feature.npz", allow_pickle=True)
val_feat_X = val_feat['features']
val_feat_Y = val_feat['label']

test_feat_X = np.load("datasets/test/test_feature.npz", allow_pickle=True)['features']

# Step 2: flattening the 13x768 matrices to 1-D array
train_feat_X_flattened = train_feat_X.reshape(train_feat_X.shape[0], -1)
val_feat_X_flattened = val_feat_X.reshape(val_feat_X.shape[0], -1)
test_feat_X_flattened = test_feat_X.reshape(test_feat_X.shape[0], -1)

# Step 3: standardizing the inputs using standard scalar form sklearn
scalar = StandardScaler()
train_feat_X_scaled = scalar.fit_transform(train_feat_X_flattened)
val_feat_X_scaled = scalar.transform(val_feat_X_flattened)
test_feat_X_scaled = scalar.transform(test_feat_X_flattened)

# Step 4: reducing data dimensionality from 9984 (13*768 = 9984) to 75 using PCA
pca = PCA(n_components=75)
pca.fit(train_feat_X_scaled)
train_feat_X_transformed = pca.transform(train_feat_X_scaled)
val_feat_X_transformed = pca.transform(val_feat_X_scaled)
test_feat_X_transformed = pca.transform(test_feat_X_scaled)

# Step 5: training Support Vector Classifier on transformed data
svc = SVC(C=0.6, kernel='rbf',random_state=42, probability=True)
svc.fit(train_feat_X_transformed, train_feat_Y)

# probability data from model 2
feat_train_pred = svc.predict_proba(train_feat_X_transformed)[:,1].reshape(-1,1)
feat_val_pred = svc.predict_proba(val_feat_X_transformed)[:,1].reshape(-1,1)
feat_test_pred = svc.predict_proba(test_feat_X_transformed)[:,1].reshape(-1,1)



## Training Model 3:--------------

# Step 1: read emoticon and text sequence dataset
train_emoticon_df = pd.read_csv("datasets/train/train_emoticon.csv")
train_emoticon_X = train_emoticon_df['input_emoticon'].tolist()
train_emoticon_Y = train_emoticon_df['label'].tolist()
train_emoticon_Y = np.array(train_emoticon_Y)

val_emoticon_df = pd.read_csv("datasets/valid/valid_emoticon.csv")
val_emoticon_X = val_emoticon_df['input_emoticon'].tolist()
val_emoticon_Y = val_emoticon_df['label'].tolist()
val_emoticon_Y = np.array(val_emoticon_Y)

test_emoticon_X = pd.read_csv("datasets/test/test_emoticon.csv")['input_emoticon'].tolist()

train_text_seq_df = pd.read_csv("datasets/train/train_text_seq.csv")
train_text_seq_X = train_text_seq_df['input_str'].tolist()
train_text_seq_Y = train_text_seq_df['label'].tolist()
train_text_seq_Y = np.array(train_text_seq_Y)

val_text_seq_df = pd.read_csv("datasets/valid/valid_text_seq.csv")
val_text_seq_X = val_text_seq_df['input_str'].tolist()
val_text_seq_Y = val_text_seq_df['label'].tolist()
val_text_seq_Y = np.array(val_text_seq_Y)

test_text_seq_X = pd.read_csv("datasets/test/test_text_seq.csv")['input_str'].tolist()

training_data_count = len(train_emoticon_X)
validation_data_count = len(val_emoticon_X)
test_data_count = len(test_emoticon_X)

# Step 2: map corresponding datapoints from dataset 1 to dataset 3
feature_mapping = dict()
combined_emoticon_X = train_emoticon_X+test_emoticon_X
combined_text_seq_X = train_text_seq_X+test_text_seq_X

for i in range(training_data_count+test_data_count):
    feature_mapping[combined_emoticon_X[i]] = combined_text_seq_X[i]

# Step 3: Create a list of all available emoticons
emozi_set = set()
for datapoint in combined_emoticon_X:
    for emozi in datapoint:
        emozi_set.add(emozi)
emozi_list = sorted(list(emozi_set),key=lambda ele:
                        ele.encode('unicode-escape').decode('ASCII'))

# Step 4: find the text encoding for each emoticon
toppers = ['😛', '🚼', '🙯', '🙼', '🛐', '😑', '😣'] # 7 "frequent" emoticons
toppers_enc = ['15436', '422', '262', '284', '464', '1596', '614'] # their encodings present in dataset 3

symbol_label = dict() # dictionary to store encoding of each emoticon

for emozi in emozi_list: # selecting an emoticon from the available list
    e_seq_list = []  # list to store datapoints from dataset 1 that contains that emoticon
    t_seq_list = []  # list to store text sequence from dataset 3 corresponding to each datapoint in "e_seq_list"
    for seq in combined_emoticon_X:
        if emozi in seq:
            e_seq_list.append(seq)
            t_seq_list.append(feature_mapping[seq])

    # code to find common substrings in emoticon sequence and text sequence
    common_emozi = get_common_substrings(e_seq_list[0], e_seq_list[1])
    common_text = get_common_substrings(t_seq_list[0], t_seq_list[1])

    for i in range(len(e_seq_list)):
        curr_emozi_seq = e_seq_list[i]
        curr_text_seq = t_seq_list[i]
        
        for emozi_str in common_emozi.copy():
            if curr_emozi_seq.find(emozi_str) == -1:
                common_emozi.remove(emozi_str)
        for text in common_text.copy():
            if curr_text_seq.find(text) == -1:
                common_text.remove(text)
    
    for emozi_ in common_emozi.copy():
        if emozi_ in toppers:
            common_emozi.remove(emozi_)

    
    for enc in common_text.copy():
        for enc_ in toppers_enc:
            if enc_.find(enc) != -1:
                common_text.remove(enc)
                break

    # storing the (emoticon, encoding) pair in dictionary 
    symbol_label[emozi] = max([int(string) for string in common_text])

# adding "frequent" emoticon's encoding to the dictionary as well 
for i in range(len(toppers)):
    symbol_label[toppers[i]] = int(toppers_enc[i])   

# Step 5: Rebuilding the text_sequence dataset but with each emoticon's
# encoding separated from each other using Dataset 1 and emoji encoding
# dictionary (without including encodings of "frequent" emoticons (because
# as discussed in Dataset 1 methodology, they do not contribute to the 
# variations in label of the dataset).
train_text_X2 = []
val_text_X2 = []
test_text_X2 = []

for string in train_emoticon_X:
    features = []
    for emozi in string:
        if emozi in toppers:
            # features.append(0)
            continue
        else:
            features.append(symbol_label[emozi])
    features = np.array(features)
    train_text_X2.append(features)
train_text_X2 = np.array(train_text_X2)

for string in val_emoticon_X:
    features = []
    for emozi in string:
        if emozi in toppers:
            # features.append(0)
            continue
        else:
            features.append(symbol_label[emozi])
    features = np.array(features)
    val_text_X2.append(features)
val_text_X2 = np.array(val_text_X2)

for string in test_emoticon_X:
    features = []
    for emozi in string:
        if emozi in toppers:
            # features.append(0)
            continue
        else:
            features.append(symbol_label[emozi])
    features = np.array(features)
    test_text_X2.append(features)
test_text_X2 = np.array(test_text_X2)

# Step 6: Create a set of all encodings and provide a label to each encoding
code_set = set()
for symbol in symbol_label:
    code_set.add(symbol_label[symbol])
code_set = list(code_set) # contains list of all encodings
code_to_label = {code:label for label,code in enumerate(code_set)} # dictionary to store ordinal no. of each encoding

# Step 7: Transform training and test set datapoints according to new encodings
train_X2_transformed = [] # final training set
val_X2_transformed = [] # final validation set
test_X2_transformed = [] # final test set

for datapoint in train_text_X2:
    train_X2_transformed.append([code_to_label[code] for code in datapoint])
for datapoint in val_text_X2:
    val_X2_transformed.append([code_to_label[code] for code in datapoint])
for datapoint in test_text_X2:
    test_X2_transformed.append([code_to_label[code] for code in datapoint])
train_X2_transformed = np.array(train_X2_transformed)
val_X2_transformed = np.array(val_X2_transformed)
test_X2_transformed = np.array(test_X2_transformed)

# Step 8: Train the model on transformed training set
model3 = Sequential()
model3.add(Embedding(input_dim=len(code_set), output_dim=12)) # input_dim=219 unique symbols, output_dim=12-dim embeddings
model3.add(GRU(16)) # GRU to learn positional embeddings. Can be replaced with LSTM or CNN
model3.add(Dropout(0.3)) # Dropout layer for regularization to prevent overfitting
model3.add(Dense(8, activation='relu')) # Dense layer with 'relu' activation function to learn a non-linear decision boundary
model3.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model3.fit(train_X2_transformed, train_text_seq_Y,
        epochs = 13,
        verbose = 0,
        shuffle = True
    )

#probability data from model 3
text_train_pred = model3.predict(train_X2_transformed)
text_val_pred = model3.predict(val_X2_transformed)
text_test_pred = model3.predict(test_X2_transformed)

## Training "upper" model:--------------------

# Step 1: create probability dataset along with the actual predicted class values

## columns are arranged as follows: [p1, p2, p3, c1, c2, c3]
## p1 = probability predicted by model trained on dataset 1
## c1 = class predicted by model trained on dataset 1
## p2 = probability predicted by model trained on dataset 2
## c2 = class predicted by model trained on dataset 2
## p3 = probability predicted by model trained on dataset 3
## c3 = class predicted by model trained on dataset 3

probability_X_train = np.array([[emoticon_train_pred[i][0],feat_train_pred[i][0],text_train_pred[i][0], round(emoticon_train_pred[i][0]), round(feat_train_pred[i][0]), round(text_train_pred[i][0])] for i in range(training_data_count)])
probability_X_val = np.array([[emoticon_valid_pred[i][0], feat_val_pred[i][0], text_val_pred[i][0], round(emoticon_valid_pred[i][0]), round(feat_val_pred[i][0]), round(text_val_pred[i][0])] for i in range(validation_data_count)])
probability_X_test = np.array([[emoticon_test_pred[i][0], feat_test_pred[i][0], text_test_pred[i][0], round(emoticon_test_pred[i][0]), round(feat_test_pred[i][0]), round(text_test_pred[i][0])] for i in range(test_data_count)])

y_train = train_emoticon_Y
y_val = val_emoticon_Y

# Step 2: feature transformation

## in binary classification the predictions can be seen as checking a condition if the class probability is greater than 0.5 or not,
## that means the closer the value is to 0.5 the farther it is from complete sureity(0 or 1) and also the more unsure the model is about that prediction.
## following transfromation converts the probability value into a metric about how unsure the model is about the prediction by calculating how the probability
## is from either of the ends(0 or 1).

probability_X_train[:,:3] = 0.5-abs(probability_X_train[:,:3]-0.5)
probability_X_val[:,:3] = 0.5-abs(probability_X_val[:,:3]-0.5)
probability_X_test[:,:3] = 0.5-abs(probability_X_test[:,:3]-0.5)

# Step 3: using Standard scalar for standardizing the "unsureity" values
scalar2 = StandardScaler()
columns_to_transform = [0,1,2]
transformer = ColumnTransformer(
    transformers=[("scaler", scalar2, columns_to_transform)], remainder='passthrough'
)
probability_X_train_scaled = transformer.fit_transform(probability_X_train)
probability_X_val_scaled = transformer.transform(probability_X_val)
probability_X_test_scaled = transformer.transform(probability_X_test)

# Step 4: Training Support vector classifier on "unsureity" values
stacked_svc = SVC(kernel='rbf', gamma=0.1, C=1, random_state=42)
stacked_svc.fit(probability_X_train_scaled, y_train)

# Step 5: predicting classes for test data
predictions = stacked_svc.predict(probability_X_test_scaled) # predicts the probability of a datapoint having the label 1
with open("pred_combined.txt", "w") as f:
    for pred in predictions:
        f.write(f"{pred}\n")