
def dataset3():
    
    import pandas as pd
    import numpy as np
    from sklearn.metrics import accuracy_score
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization, Attention
    from tensorflow.keras import layers
    
    # read text sequence dataset
    train_seq_df = pd.read_csv("datasets/train/train_text_seq.csv")
    train_seq_X = train_seq_df['input_str'].tolist()
    train_seq_Y = train_seq_df['label'].tolist()
       
    test_seq_X = pd.read_csv("datasets/test/test_text_seq.csv")['input_str'].tolist()

    # Each input which is a string of 50 digits is coverted into numpy integer array of 50 digits([1 2 0 4 2...])
    X_train= np.array([list(map(int,train_seq_X[i])) for i in range(7080)])
    y_train=np.array(train_seq_Y)
    X_test=np.array([list(map(int,test_seq_X[i])) for i in range(len(test_seq_X))])

    # LSTMs are well-suited for sequence data like time series or text, but here we can treat each sample's 50 features as a sequence. 
    # We’ll design an LSTM network and apply it to the dataset.
        
    # Model definition
    model = Sequential()
    
    #1. Add an Embedding layer '''
        
    embedding_dim = 16
    model.add(Embedding(input_dim=214, output_dim=embedding_dim, input_length=50))
    
    # 2. LSTM layer with 30 units
    model.add(LSTM(30))
    
    # 3. Dropout layer to prevent overfitting (0.5 dropout)
    model.add(Dropout(0.5))
    
    # 4. Batch Normalization layer for stable learning
    model.add(BatchNormalization())
    
    # 5. Dense layer with 15 units (ReLU activation)
    model.add(Dense(15, activation='relu'))
    
    # 6. Another Dropout layer to further regularize
    model.add(Dropout(0.5))
    
    # 7. Dense layer with 15 units (ReLU activation)
    model.add(Dense(15, activation='relu'))
    
    # 8. Final output layer for binary classification (1 unit, sigmoid activation)
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.004), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    # Train the model 
    model.fit(X_train, y_train, epochs=75, batch_size=32,verbose=1)
    
    # Make predictions on the test set
    predictions = model.predict(X_test)
    
    # Convert predictions to binary labels (0 or 1)
    predicted_labels = (predictions > 0.5).astype(int).reshape(-1)
    
    # Store the predictions in a text file with the column name "test label"
    with open("test_labels.txt", "w") as file:
        file.write("test label\n")  # Column header
        for label in predicted_labels:
            file.write(f"{label}\n")

    # Model Summary
    if percent==1:
        model.summary()
    return accuracy

