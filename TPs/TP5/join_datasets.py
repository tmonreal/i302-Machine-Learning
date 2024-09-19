import pandas as pd

# Read the train and test CSV files
train_df = pd.read_csv("TP4/data/mnist_train.csv")
test_df = pd.read_csv("TP4/data/mnist_test.csv")

# Verify the shapes to ensure they are the same
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# Concatenate the two DataFrames
full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# Verify the shape of the concatenated DataFrame
print("Full dataset shape:", full_df.shape)

# Optionally, save the combined DataFrame to a new CSV file
full_df.to_csv("TP4/data/MNIST_dataset.csv", index=False)

print("The full dataset has been created and saved.")
