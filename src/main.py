from utils import (
    load_data,
    train_svm,
    train_random_forest,
    train_pca_svm,
    train_deep_learning_model
)

def main():
    # Load Data
    x_train, x_test, y_train, y_test = load_data()

    # Train Models
    train_svm(x_train, x_test, y_train, y_test)
    train_random_forest(x_train, x_test, y_train, y_test)
    train_pca_svm(x_train, x_test, y_train, y_test)
    train_deep_learning_model(x_train, x_test, y_train, y_test)

    print("All Models Trained Successfully!")

if __name__ == "__main__":
    main()

