from tokenize_text import *


def main():
    global X_train, X_val, y_train, y_val

    print("\tShape Train and Validation Set")
    print("Train: {}, Validation: {}".format(X_train.shape, X_val.shape))


if __name__ == '__main__':
    main()
    