import pandas as pd
import streamlit as st
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler

def select_k_best(X, y, k):
    skb = SelectKBest(score_func=f_classif, k=k)
    skb.fit(X, y)
    selected_features = X.columns[skb.get_support()]
    return selected_features

def forward_elimination(X, y, k):
    lr = LinearRegression()
    fe = RFE(estimator=lr, n_features_to_select=k)
    fe.fit(X, y)
    selected_features = X.columns[fe.support_]
    return selected_features

def backward_elimination(X, y, k):
    X_be = X.copy()
    selected_features = []
    worst_features = []
    while X_be.shape[1] > k:
        model = LinearRegression()
        model.fit(X_be, y)
        p_values = pd.Series(model.coef_, index=X_be.columns)
        worst_feature = p_values.idxmax()
        if worst_feature in selected_features:
            continue
        worst_features.append(worst_feature)
        selected_features.append(worst_feature)
        X_be.drop(worst_feature, axis=1, inplace=True)
    return selected_features, worst_features

# Streamlit app
def main():
    st.title("Feature Selection Demo")

    # File upload
    st.sidebar.title("Upload CSV file")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Read the dataset
            df = pd.read_csv(uploaded_file)

            # Display the dataset
            st.subheader("Dataset")
            st.dataframe(df)

            # Drop unnecessary columns
            df.drop(['User_ID', 'User_Engagement'], axis=1, inplace=True)

            # Separate features and target variable
            X = df.drop('User_Interactions', axis=1)
            y = df['User_Interactions']

            # Preprocess categorical columns
            categorical_cols = X.select_dtypes(include=['object']).columns
            X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

            # Preprocess numeric columns
            numeric_cols = X_encoded.select_dtypes(include=['int64', 'float64']).columns
            scaler = StandardScaler()
            X_encoded[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])

            # Feature selection
            k = 5

            st.subheader("Feature Selection Results")

            # SelectKBest
            st.write("SelectKBest:")
            selected_features_skb = select_k_best(X_encoded, y, k)
            st.write("Top Features:", selected_features_skb)

            # Forward Elimination
            st.write("Forward Elimination:")
            selected_features_fe = forward_elimination(X_encoded, y, k)
            st.write("Top Features:", selected_features_fe)

            # Backward Elimination
            st.write("Backward Elimination:")
            selected_features_be, worst_features_be = backward_elimination(X_encoded, y, k)
            st.write("Top Features:")
            for feature in selected_features_be:
                if feature not in worst_features_be:
                    st.write("-", feature)
            st.write("Bottom Features:")
            for feature in worst_features_be:
                if feature not in selected_features_be:
                    st.write("-", feature)

        except Exception as e:
            st.error("Error: " + str(e))

# Run the app
if __name__ == "__main__":
    main()

