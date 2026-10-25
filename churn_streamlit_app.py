import streamlit as st
from predict_churn import load_artifact, predict_from_dict

st.set_page_config(page_title='Churn Prediction', layout='centered')

st.title('Customer Churn - Logistic Regression')
st.write('Enter customer features and click Generate Prediction to get churn probability and label.')

@st.cache_resource
def get_model():
    return load_artifact()

try:
    model, feature_names, cat_cols = get_model()
except Exception as e:
    st.error(f'Error loading model artifact: {e}')
    st.stop()

# Provide inputs
with st.form('input_form'):
    st.header('Customer features')
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox('Gender', ['Male', 'Female'])
        senior = st.selectbox('SeniorCitizen', [0, 1], index=0)
        partner = st.selectbox('Partner', ['Yes', 'No'])
        dependents = st.selectbox('Dependents', ['Yes', 'No'])
        tenure = st.number_input('tenure (months)', min_value=0, max_value=240, value=12)
        phone = st.selectbox('PhoneService', ['Yes', 'No'])
        multiple_lines = st.selectbox('MultipleLines', ['No', 'Yes', 'No phone service'])
        internet = st.selectbox('InternetService', ['DSL', 'Fiber optic', 'No'])
    with col2:
        online_security = st.selectbox('OnlineSecurity', ['Yes', 'No', 'No internet service'])
        online_backup = st.selectbox('OnlineBackup', ['Yes', 'No', 'No internet service'])
        device_protection = st.selectbox('DeviceProtection', ['Yes', 'No', 'No internet service'])
        tech_support = st.selectbox('TechSupport', ['Yes', 'No', 'No internet service'])
        streaming_tv = st.selectbox('StreamingTV', ['Yes', 'No', 'No internet service'])
        streaming_movies = st.selectbox('StreamingMovies', ['Yes', 'No', 'No internet service'])
        contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])

    paperless = st.selectbox('PaperlessBilling', ['Yes', 'No'])
    payment = st.selectbox('PaymentMethod', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    monthly = st.number_input('MonthlyCharges', min_value=0.0, value=70.0, format="%.2f")
    total = st.number_input('TotalCharges', min_value=0.0, value=float(tenure) * float(monthly), format="%.2f")

    submitted = st.form_submit_button('Generate prediction')

if submitted:
    test_raw = {
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone,
        'MultipleLines': multiple_lines,
        'InternetService': internet,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment,
        'MonthlyCharges': float(monthly),
        'TotalCharges': float(total)
    }

    try:
        label, proba = predict_from_dict(test_raw, model, feature_names, cat_cols)
    except Exception as e:
        st.error(f'Error during prediction: {e}')
    else:
        st.subheader('Prediction')
        st.write('Input:')
        st.json(test_raw)
        if label == 1:
            st.error(f'Predicted churn = {label} (probability {proba:.3f})')
        else:
            st.success(f'Predicted churn = {label} (probability {proba:.3f})')

        st.caption('Note: This is a baseline logistic regression. For production use consider input validation, scaling, and model monitoring.')

# Footer
st.markdown('---')
st.caption('Educational baseline · Built with scikit-learn and Streamlit')
