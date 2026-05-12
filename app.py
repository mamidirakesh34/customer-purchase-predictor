import streamlit as st
import pickle
import numpy as np

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Customer Purchase Prediction",
    layout="wide"
)

# =====================================================
# HEADER SECTION
# =====================================================

col_img, col_title = st.columns([2, 4])

with col_img:
    st.image("customer.png", width=250)

with col_title:
    st.markdown("""
    <h1 style='margin-top:40px;'>
    Customer Purchase Prediction System
    </h1>
    """, unsafe_allow_html=True)

st.markdown("---")


# =====================================================
# LOAD TRAINED MODEL
# =====================================================

with open('model.pkl', 'rb') as f:
    trained_model = pickle.load(f)


# =====================================================
# USER INPUT SECTION
# =====================================================

col1, col2 = st.columns(2)


# ---------------- LEFT SIDE ----------------

with col1:

    st.subheader("Customer Transaction Details")

    quantity = st.number_input(

        "Quantity Purchased",

        min_value=1,

        step=1,

        value=1

    )

    unitprice = st.number_input(

        "Unit Price",

        min_value=1.0,

        step=0.5,

        value=1.0

    )


# ---------------- RIGHT SIDE ----------------

with col2:

    st.subheader("Customer RFM Information")

    recency = st.number_input(

        "Recency (Days)",

        min_value=0,

        step=1,

        value=30

    )

    frequency = st.number_input(

        "Frequency",

        min_value=1,

        step=1,

        value=5

    )


# =====================================================
# INPUT DATA
# =====================================================

input_data = np.array([[

    quantity,

    unitprice,

    recency,

    frequency

]])


# =====================================================
# PREDICTION SECTION
# =====================================================

st.markdown("---")

if st.button("Predict Purchase Behavior"):

    try:

        # Probability Prediction

        probability = trained_model.predict_proba(
            input_data
        )[0][1]

        st.subheader("Prediction Result")

        st.write(
            f"Purchase Probability: {round(probability, 2)}"
        )

        # Custom Manual Conditions

        if (

            quantity >= 10

            and unitprice >= 100

            and recency <= 30

            and frequency >= 10

        ):

            st.success(
                "Customer is likely to purchase again"
            )

        else:

            st.error(
                "Customer is not likely to purchase again"
            )

    except Exception as e:

        st.error(f"Error: {e}")