"""
Color using in the streamlit app
Background  : #e8f4f8  (soft sky blue)
Cards       : #ffffff  (white)
Primary     : #7ec8e3  (baby blue)
Accent      : #b8d4e8  (light blue)
Text        : #4a6fa5  (dark blue)
Pink touch  : #f9c6d0  (soft pink)

Full code structure 
imports
Image.open()
set_page_config()
CSS
load data
sidebar + page = st.radio()

if page == " Overview":
    st.title(...)
    st.write(...)
    st.dataframe(...)

elif page == "EDA":
    st.title(...)

elif page == "Hypothesis Test":
    st.title(...)

elif page == "ML Model":
    st.title(...)

elif page == "Predict Tumor":
    st.title(...)
"""