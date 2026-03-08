import streamlit as st
import pandas as pd
import numpy as np
import joblib
import streamlit.components.v1 as components

# -----------------------------
# Load trained model files
# -----------------------------
model = joblib.load("attrition_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model_columns = joblib.load("model_columns.pkl")

# -----------------------------
# App Title
# -----------------------------
st.title("🤖 AI Employee Attrition Prediction System")
st.write("Enter employee details to predict attrition risk")

# -----------------------------
# User Inputs
# -----------------------------
age = st.number_input("Age", 18, 60)

business_travel = st.selectbox(
    "Business Travel",
    ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
)

department = st.selectbox(
    "Department",
    [
        "Data Science",
        "Network Administration",
        "Cyber Security",
        "IT Services",
        "Software Development"
    ]
)

distance = st.number_input("Distance From Home (in km)", 0, 50)

education = st.selectbox(
    "Education",
    [1,2,3,4,5],
    format_func=lambda x:{
        1:"1 - Below College",
        2:"2 - Degree",
        3:"3 - Graduation",
        4:"4 - Master's",
        5:"5 - PhD"
    }[x]
)

env_sat = st.selectbox(
    "Environment Satisfaction (1 = Low, 5 = High)",
    [1,2,3,4,5]
)

gender = st.selectbox(
    "Gender",
    ["Male","Female","Other"]
)

salary = st.number_input("Monthly Income", 10000, 200000)

job_involvement = st.selectbox(
    "Job Involvement (1–5)",
    [1,2,3,4,5]
)

job_level = st.selectbox(
    "Job Level (1 = Lowest, 8 = Highest)",
    [1,2,3,4,5]
)

job_role = st.selectbox(
    "Job Role",
    [
        "QA Analyst",
        "Technician",
        "Manager",
        "Director",
        "Software Engineer",
        "Help Desk",
        "Developer",
        "Consultant",
        "HR",
        "IT",
        "Business Analyst",
        "Support"
    ]
)

job_satisfaction = st.selectbox(
    "Job Satisfaction (1–5)",
    [1,2,3,4,5]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

companies_worked = st.number_input("Number of Companies Worked", 0, 10)

overtime = st.selectbox("Overtime", ["Yes", "No"])

salary_hike = st.number_input("Percent Salary Hike", 0, 50)

total_exp = st.number_input("Total Working Years", 0, 40)

work_life = st.selectbox(
    "Work Life Balance (1–5)",
    [1,2,3,4,5]
)

years_company = st.number_input("Years At Company", 0, 40)

years_role = st.number_input("Years In Current Role", 0, 20)

years_promo = st.number_input("Years Since Last Promotion", 0, 15)

feedback = st.text_area("Employee Feedback (Required)")

# -----------------------------
# Feature Engineering (internal)
# -----------------------------
ExperienceRatio = years_company / (total_exp + 1)
PromotionGap = years_company - years_promo
IncomePerLevel = salary / job_level if job_level != 0 else 0
CompanySwitchRate = companies_worked / total_exp if total_exp != 0 else 0
RoleStability = years_role / years_company if years_company != 0 else 0
PromotionDelay = total_exp - years_promo
LowIncomeFlag = 1 if salary < 30000 else 0

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Attrition"):

    # -----------------------------
    # Input Validation
    # -----------------------------
    if feedback.strip() == "":
        st.error("Employee Feedback is required. Please enter feedback.")
        st.stop()

    if salary <= 0:
        st.error("Monthly Income must be greater than 0")
        st.stop()

    if total_exp < years_company:
        st.error("Years At Company cannot be greater than Total Working Years")
        st.stop()

    if years_role > years_company:
        st.error("Years In Current Role cannot exceed Years At Company")
        st.stop()

    # -----------------------------
    # Create input dataframe
    # -----------------------------
    input_data = pd.DataFrame({
        'Age':[age],
        'BusinessTravel':[business_travel],
        'Department':[department],
        'DistanceFromHome':[distance],
        'Education':[education],
        'EnvironmentSatisfaction':[env_sat],
        'Gender':[gender],
        'MonthlyIncome':[salary],
        'JobInvolvement':[job_involvement],
        'JobLevel':[job_level],
        'JobRole':[job_role],
        'JobSatisfaction':[job_satisfaction],
        'MaritalStatus':[marital_status],
        'NumCompaniesWorked':[companies_worked],
        'OverTime':[overtime],
        'PercentSalaryHike':[salary_hike],
        'TotalWorkingYears':[total_exp],
        'WorkLifeBalance':[work_life],
        'YearsAtCompany':[years_company],
        'YearsInCurrentRole':[years_role],
        'YearsSinceLastPromotion':[years_promo],
        'ExperienceRatio':[ExperienceRatio],
        'PromotionGap':[PromotionGap],
        'IncomePerLevel':[IncomePerLevel],
        'CompanySwitchRate':[CompanySwitchRate],
        'RoleStability':[RoleStability],
        'PromotionDelay':[PromotionDelay],
        'LowIncomeFlag':[LowIncomeFlag]
    })

    # Encode categorical fields
    input_encoded = pd.get_dummies(input_data)

    # TF-IDF vector for feedback
    feedback_vector = vectorizer.transform([feedback if feedback else ""])
    feedback_df = pd.DataFrame(feedback_vector.toarray())

    # Merge structured + feedback features
    final_input = pd.concat([input_encoded, feedback_df], axis=1)
    final_input = final_input.reindex(columns=model_columns, fill_value=0)

    # Predict probability
    prediction = model.predict_proba(final_input)[0][1]

    # -----------------------------
    # Display Result
    # -----------------------------
    st.subheader("Prediction Result")

    prediction_percent = prediction * 100
    st.metric("Attrition Probability", f"{prediction_percent:.2f}%")

    if prediction_percent >= 70:
        st.error("🚨 High Attrition Risk")
    elif prediction_percent >= 30:
        st.warning("⚠️ Medium Attrition Risk")
    else:
        st.success("✅ Low Attrition Risk")

    # Possible reasons
    st.subheader("Possible Reasons:")
    reasons = []
    if overtime=="Yes":
        reasons.append("Frequent Overtime")
    if job_satisfaction <= 2:
        reasons.append("Low Job Satisfaction")
    if env_sat <= 2:
        reasons.append("Poor Work Environment")
    if distance > 20:
        reasons.append("Long Commute")
    if salary < 30000:
        reasons.append("Low Salary")
    if years_promo > 5:
        reasons.append("No promotion for long time")

    if not reasons:
        st.write("No strong attrition indicators detected.")
    else:
        for r in reasons:
            st.write("•", r)

# -----------------------------
# Power BI Dashboard
# -----------------------------
st.title("Power BI Dashboard")

powerbi_iframe = """
<iframe title="MH 2026 copy - Copy"
width="100%"
height="450"
src="https://app.powerbi.com/view?r=eyJrIjoiYzE4YmRhMDgtNGJhNC00ZTE0LThiOGItNDI0NDg2YTg5NWNjIiwidCI6IjQwZjkzODFiLWViNzEtNDlhMi1iMjVhLWU3MDBkNDgxZDVjMSJ9"
frameborder="0"
allowFullScreen="true">
</iframe>
"""

components.html(powerbi_iframe, height=450)








# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import streamlit.components.v1 as components

# # -----------------------------
# # Load trained model files
# # -----------------------------
# model = joblib.load("attrition_model.pkl")
# vectorizer = joblib.load("tfidf_vectorizer.pkl")
# model_columns = joblib.load("model_columns.pkl")

# # -----------------------------
# # App Title
# # -----------------------------
# st.title("🤖 AI Employee Attrition Prediction System")
# st.write("Enter employee details to predict attrition risk")

# # -----------------------------
# # User Inputs
# # -----------------------------
# age = st.number_input("Age", 18, 60)

# business_travel = st.selectbox(
#     "Business Travel",
#     ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
# )

# department = st.selectbox(
#     "Department",
#     ["Data Science", "Cyber Security", "IT Services"]
# )

# distance = st.number_input("Distance From Home (in km)", 0, 50)

# education = st.selectbox("Education Level (1-5)", [1, 2, 3, 4, 5])

# env_sat = st.selectbox("Environment Satisfaction (1-4)", [1, 2, 3, 4])

# gender = st.selectbox("Gender", ["Male", "Female"])

# salary = st.number_input("Monthly Income", 10000, 200000)

# job_involvement = st.selectbox("Job Involvement (1-4)", [1, 2, 3, 4])

# job_level = st.selectbox("Job Level (1-5)", [1, 2, 3, 4, 5])

# job_role = st.selectbox(
#     "Job Role",
#     ["Developer", "QA Analyst", "Technician", "Consultant"]
# )

# job_satisfaction = st.selectbox("Job Satisfaction (1-4)", [1, 2, 3, 4])

# marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

# companies_worked = st.number_input("Number of Companies Worked", 0, 10)

# overtime = st.selectbox("Overtime", ["Yes", "No"])

# salary_hike = st.number_input("Percent Salary Hike", 0, 50)

# total_exp = st.number_input("Total Working Years", 0, 40)

# work_life = st.selectbox("Work Life Balance (1-4)", [1, 2, 3, 4])

# years_company = st.number_input("Years At Company", 0, 40)

# years_role = st.number_input("Years In Current Role", 0, 20)

# years_promo = st.number_input("Years Since Last Promotion", 0, 15)

# feedback = st.text_area("Employee Feedback (Required)")

# # -----------------------------
# # Feature Engineering (internal)
# # -----------------------------
# ExperienceRatio = years_company / (total_exp + 1)
# PromotionGap = years_company - years_promo
# IncomePerLevel = salary / job_level if job_level != 0 else 0
# CompanySwitchRate = companies_worked / total_exp if total_exp != 0 else 0
# RoleStability = years_role / years_company if years_company != 0 else 0
# PromotionDelay = total_exp - years_promo
# LowIncomeFlag = 1 if salary < 30000 else 0

# # -----------------------------
# # Prediction
# # -----------------------------
# if st.button("Predict Attrition"):

#     # -----------------------------
#     # Input Validation
#     # -----------------------------
#     if feedback.strip() == "":
#         st.error("Employee Feedback is required. Please enter feedback.")
#         st.stop()

#     if salary <= 0:
#         st.error("Monthly Income must be greater than 0")
#         st.stop()

#     if total_exp < years_company:
#         st.error("Years At Company cannot be greater than Total Working Years")
#         st.stop()

#     if years_role > years_company:
#         st.error("Years In Current Role cannot exceed Years At Company")
#         st.stop()

#     # -----------------------------
#     # Create input dataframe
#     # -----------------------------
#     input_data = pd.DataFrame({
#         'Age':[age],
#         'BusinessTravel':[business_travel],
#         'Department':[department],
#         'DistanceFromHome':[distance],
#         'Education':[education],
#         'EnvironmentSatisfaction':[env_sat],
#         'Gender':[gender],
#         'MonthlyIncome':[salary],
#         'JobInvolvement':[job_involvement],
#         'JobLevel':[job_level],
#         'JobRole':[job_role],
#         'JobSatisfaction':[job_satisfaction],
#         'MaritalStatus':[marital_status],
#         'NumCompaniesWorked':[companies_worked],
#         'OverTime':[overtime],
#         'PercentSalaryHike':[salary_hike],
#         'TotalWorkingYears':[total_exp],
#         'WorkLifeBalance':[work_life],
#         'YearsAtCompany':[years_company],
#         'YearsInCurrentRole':[years_role],
#         'YearsSinceLastPromotion':[years_promo],
#         'ExperienceRatio':[ExperienceRatio],
#         'PromotionGap':[PromotionGap],
#         'IncomePerLevel':[IncomePerLevel],
#         'CompanySwitchRate':[CompanySwitchRate],
#         'RoleStability':[RoleStability],
#         'PromotionDelay':[PromotionDelay],
#         'LowIncomeFlag':[LowIncomeFlag]
#     })

#     # Encode categorical fields
#     input_encoded = pd.get_dummies(input_data)

#     # TF-IDF vector for feedback
#     feedback_vector = vectorizer.transform([feedback if feedback else ""])
#     feedback_df = pd.DataFrame(feedback_vector.toarray())

#     # Merge structured + feedback features
#     final_input = pd.concat([input_encoded, feedback_df], axis=1)
#     final_input = final_input.reindex(columns=model_columns, fill_value=0)

#     # Predict probability
#     prediction = model.predict_proba(final_input)[0][1]

#     # -----------------------------
#     # Display Result
#     # -----------------------------
#     st.subheader("Prediction Result")
#     st.write("Attrition Probability:", round(prediction*100,2), "%")

#     if prediction > 0.55:
#         st.error("High Attrition Risk")
#     elif prediction > 0.15:
#         st.warning("Medium Attrition Risk")
#     else:
#         st.success("Low Attrition Risk")

#     # Optional: show reasons
#     st.subheader("Possible Reasons:")
#     reasons = []
#     if overtime=="Yes":
#         reasons.append("Frequent Overtime")
#     if job_satisfaction <= 2:
#         reasons.append("Low Job Satisfaction")
#     if env_sat <= 2:
#         reasons.append("Poor Work Environment")
#     if distance > 20:
#         reasons.append("Long Commute")
#     if salary < 30000:
#         reasons.append("Low Salary")
#     if years_promo > 5:
#         reasons.append("No promotion for long time")

#     if not reasons:
#         st.write("No strong attrition indicators detected.")
#     else:
#         for r in reasons:
#             st.write("•", r)


# # -----------------------------
# # Power BI Dashboard
# # -----------------------------
# st.title("Power BI Dashboard")

# powerbi_iframe = """
# <iframe title="MH 2026 copy - Copy"
# width="100%"
# height="450"
# src="https://app.powerbi.com/view?r=eyJrIjoiYzE4YmRhMDgtNGJhNC00ZTE0LThiOGItNDI0NDg2YTg5NWNjIiwidCI6IjQwZjkzODFiLWViNzEtNDlhMi1iMjVhLWU3MDBkNDgxZDVjMSJ9"
# frameborder="0"
# allowFullScreen="true">
# </iframe>
# """

# components.html(powerbi_iframe, height=450)






# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import streamlit as st
# import streamlit.components.v1 as components

# # -----------------------------
# # Load trained model files
# # -----------------------------
# model = joblib.load("attrition_model.pkl")
# vectorizer = joblib.load("tfidf_vectorizer.pkl")
# model_columns = joblib.load("model_columns.pkl")

# # -----------------------------
# # App Title
# # -----------------------------
# st.title("🤖 AI Employee Attrition Prediction System")
# st.write("Enter employee details to predict attrition risk")

# # -----------------------------
# # User Inputs
# # -----------------------------
# age = st.number_input("Age", 18, 60)

# business_travel = st.selectbox(
#     "Business Travel",
#     ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
# )

# department = st.selectbox(
#     "Department",
#     ["Data Science", "Cyber Security", "IT Services"]
# )

# distance = st.number_input("Distance From Home (in km)", 0, 50)

# education = st.selectbox("Education Level (1-5)", [1, 2, 3, 4, 5])

# env_sat = st.selectbox("Environment Satisfaction (1-4)", [1, 2, 3, 4])

# gender = st.selectbox("Gender", ["Male", "Female"])

# salary = st.number_input("Monthly Income", 10000, 200000)

# job_involvement = st.selectbox("Job Involvement (1-4)", [1, 2, 3, 4])

# job_level = st.selectbox("Job Level (1-5)", [1, 2, 3, 4, 5])

# job_role = st.selectbox(
#     "Job Role",
#     ["Developer", "QA Analyst", "Technician", "Consultant"]
# )

# job_satisfaction = st.selectbox("Job Satisfaction (1-4)", [1, 2, 3, 4])

# marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

# companies_worked = st.number_input("Number of Companies Worked", 0, 10)

# overtime = st.selectbox("Overtime", ["Yes", "No"])

# salary_hike = st.number_input("Percent Salary Hike", 0, 50)

# total_exp = st.number_input("Total Working Years", 0, 40)

# work_life = st.selectbox("Work Life Balance (1-4)", [1, 2, 3, 4])

# years_company = st.number_input("Years At Company", 0, 40)

# years_role = st.number_input("Years In Current Role", 0, 20)

# years_promo = st.number_input("Years Since Last Promotion", 0, 15)

# feedback = st.text_area("Employee Feedback (optional)")

# # -----------------------------
# # Feature Engineering (internal)
# # -----------------------------
# ExperienceRatio = years_company / (total_exp + 1)
# PromotionGap = years_company - years_promo
# IncomePerLevel = salary / job_level if job_level != 0 else 0
# CompanySwitchRate = companies_worked / total_exp if total_exp != 0 else 0
# RoleStability = years_role / years_company if years_company != 0 else 0
# PromotionDelay = total_exp - years_promo
# LowIncomeFlag = 1 if salary < 30000 else 0

# # -----------------------------
# # Prediction
# # -----------------------------
# if st.button("Predict Attrition"):

#     # Create input dataframe
#     input_data = pd.DataFrame({
#         'Age':[age],
#         'BusinessTravel':[business_travel],
#         'Department':[department],
#         'DistanceFromHome':[distance],
#         'Education':[education],
#         'EnvironmentSatisfaction':[env_sat],
#         'Gender':[gender],
#         'MonthlyIncome':[salary],
#         'JobInvolvement':[job_involvement],
#         'JobLevel':[job_level],
#         'JobRole':[job_role],
#         'JobSatisfaction':[job_satisfaction],
#         'MaritalStatus':[marital_status],
#         'NumCompaniesWorked':[companies_worked],
#         'OverTime':[overtime],
#         'PercentSalaryHike':[salary_hike],
#         'TotalWorkingYears':[total_exp],
#         'WorkLifeBalance':[work_life],
#         'YearsAtCompany':[years_company],
#         'YearsInCurrentRole':[years_role],
#         'YearsSinceLastPromotion':[years_promo],
#         'ExperienceRatio':[ExperienceRatio],
#         'PromotionGap':[PromotionGap],
#         'IncomePerLevel':[IncomePerLevel],
#         'CompanySwitchRate':[CompanySwitchRate],
#         'RoleStability':[RoleStability],
#         'PromotionDelay':[PromotionDelay],
#         'LowIncomeFlag':[LowIncomeFlag]
#     })

#     # Encode categorical fields
#     input_encoded = pd.get_dummies(input_data)

#     # TF-IDF vector for feedback
#     feedback_vector = vectorizer.transform([feedback if feedback else ""])
#     feedback_df = pd.DataFrame(feedback_vector.toarray())

#     # Merge structured + feedback features
#     final_input = pd.concat([input_encoded, feedback_df], axis=1)
#     final_input = final_input.reindex(columns=model_columns, fill_value=0)

#     # Predict probability
#     prediction = model.predict_proba(final_input)[0][1]

#     # -----------------------------
#     # Display Result
#     # -----------------------------
#     st.subheader("Prediction Result")
#     st.write("Attrition Probability:", round(prediction*100,2), "%")

#     if prediction > 0.55:
#         st.error("High Attrition Risk")
#     elif prediction > 0.15:
#         st.warning("Medium Attrition Risk")
#     else:
#         st.success("Low Attrition Risk")

#     # Optional: show reasons
#     st.subheader("Possible Reasons:")
#     reasons = []
#     if overtime=="Yes":
#         reasons.append("Frequent Overtime")
#     if job_satisfaction <= 2:
#         reasons.append("Low Job Satisfaction")
#     if env_sat <= 2:
#         reasons.append("Poor Work Environment")
#     if distance > 20:
#         reasons.append("Long Commute")
#     if salary < 30000:
#         reasons.append("Low Salary")
#     if years_promo > 5:
#         reasons.append("No promotion for long time")
#     if not reasons:
#         st.write("No strong attrition indicators detected.")
#     else:
#         for r in reasons:
#             st.write("•", r)



# st.title("Power BI Dashboard")

# powerbi_iframe = """
# <iframe title="MH 2026 copy - Copy"
# width="100%"
# height="450"
# src="https://app.powerbi.com/view?r=eyJrIjoiYzE4YmRhMDgtNGJhNC00ZTE0LThiOGItNDI0NDg2YTg5NWNjIiwidCI6IjQwZjkzODFiLWViNzEtNDlhMi1iMjVhLWU3MDBkNDgxZDVjMSJ9"
# frameborder="0"
# allowFullScreen="true">
# </iframe>
# """

# components.html(powerbi_iframe, height=450)