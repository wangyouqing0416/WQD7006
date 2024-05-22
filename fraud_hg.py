import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
import xgboost as xgb  # XGBoost Classifier
import pickle

# Load the Random Forest model
with open("rf_model.pkl", "rb") as f:
     loaded_model = pickle.load(f)

# https://icons.getbootstrap.com/ for icons

def streamlit_menu(example=1, options=["Home", "Contact"], icons=["coin", "bar-chart"]):
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",  # required
                options=options,  # required
                icons=icons,  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
            )
        return selected

    if example == 2:
        # 2. horizontal menu w/o custom style
        selected = option_menu(
            menu_title=None,  # required
            options=options,  # required
            icons=icons,  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
        return selected

    if example == 3:
        # 3. horizontal menu with custom style
        selected = option_menu(
            menu_title=None,  # required
            options=options,  # required
            icons=icons,  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "green"},
            },
        )
        return selected

fraud_df = pd.read_csv('https://raw.githubusercontent.com/data-cracker/datasets/main/fraud_oracle.csv')
fraud_df = fraud_df[fraud_df['Age'] != 0]

# Streamlit UI
# Set page configuration
st.set_page_config(layout="wide")

# 1 = sidebar menu, 2 = horizontal menu, 3 = horizontal menu w/ custom menu
selected = streamlit_menu(example = 1, 
                          options=["About", "Exploratory Data Analysis", "Predictive Model", "Source Codes"],
                          icons=["house", "bar-chart-fill", "bar-chart-steps", "file-earmark-medical-fill"])

if selected == "About":
    # Title of the page
    st.markdown("<h2>Vehicle Insurance Fraud Detection Using Machine Learning</h2>", unsafe_allow_html=True)

    # Background section
    st.markdown("<h3>Background</h3>", unsafe_allow_html=True)
    st.markdown("""
    Fraud in the insurance sector is a serious problem that can cost insurance firms a large amount of loss in revenue and potentially drive up the premiums for honest policyholders to cover it. The Association of British Insurers states that insurers must pay for the expense of looking into possible frauds, implying that honest clients will pay greater rates. Spending time in investigations of fraudulent insurance claims also hinders their capacity to respond promptly to legitimate claims. Moreover, insurance fraud is also known to put people's lives in danger by funding and enabling more serious crimes including money laundering and, in certain situations, staging car accidents.
    """)

    # Problem Statement section
    st.markdown("<h3>Problem Statement</h3>", unsafe_allow_html=True)
    st.markdown("""
    Traditional methods of fraud detection, such as manual review and rule-based systems, have limitations in effectively identifying fraudulent behavior due to their reliance on predetermined rules and human judgment. Traditional ways may overlook subtle patterns and evolving fraud tactics.

    To address these challenges, the application of machine learning in insurance fraud detection has gained its importance. Machine learning algorithms can analyze large volumes of data, identify complex patterns, and make predictions based on historical claim data and other relevant factors. By leveraging advanced analytics techniques, machine learning models can detect anomalies, outliers, and suspicious patterns indicative of fraudulent behavior more accurately and efficiently than traditional methods.
    """)

    st.markdown("<h3>About this dataset</h3>", unsafe_allow_html=True)
    st.write("This dataset contains vehicle dataset - attribute, model, accident details, etc along with policy details - policy type, tenure etc. The target is to detect if a claim application is fraudulent or not - FraudFound_P")
    st.markdown('''
    **Dataset Description**  

    | #    | Column               | Description                                           |
    |------|----------------------|-------------------------------------------------------|
    | 1    | Month                | Month of car accident occurrence                      |
    | 2    | WeekOfMonth          | Week number of car accident                           |
    | 3    | DayOfWeek            | Day of car accident                                   |
    | 4    | Make                 | Vehicle make or model                                 |
    | 5    | AccidentArea         | Area of car accident occurrence                                  |
    | 6    | DayOfWeekClaimed     | Day of insurance claim                                |
    | 7    | MonthClaimed         | Month of insurance claim                              |
    | 8    | WeekOfMonthClaimed   | Week number of insurance claim                               |
    | 9    | Sex                  | Gender of policyholder                                |
    | 10   | MaritalStatus        | Marital status of policyholder                        |
    | 11   | Age                  | Age of policyholder                                   |
    | 12   | Fault                | Fault attribution in accident                         |
    | 13   | PolicyType           | Type of insurance policy                              |
    | 14   | VehicleCategory      | Vehicle category                                      |
    | 15   | VehiclePrice         | Price range of vehicle                                |
    | 16   | FraudFound_P         | Fraud indicator (0 = Not Fraud, 1 = Fraud)            |
    | 17   | PolicyNumber         | Insurance policy number                               |
    | 18   | RepNumber            | Representative number                                 |
    | 19   | Deductible           | Insurance deductible amount                           |
    | 20   | DriverRating         | Driver rating                                         |
    | 21   | Days_Policy_Accident | Days since policy start to accident                   |
    | 22   | Days_Policy_Claim    | Days since policy start to claim                      |
    | 23   | PastNumberOfClaims   | Previous number of claims                             |
    | 24   | AgeOfVehicle         | Age of vehicle                                        |
    | 25   | AgeOfPolicyHolder    | Age range of policyholder                             |
    | 26   | PoliceReportFiled    | Police report filed indicator                         |
    | 27   | WitnessPresent       | Witness present indicator                             |
    | 28   | AgentType            | Type of insurance agent                               |
    | 29   | NumberOfSuppliments  | Number of supplementary claims                        |
    | 30   | AddressChange_Claim  | Address change before claim                           |
    | 31   | NumberOfCars         | Number of cars owned                                  |
    | 32   | Year                 | Year of incident                                      |
    | 33   | BasePolicy           | Base insurance policy type                            |
    ''')

if selected == "Exploratory Data Analysis":
    # Title of the page
    st.markdown("<h2>Visualisation Dashboard</h2>", unsafe_allow_html=True)

    # Map labels
    labels = {0: 'Not A Fraud', 1: 'A Fraud'}
    fraud_df['Fraud Indicator'] = fraud_df['FraudFound_P'].map(labels)

    # Create the pie chart for 'Fraud Indicator'
    fig_fraud = px.pie(fraud_df, names='Fraud Indicator', title='Vehicle Fraud Claim Distribution', width=600, height=400)

    # Update traces to show both count and percentage, and use arrows for labels
    fig_fraud.update_traces(textposition='outside', textinfo='label+percent',
                            insidetextorientation='radial',
                            texttemplate='%{label}<br>Count: %{value}<br>Percentage: %{percent}',
                            pull=[0.1, 0])  # Pull the slices apart slightly for emphasis

    # Create the pie chart for 'AccidentArea'
    fig_accident_area = px.pie(fraud_df, names='AccidentArea', title='Accident Area Distribution', width=600, height=400)

    # Update traces to show both count and percentage, and use arrows for labels
    fig_accident_area.update_traces(textposition='outside', textinfo='label+percent',
                                    insidetextorientation='radial',
                                    texttemplate='%{label}<br>Count: %{value}<br>Percentage: %{percent}',
                                    pull=[0.1, 0])  # Pull the slices apart slightly for emphasis

    # Create box plot for 'Age' by Fraud Indicator
    fig_age = px.box(fraud_df, x='Fraud Indicator', y='Age', title='Age Distribution by Fraud Indicator')

    # Create box plot for 'DriverRating' by Fraud Indicator
    fig_driver_rating = px.box(fraud_df, x='Fraud Indicator', y='DriverRating', title='Driver Rating Distribution by Fraud Indicator')

    # Display the pie charts side by side
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig_fraud)

    with col2:
        st.plotly_chart(fig_accident_area)

    # Display the box plots side by side
    col3, col4 = st.columns(2)

    with col3:
        st.plotly_chart(fig_age)

    with col4:
        st.plotly_chart(fig_driver_rating)

    # Aggregate the data for plotting
    agg_df = fraud_df.groupby(['Fraud Indicator', 'VehicleCategory']).size().reset_index(name='Count')

    # Calculate percentages
    total_counts = agg_df.groupby('Fraud Indicator')['Count'].transform('sum')
    agg_df['Percentage'] = (agg_df['Count'] / total_counts) * 100

    # Create a horizontal bar chart of frequency of each fraud indicator in the insurance claim, broken down by vehicle category
    fig = px.bar(
        agg_df,
        y='Fraud Indicator',
        x='Count',
        color='VehicleCategory',
        barmode='group',
        labels={'x': 'Count', 'y': 'Fraud Indicator'},
        title='Fraud Indicator Distribution by Vehicle Category',
        orientation='h',
        text=agg_df.apply(lambda row: f"{row['Count']} ({row['Percentage']:.1f}%)", axis=1)
    )

    # Customize the appearance of the bar chart
    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_title='Count', yaxis_title='Fraud Indicator', legend_title='Vehicle Category')

    st.plotly_chart(fig)

    st.dataframe(fraud_df, width=1800, height=500)

def encode():
    fraud_df[['Month']] = fraud_df[['Month']].replace({m: i for i, m in enumerate(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])})
    fraud_df[['DayOfWeek']] = fraud_df[['DayOfWeek']].replace({d: i for i, d in enumerate(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])})
    fraud_df[['Make']] = fraud_df[['Make']].replace({m: i for i, m in enumerate(['Lexus', 'Ferrari', 'Mecedes', 'Porche', 'Jaguar', 'BMW', 'Nisson', 'Saturn', 'Mercury', 'Dodge', 'Saab', 'VW', 'Ford', 'Accura', 'Chevrolet', 'Mazda', 'Honda', 'Toyota', 'Pontiac'])})
    fraud_df[['AccidentArea']] = fraud_df[['AccidentArea']].replace({'Rural': 0, 'Urban': 1})
    fraud_df[['DayOfWeekClaimed']] = fraud_df[['DayOfWeekClaimed']].replace({d: i for i, d in enumerate(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])})
    fraud_df[['MonthClaimed']] = fraud_df[['MonthClaimed']].replace({m: i for i, m in enumerate(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])})
    fraud_df[['Sex']] = fraud_df[['Sex']].replace({'Female': 0, 'Male': 1})
    fraud_df[['MaritalStatus']] = fraud_df[['MaritalStatus']].replace({'Widow': 0, 'Divorced': 1, 'Single': 2, 'Married': 3})
    fraud_df[['Fault']] = fraud_df[['Fault']].replace({'Third Party': 0, 'Policy Holder': 1})
    fraud_df[['PolicyType']] = fraud_df[['PolicyType']].replace({p: i for i, p in enumerate(['Sport - Liability', 'Sport - All Perils', 'Utility - Liability', 'Utility - Collision', 'Utility - All Perils', 'Sport - Collision', 'Sedan - All Perils', 'Sedan - Liability', 'Sedan - Collision'])})
    fraud_df[['VehicleCategory']] = fraud_df[['VehicleCategory']].replace({'Utility': 0, 'Sport': 1, 'Sedan': 2})
    fraud_df[['VehiclePrice']] = fraud_df[['VehiclePrice']].replace({p: i for i, p in enumerate(['less than 20000', '20000 to 29000', '30000 to 39000', '40000 to 59000', '60000 to 69000', 'more than 69000'])})
    fraud_df[['Days_Policy_Accident']] = fraud_df[['Days_Policy_Accident']].replace({d: i for i, d in enumerate(['none', '1 to 7', '8 to 15', '15 to 30', 'more than 30'])})
    fraud_df[['Days_Policy_Claim']] = fraud_df[['Days_Policy_Claim']].replace({d: i for i, d in enumerate(['8 to 15', '15 to 30', 'more than 30'])})
    fraud_df[['PastNumberOfClaims']] = fraud_df[['PastNumberOfClaims']].replace({c: i for i, c in enumerate(['none', '1', '2 to 4', 'more than 4'])})
    fraud_df[['AgeOfVehicle']] = fraud_df[['AgeOfVehicle']].replace({a: i for i, a in enumerate(['new', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', 'more than 7'])})
    fraud_df[['AgeOfPolicyHolder']] = fraud_df[['AgeOfPolicyHolder']].replace({a: i for i, a in enumerate(['18 to 20', '21 to 25', '26 to 30', '31 to 35', '36 to 40', '41 to 50', '51 to 65', 'over 65'])})
    fraud_df[['PoliceReportFiled']] = fraud_df[['PoliceReportFiled']].replace({'Yes': 0, 'No': 1})
    fraud_df[['WitnessPresent']] = fraud_df[['WitnessPresent']].replace({'Yes': 0, 'No': 1})
    fraud_df[['AgentType']] = fraud_df[['AgentType']].replace({'Internal': 0, 'External': 1})
    fraud_df[['NumberOfSuppliments']] = fraud_df[['NumberOfSuppliments']].replace({s: i for i, s in enumerate(['none', '1 to 2', '3 to 5', 'more than 5'])})
    fraud_df[['AddressChange_Claim']] = fraud_df[['AddressChange_Claim']].replace({a: i for i, a in enumerate(['no change', 'under 6 months', '1 year', '2 to 3 years', '4 to 8 years'])})
    fraud_df[['NumberOfCars']] = fraud_df[['NumberOfCars']].replace({c: i for i, c in enumerate(['1 vehicle', '2 vehicles', '3 to 4', '5 to 8', 'more than 8'])})
    fraud_df[['BasePolicy']] = fraud_df[['BasePolicy']].replace({b: i for i, b in enumerate(['All Perils', 'Liability', 'Collision'])})

encode()

month_mapping = {'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'Jun': 5, 'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11}
day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
make_mapping = {'Lexus': 0, 'Ferrari': 1, 'Mecedes': 2, 'Porche': 3, 'Jaguar': 4, 'BMW': 5, 'Nisson': 6, 'Saturn': 7, 'Mercury': 8, 'Dodge': 9,
                'Saab': 10, 'VW': 11, 'Ford': 12, 'Accura': 13, 'Chevrolet': 14, 'Mazda': 15, 'Honda': 16, 'Toyota': 17, 'Pontiac': 18}
area_mapping = {'Rural': 0, 'Urban': 1}
sex_mapping = {'Female': 0, 'Male': 1}
marital_status_mapping = {'Widow': 0, 'Divorced': 1, 'Single': 2, 'Married': 3}
fault_mapping = {'Third Party': 0, 'Policy Holder': 1}
policy_type_mapping = {'Sport - Liability': 0, 'Sport - All Perils': 1, 'Utility - Liability': 2, 'Utility - Collision': 3, 'Utility - All Perils': 4,
                    'Sport - Collision': 5, 'Sedan - All Perils': 6, 'Sedan - Liability': 7, 'Sedan - Collision': 8}
vehicle_category_mapping = {'Utility': 0, 'Sport': 1, 'Sedan': 2}
vehicle_price_mapping = {'less than 20000': 0, '20000 to 29000': 1, '30000 to 39000': 2, '40000 to 59000': 3, '60000 to 69000': 4, 'more than 69000': 5}
policy_accident_mapping = {'none': 0, '1 to 7': 1, '8 to 15': 2, '15 to 30': 3, 'more than 30': 4}
policy_claim_mapping = {'8 to 15': 0, '15 to 30': 1, 'more than 30': 2}
claims_mapping = {'none': 0, '1': 1, '2 to 4': 2, 'more than 4': 3}
vehicle_age_mapping = {'new': 0, '2 years': 1, '3 years': 2, '4 years': 3, '5 years': 4, '6 years': 5, '7 years': 6, 'more than 7': 7}
policyholder_age_mapping = {'18 to 20': 0, '21 to 25': 1, '26 to 30': 2, '31 to 35': 3, '36 to 40': 4, '41 to 50': 5, '51 to 65': 6, 'over 65': 7}
report_mapping = {'Yes': 0, 'No': 1}
witness_mapping = {'Yes': 0, 'No': 1}
agent_type_mapping = {'Internal': 0, 'External': 1}
supplements_mapping = {'none': 0, '1 to 2': 1, '3 to 5': 2, 'more than 5': 3}
address_change_mapping = {'no change': 0, 'under 6 months': 1, '1 year': 2, '2 to 3 years': 3, '4 to 8 years': 4}
cars_mapping = {'1 vehicle': 0, '2 vehicles': 1, '3 to 4': 2, '5 to 8': 3, 'more than 8': 4}
base_policy_mapping = {'All Perils': 0, 'Liability': 1, 'Collision': 2}

def predict_note_authentication(make,accidentarea,sex,maritalstatus,age,policytype,vehiclecategory,vehicleprice,deductible,
                                driverrating,pastnumberofclaims,ageofvehicle,policereportfiled,witnesspresent,agenttype,
                                numberofsuppliments,addresschange_claim,numberofcars,basepolicy):
    inputs = [make, accidentarea, sex,maritalstatus, age,policytype,vehiclecategory, vehicleprice, deductible,
              driverrating,pastnumberofclaims, ageofvehicle, policereportfiled,witnesspresent, agenttype,
               numberofsuppliments, addresschange_claim,numberofcars, basepolicy]
    encoded_inputs = [
        make_mapping.get(make, make), 
        area_mapping.get(accidentarea, accidentarea), sex_mapping.get(sex, sex), marital_status_mapping.get(maritalstatus, maritalstatus), age, 
        policy_type_mapping.get(policytype,policytype), vehicle_category_mapping.get(vehiclecategory, vehiclecategory),
        vehicle_price_mapping.get(vehicleprice, vehicleprice),  deductible, driverrating,
        claims_mapping.get(pastnumberofclaims, pastnumberofclaims), vehicle_age_mapping.get(ageofvehicle, ageofvehicle),
        report_mapping.get(policereportfiled, policereportfiled), 
        witness_mapping.get(witnesspresent, witnesspresent), agent_type_mapping.get(agenttype, agenttype), 
        supplements_mapping.get(numberofsuppliments, numberofsuppliments), address_change_mapping.get(addresschange_claim, addresschange_claim),
        cars_mapping.get(numberofcars, numberofcars), base_policy_mapping.get(basepolicy, basepolicy)
    ]
    prediction=loaded_model.predict([encoded_inputs])
    print(prediction)
    return prediction

if selected == "Predictive Model":
    st.title(f" {selected}")
    col1, col2 = st.columns(2)
    with col1:
        #month = st.selectbox("Month", list(month_mapping.keys()))
        #weekofmonth = st.number_input("Week of Month", min_value=1, max_value=5, value=1)
        #dayofweek = st.selectbox("Day of Week", list(day_mapping.keys()))
        make = st.selectbox("Make", list(make_mapping.keys()))
        accidentarea = st.selectbox("Accident Area", list(area_mapping.keys()))
        #dayofweekclaimed = st.selectbox("Day of Week Claimed", list(day_mapping.keys()))
        #monthclaimed = st.selectbox("Month Claimed", list(month_mapping.keys()))
        #weekofmonthclaimed = st.number_input("Week of Month Claimed", min_value=1, max_value=5, value=1)
        sex = st.selectbox("Sex", list(sex_mapping.keys()))
        maritalstatus = st.selectbox("Marital Status", list(marital_status_mapping.keys()))
        age = st.number_input("Age", value=18)
        #fault = st.selectbox("Fault", list(fault_mapping.keys()))
        policytype =st.selectbox("Policy Type", list(policy_type_mapping.keys()))
        vehiclecategory = st.selectbox("Vehicle Category", list(vehicle_category_mapping.keys()))
        vehicleprice = st.selectbox("Vehicle Price", list(vehicle_price_mapping.keys()))
        #policynumber = st.number_input("Policy Number", min_value=0)
        deductible = st.number_input("Deductible", min_value=0)
        driverrating = st.number_input("Driver Rating", min_value=1, max_value=5, value=3)
    with col2:
        #repnumber = st.number_input("Rep Number", min_value=0)
        
        
        #days_policy_accident = st.selectbox("Days Policy Accident", list(policy_accident_mapping.keys()))
        #days_policy_claim = st.selectbox("Days Policy Claim", list(policy_claim_mapping.keys()))
        pastnumberofclaims = st.selectbox("Past Number of Claims", list(claims_mapping.keys()))
        ageofvehicle = st.selectbox("Age of Vehicle", list(vehicle_age_mapping.keys()))
        #ageofpolicyholder = st.selectbox("Age of Policy Holder", list(policyholder_age_mapping.keys()))
        policereportfiled = st.selectbox("Police Report Filed", list(report_mapping.keys()))
        witnesspresent = st.selectbox("Witness Present", list(witness_mapping.keys()))
        agenttype = st.selectbox("Agent Type", list(agent_type_mapping.keys()))
        numberofsuppliments = st.selectbox("Number of Supplements", list(supplements_mapping.keys()))
        addresschange_claim = st.selectbox("Address Change - Claim", list(address_change_mapping.keys()))
        numberofcars = st.selectbox("Number of Cars", list(cars_mapping.keys()))
        #year = st.number_input("Year", min_value=1990, max_value=2024, value=2020)
        basepolicy = st.selectbox("Base Policy", list(base_policy_mapping.keys()))
    if st.button("Predict"):
        result=predict_note_authentication(make,accidentarea,sex,
                                maritalstatus,age,policytype,vehiclecategory,vehicleprice,deductible,
                                driverrating,pastnumberofclaims,ageofvehicle,
                                policereportfiled,witnesspresent,agenttype,numberofsuppliments,addresschange_claim,numberofcars,basepolicy)
        if result == 1:
                st.error('The prediction output is：Opps, there is a risk of fraud.')
        else:
                st.success('The prediction output is：There is no risk of fraud.')

if selected == "Source Codes":
    st.markdown("<h2>Group member</h2>", unsafe_allow_html=True)
    st.write("WANG YOUQING 22099247")
    st.write("Ooi Hian Gee 17203457")
    st.write("Xiaofeng He 22070924")
    st.write("Zhang Feifan 22083811")
    st.write("Yan Chenxue 22100248")
    st.markdown("<h2>Code in colab:</h2>", unsafe_allow_html=True)
    st.write("https://colab.research.google.com/drive/1TnwhDVq9qA4Yi_E5pTONbj-nH-Us8SGo?usp=sharing")
    st.markdown("<h2>Code in github:</h2>", unsafe_allow_html=True)
    st.write("https://github.com/wangyouqing0416/WQD7006")