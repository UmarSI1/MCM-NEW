
############################################################################################################################################################################################################################################################
############################################################################################################################################################################################################################################################   
##############################################################################################################  --- Import's and globals--- ###############################################################################################################
import json
import pickle
import pandas as pd
import streamlit as st
from azure.storage.blob import BlobServiceClient
import time
from datetime import datetime, timedelta
import io
import concurrent.futures
import re
import altair as alt
import random

#importing all the graphs for daily analysis from the respective files for e.g. - general graphs from general.py
from general import *
from general_temp_for_company import *
from general_temp_for_harm import *
from general_temp_for_harm_and_comp import *



#############################################
#Loading secrets
config_file_path = r'C:\Users\Umar.Saad\Downloads\MCM-NEW-master\MCM-NEW-master\secrets\config.json'

# Function to load configuration from the JSON file
def load_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Load configuration
# config = load_config(config_file_path)

# Access configuration values
# connection_string = config.get('connection_string')
# container_name = config.get('container_name')

# Example usage
# print(f"API Key: {connection_string}")
# print(f"Endpoint: {container_name}")
#############################################


#############################################
#Listing specific companies and harms we want
list_of_companies = ['TikTok', 'Pinterest', 'Snapchat', 'LinkedIn', 'X', 'Facebook', 'Instagram','YouTube','Reddit','Bumble']    
list_of_harms = ['STATEMENT_CATEGORY_ILLEGAL_OR_HARMFUL_SPEECH', 'STATEMENT_CATEGORY_SCOPE_OF_PLATFORM_SERVICE', 'STATEMENT_CATEGORY_PROTECTION_OF_MINORS', 'STATEMENT_CATEGORY_VIOLENCE', 'STATEMENT_CATEGORY_PORNOGRAPHY_OR_SEXUALIZED_CONTENT', 
                        'STATEMENT_CATEGORY_DATA_PROTECTION_AND_PRIVACY_VIOLATIONS', 'STATEMENT_CATEGORY_SCAMS_AND_FRAUD', 'STATEMENT_CATEGORY_SELF_HARM', 
                        'STATEMENT_CATEGORY_NEGATIVE_EFFECTS_ON_CIVIC_DISCOURSE_OR_ELECTIONS', 'STATEMENT_CATEGORY_INTELLECTUAL_PROPERTY_INFRINGEMENTS', 
                        'STATEMENT_CATEGORY_UNSAFE_AND_ILLEGAL_PRODUCTS', 'STATEMENT_CATEGORY_NON_CONSENSUAL_BEHAVIOUR', 'STATEMENT_CATEGORY_ANIMAL_WELFARE', 
                        'STATEMENT_CATEGORY_RISK_FOR_PUBLIC_SECURITY']
#############################################


############################################################################################################################################################################################################################################################
############################################################################################################################################################################################################################################################   
##############################################################################################################  --- Connecting to azure / getting datasets --- ###############################################################################################################

# # Azure Blob Storage configuration
connection_string = 'DefaultEndpointsProtocol=https;AccountName=asatrustandsafetycv;AccountKey=HrJteCB33VFGftZQQFcp0AL1oiv6XOYtUD7FHosKK67v6+KLTmYLrQSrEL0Np+ODbZrCUNvvZ2Zd+AStGD1jPw==;EndpointSuffix=core.windows.net'
container_name = 'dsanew'

# connecting to azure and getting all blobs in container
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)
blobs_list = container_client.list_blobs()

#assigning each blob in list to a dataset
unorganised_datasets = [blob.name for blob in blobs_list]
#datasets = [filename for filename in unorganised_datasets if re.match(r'^\d', filename)]

datasets = [
    filename for filename in unorganised_datasets
    if re.match(r'^\d', filename) and 'historical' not in filename.lower()
]

#reversing datasets to get latest dataset first
for i in range(len(datasets)):
    if datasets[i].endswith(".pkl"):
        datasets[i] = datasets[i][:-4]
datasets.reverse()


########################################
#Fetches the seelected dataset from the blob list and donwloads it and filters needed data e.G. (list_of_companies)
def load_data_from_dataset(selected_dataset):
    blob_name = selected_dataset
    blob_client = container_client.get_blob_client(blob_name)
    
    # Download the blob content to bytes
    download_stream = blob_client.download_blob()
    blob_data = download_stream.readall()

    # Convert bytes to a file-like object
    data = pickle.load(io.BytesIO(blob_data))

    # Extract necessary lists
    List_of_companies = list(data.keys())
    harm_dic = data[List_of_companies[0]]
    List_of_harms = list(harm_dic.keys())
    content_dic = harm_dic[List_of_harms[0]]
    List_of_content_type = list(content_dic.keys())
    action_dic = content_dic[List_of_content_type[0]]
    List_of_moderation_action = list(action_dic.keys())
    automation_dic = action_dic[List_of_moderation_action[0]]
    List_of_automation_status = list(automation_dic.keys())
    
    #Returning the necessary lists
    return data, List_of_companies, List_of_harms, List_of_content_type, List_of_moderation_action, List_of_automation_status
########################################

########################################
def load_data(selected_dataset):
    """Load data from the blob storage."""
    blob_name = f"{selected_dataset}.pkl"
    #print(blob_name)
    blob_client = container_client.get_blob_client(blob_name)
    
    # Download the blob content to bytes and load it as a dictionary
    download_stream = blob_client.download_blob()
    blob_data = download_stream.readall()
    
    return pickle.load(io.BytesIO(blob_data))
########################################


########################################
def process_automation_status(automation_status):
    acc_totals_per_harm = 0
    manual_totals_per_harm = 0
    
    for acc, automation_detection in automation_status.items():
        if pd.notna(automation_detection):  # Check if the count is not NaN
            if acc == 'Yes':
                acc_totals_per_harm += automation_detection
            else:
                manual_totals_per_harm += automation_detection
                
    return acc_totals_per_harm, manual_totals_per_harm
########################################



def plot_acc_totals_per_harm_company_harm_historical(data, company_selected, harm_selected):
    """ Sum all numbers for acc per harm and return the results. """
    acc_totals_per_harm =  data[company_selected][harm_selected]['Yes']
    manual_totals_per_harm = data[company_selected][harm_selected]['No']

    return acc_totals_per_harm, manual_totals_per_harm
########################################
def process_data_excel(data, dataset1, dataset2):

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data)

    #(df)

    # Preprocess the DataFrame
    df['decision_visibility'].replace('', pd.NA, inplace=True)
    df['decision_visibility'].fillna('["ACCOUNT MODERATION"]', inplace=True)
    df['application_date'] = pd.to_datetime(df['application_date'], errors='coerce')
    df['content_date'] = pd.to_datetime(df['content_date'], errors='coerce')

    # Extract unique values for each dimension
    List_of_companies = df['platform_name'].unique()
    List_of_harms = df['category'].unique()
    List_of_automation_detection = df['automated_detection'].unique()

    # Iterate through the combinations of the different fields
    for company in List_of_companies:
        company_df = df[df['platform_name'] == company]

        for harm in List_of_harms:
            harm_df = company_df[company_df['category'] == harm]

            for automation_detection in List_of_automation_detection:
                # Count the number of rows matching the current combination of values
                count = (harm_df['automated_detection'] == automation_detection).sum()

                if dataset2[company][harm][automation_detection] is None:
                    dataset2[company][harm][automation_detection] = 0

                # Store the count in the nested dictionary structure
                dataset1[company][harm][automation_detection] += count

                # Calculate the time difference in minutes
                time_difference = (harm_df["application_date"] - harm_df["content_date"]).dt.total_seconds() / 60

                # Calculate the mean of the time differences and convert it to an integer
                mean_minutes = int(time_difference.mean())

                # Update your dataset
                dataset2[company][harm][automation_detection] += mean_minutes

    


########################################
########################################



############################################################################################################################################################################################################################################################
############################################################################################################################################################################################################################################################   
##############################################################################################################  --- Streamlit Page --- #####################################################################################################################

def main():

    st.set_page_config(layout="wide")
    st.write('<h1 style="text-align: center; text-decoration: underline;">Content moderation daily monitor</h1>', unsafe_allow_html=True)
    st.write('<h4 style="text-align: center;">This dashboard presents the daily count of moderation actions categorized by harm and platform provided by the DSA Transparency Database.</h4>', unsafe_allow_html=True)
    st.markdown("---")
    
    
    ##############################################################################################################  --- Harm definition's section --- ###############################################################################################################
    
    with st.expander("Harm definition according to the DSA documentation", expanded=True):
    
        question = st.selectbox(
            "Select a Harm",
            ["Animal welfare", "Data protection and privacy violations", "Illegal or harmful speech", "Intellectual property infringements", "Negative effects on civic discourse or elections", 
             "Non-consensual behaviour", "Online bullying/intimidation", "Pornography or sexualized content", "Protection of minors", "Risk for public security", "Scams and/or fraud", "Self-harm", 
             "Scope of platform service", "Unsafe and/or illegal products", "Violence"])
        
        if question == "Animal welfare":
            st.write("This category includes: Animal harm, Unlawful sale of animals.")
        elif question == "Data protection and privacy violations":
            st.write("This category includes: Biometric data breach, Missing processing ground for data, Right to be forgotten, Data falsification.")
        elif question == "Illegal or harmful speech":
            st.write("This category includes: Defamation, Discrimination, Hate speech.")
        elif question == "Intellectual property infringements":
            st.write("This category includes: Copyright infringement, Design infringement, Geographical indications infringements, Patent infringement, Trade secret infringement, Trademark infringement.")
        elif question == "Negative effects on civic discourse or elections":
            st.write("This category includes: Disinformation, Foreign information manipulation and interference, Misinformation.")
        elif question == "Non-consensual behaviour":
            st.write("This category includes: Non-consensual image sharing, Non-consensual items containing deepfake or similar technology using a third party’s features.")
        elif question == "Online bullying/intimidation":
            st.write("This category includes: Stalking.")
        elif question == "Pornography or sexualized content":
            st.write("This category includes: Adult sexual material, Image-based sexual abuse (excluding content depicting minors).")
        elif question == "Protection of minors":
            st.write("This category includes: Age-specific restrictions concerning minors, Child sexual abuse material, Grooming/sexual enticement of minors, Unsafe challenges.")
        elif question == "Risk for public security":
            st.write("This category includes: Illegal organizations, Risk for environmental damage, Risk for public health, Terrorist content.")
        elif question == "Scams and/or fraud":
            st.write("This category includes: Inauthentic accounts, Inauthentic listings, Inauthentic user reviews, Impersonation or account hijacking, Phishing, Pyramid schemes.")
        elif question == "Self-harm":
            st.write("This category includes: Content promoting eating disorders, Self-mutilation, Suicide.")
        elif question == "Scope of platform service":
            st.write("This category includes: Age-specific restrictions, Geographical requirements, Goods/services not permitted to be offered on the platform, Language requirements, Nudity.")
        elif question == "Unsafe and/or illegal products":
            st.write("This category includes: Insufficient information on traders, Regulated goods and services, Dangerous toys.")
        elif question == "Violence":
            st.write("This category includes: Coordinated harm, Gender-based violence, Human exploitation, Human trafficking, Incitement to violence and/or hatred.")

        
    
############################################################################################################################################################################################################################################################
############################################################################################################################################################################################################################################################   
##############################################################################################################  --- Historical Analysis --- ###############################################################################################################

    st.write('<h2 style="text-align: center; text-decoration: underline;">Historical Analysis</h2>', unsafe_allow_html=True)
    
    #initialising the columns
    date_initial, date_final, company_intial, harm_intial = st.columns(4)
    
    #sort this out do we need data?
    data = [datetime.strptime(d, "%Y-%m-%d") for d in datasets]
    #print("data", data)

    #setting the initial date to a defult of 5 days ago and formatting it to YYYY-MM-DD
    today = datetime.now().date()
    initial_date = today - timedelta(days=20)
    initial_date_str = initial_date.strftime("%Y-%m-%d")


    with date_initial:
        #Filter the datasets to only include dates that are less than or equal to the initial date for the initial date input and the same but opposite for final input
        filtered_dates_for_initial_date_input = [date for date in datasets if date <= initial_date_str]
       # print("DFFII", filtered_dates_for_initial_date_input)
        st.markdown("<h4 style=' text-decoration: underline;'>Select an inital date:</h4>", unsafe_allow_html=True) 
        date_initial = datetime.strptime(st.selectbox("Choose a date from the dropdown below:",filtered_dates_for_initial_date_input, index=filtered_dates_for_initial_date_input.index(initial_date_str) if initial_date_str in filtered_dates_for_initial_date_input else 0), "%Y-%m-%d")
        
    with date_final:
        filtered_dates_for_final_date_input = [date for date in datasets if date > initial_date_str]
      #  print("FDFFI", filtered_dates_for_final_date_input) 
        st.markdown("<h4 style=' text-decoration: underline;'>Select a final date:</h4>", unsafe_allow_html=True)
        date_final = datetime.strptime(st.selectbox("Choose a final date from the dropdown below:",[date for date in filtered_dates_for_final_date_input]), "%Y-%m-%d")
    
    with company_intial:
        st.markdown("<h4 style=' text-decoration: underline;'>Select a Company:</h4>", unsafe_allow_html=True)
        company_selected = st.selectbox("Choose a Company from the dropdown below:",list_of_companies)  

    with harm_intial:
        st.markdown("<h4 style=' text-decoration: underline;'>Select a Specific Harm:</h4>", unsafe_allow_html=True)
        harm_selected = st.selectbox("Choose a Harm from the dropdown below:",list_of_harms)
      
      
    ##############################################################################################################  --- Historical data GRAPHS CREATION --- ###############################################################################################################
    
    #getting all the dates between the user chosen initial date and final date
    all_dates_between_initial_final_dates = [(date_initial + timedelta(days=i)).strftime("%Y-%m-%d")  for i in range((date_final - date_initial).days + 1) if (date_initial + timedelta(days=i)).strftime("%Y-%m-%d") in datasets]

   # print("all_dates_between_initial_final_dates", all_dates_between_initial_final_dates)


  
  
    ####################################################
    #Load the data from all pkl files with the dates between initial & final dates
    start_time = time.time()
    # with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
    #      datasets_loaded = list(executor.map(load_data, all_dates_between_initial_final_dates))   

    def append_historical1(date):
        return f"{date}_historical1"
    
    def append_historical2(date):
        return f"{date}_historical2"

    with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
        # First append "_historical1" to each date
        updated_dates = list(executor.map(append_historical1, all_dates_between_initial_final_dates))
        
        # Then load the datasets with the modified dates
        datasets_loaded = list(executor.map(load_data, updated_dates))
        #datasets_loaded = list(executor.map(load_data, all_dates_between_initial_final_dates))

        

    with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
        # First append "_historical1" to each date
        updated_dates2 = list(executor.map(append_historical2, all_dates_between_initial_final_dates))
        
        # Then load the datasets with the modified dates
        datasets_loaded2 = list(executor.map(load_data, updated_dates2))
             
    end_time = time.time()
    duration = end_time - start_time
   # print(f"Function load_data ran in {duration:.4f} seconds")
    ####################################################

    ####################################################
    # Process the loaded data above
    start_time = time.time()
    def process_data(data):
        return plot_acc_totals_per_harm_company_harm_historical(data, company_selected, harm_selected)
    
    # Using ThreadPoolExecutor to parallelize the function calls
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_data, datasets_loaded))
        results2 = list(executor.map(process_data, datasets_loaded2))
        

        
    end_time = time.time()
    duration = end_time - start_time
   # print(f"Function (plot_acc_totals_per_harm_company_harm_historical) ran in {duration:.4f} seconds")
    ####################################################


    ####################################################
    #GRAPH #1
    df = pd.DataFrame({
        'Dates': all_dates_between_initial_final_dates,
        'Automated': [result[0] for result in results],
        'Manual': [result[1] for result in results]
    })
    # Melt the dataframe to a long format for Altair
    df_long = df.melt('Dates', var_name='Type', value_name='DAILY FLAGGED CONTENT')
    # Create an Altair chart
    chart = alt.Chart(df_long).mark_line().encode(
        x='Dates',
        y='DAILY FLAGGED CONTENT',
        color=alt.Color('Type', scale=alt.Scale(domain=['Automated', 'Manual'], range=['red', 'green'])),
        strokeDash='Type').properties(
        title='ACC Flag count vs User Flag count')
    
    
    ####################################################



    ####################################################
    #GRAPH #2
    acc_total = df['Automated'].sum()
    user_total = df["Manual"].sum()
    # Create a DataFrame with the totals
    data_a = {'Category': ['Automated', 'Manual'],'Total Harm Count': [acc_total, user_total]}
    df_xx = pd.DataFrame(data_a)
    color_scale = alt.Scale(domain=['Automated', 'Manual'], range=['red', 'green'])
    # Create an Altair bar chart
    chart_two = alt.Chart(df_xx).mark_bar().encode(
        x=alt.X('Category:N', title=''),
        y=alt.Y('Total Harm Count:Q', title='TOTAL FLAGGED CONTENT'),
        color=alt.Color('Category:N', scale=color_scale, legend=None)).properties(width=alt.Step(80))
    

    ####################################################


    # ####################################################
    #GRAPH #3
    df_three = pd.DataFrame({
        'Dates': all_dates_between_initial_final_dates,
        'Automated': [result[0] for result in results2],
        'Manual': [result[1] for result in results2]
    })

    # Ensure there are no division by zero errors
    df_three['Automated'] = df_three['Automated'] / df['Automated'].replace(0, pd.NA)
    df_three['Manual'] = df_three['Manual'] / df['Manual'].replace(0, pd.NA)

    # If you want to avoid NaN values after division, you can use a fill method:
    df_three['Automated'] = df_three['Automated'].fillna(0)
    df_three['Manual'] = df_three['Manual'].fillna(0)


    # Melt the dataframe to a long format for Altair
    df_long = df_three.melt('Dates', var_name='Type', value_name='MODERATION TIME (MINS)')
    # Create an Altair chart
    chart_three = alt.Chart(df_long).mark_line().encode(
        x='Dates',
        y='MODERATION TIME (MINS)',
        color=alt.Color('Type', scale=alt.Scale(domain=['Automated', 'Manual'], range=['red', 'green'])),
        strokeDash='Type').properties(
        title='ACC VS Manual Moderation Time')


    # ####################################################
    #graph4
    # acc_total4 = df_three['Automated'].mean()
    # user_total4 = df_three["Manual"].mean()

    acc_total4 = df_three['Automated'][df_three['Automated'] != 0].mean()
    user_total4 = df_three['Manual'][df_three['Manual'] != 0].mean()

    if pd.isna(acc_total4):
        acc_total4 = 0

    if pd.isna(user_total4):
        user_total4 = 0



    
    # Create a DataFrame with the totals
    data_a = {'Category': ['Automated', 'Manual'],'Total Harm Count': [int(acc_total4), int(user_total4)]}
    df_xx = pd.DataFrame(data_a)
    color_scale = alt.Scale(domain=['Automated', 'Manual'], range=['red', 'green'])
    # Create an Altair bar chart
    chart_four = alt.Chart(df_xx).mark_bar().encode(
        x=alt.X('Category:N', title=''),
        y=alt.Y('Total Harm Count:Q', title='AVERAGE  MODERATION TIME (MINS)'),
        color=alt.Color('Category:N', scale=color_scale, legend=None)).properties(width=alt.Step(80))


    
    
    # ####################################################
    #Plotting the graphs made above
    col1, col2 = st.columns(2)
    
    with col1:
        st.altair_chart(chart, use_container_width=True)
    with col2:
            formatted_number1 = format(int(acc_total), ",")
            formatted_number2 = format(int(user_total), ",")
            
            st.write(
                f"<span style='color:red; font-weight:bold;'>Automated</span> (ACC): {formatted_number1} ┃ "
                f"<span style='color:green; font-weight:bold;'>Manual</span> (User reported): {formatted_number2}",
                unsafe_allow_html=True)
            
            st.altair_chart(chart_two, use_container_width=True)


    #Plotting the graphs made above
    col1, col2 = st.columns(2)
    
    with col1:
        st.altair_chart(chart_three, use_container_width=True)

    with col2:
            # Display the chart
            formatted_number1 = format(int(acc_total4), ",")
            formatted_number2 = format(int(user_total4), ",")
            st.write(
                f"<span style='color:red; font-weight:bold;'>Automated Average Time</span>: {formatted_number1} Mins "
                f"<span style='color:green; font-weight:bold;'>Manual Average Time</span>: {formatted_number2} Mins ",
                unsafe_allow_html=True)
            st.altair_chart(chart_four, use_container_width=True)
    st.markdown("---")
    ####################################################
    
























    
    
############################################################################################################################################################################################################################################################
############################################################################################################################################################################################################################################################
##############################################################################################################  --- DAILY LIVE ANALYSIS  --- ###############################################################################################################

    ####################################################
    # Columns, Titles and dataset selectbox
    st.write('<h2 style="text-align: center; text-decoration: underline;">Daily Live Analysis</h2>', unsafe_allow_html=True)
    st.markdown("<h3 style='text-decoration: underline;'>Select a Specific Date</h3>", unsafe_allow_html=True)
    general_data_col, company_col, harm_col = st.columns(3)
    selected_dataset = st.selectbox("Choose a Date:", datasets)
    ####################################################


    ####################################################
    # Load data and extract lists from the selected dataset
    if selected_dataset:
        data, List_of_companies, List_of_harms, List_of_content_type, List_of_moderation_action, List_of_automation_status = load_data_from_dataset(selected_dataset + ".pkl")
        #load_data_from_dataset(selected_dataset + ".pkl")
    ####################################################


    ####################################################
    #making the user selection columns
    with general_data_col:
        st.markdown("<h3 style='text-decoration: underline;'>Overall Info for All Companies</h3>", unsafe_allow_html=True)
        selected_option_gen = st.checkbox("General Data")
        disable_others = selected_option_gen  # Disable other options if general data is selected

    with company_col:
        st.markdown("<h3 style='text-decoration: underline;'>Select a Specific Company</h3>", unsafe_allow_html=True)
        selected_company = st.selectbox("Choose a Company:", [None] + List_of_companies, disabled=disable_others)

    with harm_col:
        st.markdown("<h3 style='text-decoration: underline;'>Select a Specific Harm</h3>", unsafe_allow_html=True)
        selected_harm = st.selectbox("Choose a Harm:", [None] + List_of_harms, disabled=disable_others)
    ####################################################


    ########################################################################################################
    if selected_option_gen:
        st.markdown("---")
        st.subheader("Analysis for General Overview")
        col1, col2 = st.columns(2)
   
        fig1 = plot_acc_totals_per_company(data)
        fig2 = plot_acc_totals_per_harm(data)
        fig3 = plot_acc_totals_per_moderation_action(data)
        fig4 = plot_acc_totals_per_automation_status(data)
        fig5 = plot_acc_totals_per_content_type(data)
        fig6 = plot_acc_totals(data)
        fig7 = sum_harm(data)
        fig8 = plot_company_dataxxz(data, List_of_companies)
        fig9 = plot_company_dataxxz_normalized(data, List_of_companies)
        fig10 = plot_content_type_totals(data)
        fig11 = plot_moderation_action_totals(data)
        fig12 = plot_automation_status_totals(data)
        fig13 = plot_harm_totals_per_company(data)
        fig14 = plot_content_type_totals_per_company(data)
        fig15 = plot_automation_status_table_general(data)
        fig16 = plot_normalized_automation_status(data)
        fig17 = plot_harm_content_type(data)
        fig18 = plot_harm_content_type_normalized(data)
        fig19 = plot_harm_automation_status(data)
        fig20 = plot_harm_automation_status_two(data)
        fig21 = plot_content_type_automation_status(data)
        fig22 = plot_content_type_automation_status_two(data)
        fig23 = sum_reports_per_harm_per_moderation_action(data)
        fig24 = generate_moderation_action_automation_status_figure(data)
        fig25 = sum_reports_per_moderation_action_per_company(data)

        with col1:
            with st.expander("Total ACC detections per moderation action", expanded=False):
                st.dataframe(fig3, use_container_width=True)

                csv = fig3.to_csv(index=False)
    
                # Add a download button
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='moderation_data.csv',
                    mime='text/csv',
                )
        with col2:
            with st.expander("Total ACC detections per automation decision status", expanded=False):
                st.dataframe(fig4, use_container_width=True)
        with col1:
            with st.expander("Total ACC detections per content type", expanded=False):
                st.dataframe(fig5, use_container_width=True)
        with col2:
            with st.expander("Total ACC detections per company", expanded=False):
                st.dataframe(fig1, use_container_width=True)
        with col1:
            with st.expander("Total ACC detections per harm", expanded=False):
                st.dataframe(fig2, use_container_width=True)
        with col2:
            with st.expander("Total count for manual vs automated detection", expanded=False):
                st.dataframe(fig6, use_container_width=True)
        with col1:
            with st.expander("Total number of Moderation actions per harm", expanded=False):
                st.dataframe(fig7, use_container_width=True)
        with col2:
            with st.expander("Total number of Moderation Actions per Company", expanded=False):
                st.dataframe(fig8, use_container_width=True)          
        with col1:
            with st.expander("Total number of Moderation Actions per Company Normalized", expanded=False):
                st.dataframe(fig9, use_container_width=True)     
        with col2:
            with st.expander("Total number of Moderation Actions per Type of Content", expanded=False):
                st.dataframe(fig10, use_container_width=True)
        with col1:
            with st.expander("Total number of Moderation Actions per Type of Automation Status", expanded=False):
                st.dataframe(fig12, use_container_width=True)
        with col2:
            with st.expander("Total number of Moderation Actions per Type of Moderation Decision", expanded=False):
                st.dataframe(fig11, use_container_width=True)
        with col1:
            with st.expander("Number of reported Harms per Company", expanded=False):
                st.dataframe(fig13, use_container_width=True)
        with col2:
            with st.expander("Number of reported content type per Company", expanded=False):
                st.dataframe(fig14, use_container_width=True)
        with col1:
            with st.expander("Normalized counts of each automation status per company", expanded=False):
                st.dataframe(fig16, use_container_width=True)
        with col2:
            with st.expander("Number of reported content type per Harm", expanded=False):
                st.dataframe(fig17, use_container_width=True)
        with col1:
            with st.expander("Number of reported content type per Harm Normalized", expanded=False):
                st.dataframe(fig18, use_container_width=True)               
        with col2:
            with st.expander("Count for each harm per automation status", expanded=False):
                st.dataframe(fig19, use_container_width=True)
        with col1:
            with st.expander("Count for each harm per automation status normalized", expanded=False):
                st.dataframe(fig20, use_container_width=True)
        with col2:
            with st.expander("Count of each Harm per Moderation decision", expanded=False):
                st.dataframe(fig23, use_container_width=True)
        with col1:
            with st.expander("Count for each content type per automation status", expanded=False):
                st.dataframe(fig21, use_container_width=True)
        with col2:
            with st.expander("Count for each content type per automation status Normalized", expanded=False):
                st.dataframe(fig22, use_container_width=True)
        with col1:
            with st.expander("Number of reported moderation decision per company", expanded=False):
                st.dataframe(fig25, use_container_width=True)
                
    ########################################################################################################


    ########################################################################################################
    elif selected_company and selected_harm:
        st.markdown("---")
        st.subheader(f"Analysis for {selected_company} and {selected_harm}")
        col1, col2 = st.columns(2)


        fig1 = plot_acc_totals_per_company_company_harm(data, selected_company, selected_harm)
        fig2 = plot_acc_totals_per_harm_company_harm(data, selected_company, selected_harm)
        fig3 = plot_acc_totals_per_moderation_action_company_harm(data, selected_company, selected_harm)
        fig4 = plot_acc_totals_per_automation_status_company_harm(data, selected_company, selected_harm)
        fig5 = plot_acc_totals_per_content_type_company_harm(data, selected_company, selected_harm)
        fig6 = plot_acc_totals_company_harm(data, selected_company, selected_harm)
        fig7 = sum_harm3(data, selected_company, selected_harm)
        fig8 = plot_company_dataxxz3(data, selected_company, selected_harm)
        fig9 = plot_company_dataxxz3_normalized(data, selected_company, selected_harm)
        fig10 = plot_content_type_totals3(data, selected_company, selected_harm)
        fig11 = plot_moderation_action_totals3(data, selected_company, selected_harm)
        fig12 = plot_automation_status_totals3(data, selected_company, selected_harm)
        fig13 = plot_harm_totals_per_company3(data, selected_company, selected_harm)
        fig14 = plot_content_type_totals_per_company3(data, selected_company, selected_harm)
        fig15 = plot_automation_status_table_general3(data, selected_company, selected_harm)
        fig16 = plot_normalized_automation_status3(data, selected_company, selected_harm)
        fig17 = plot_harm_content_type3_normalized(data, selected_company, selected_harm)
        fig18 = plot_harm_content_type_normalized3(data, selected_company, selected_harm)
        fig19 = plot_harm_automation_status3(data, selected_company, selected_harm)
        fig20 = plot_harm_automation_status3_normalized(data, selected_company, selected_harm)
        fig21 = plot_content_type_automation_status3(data, selected_company, selected_harm)
        fig22 = plot_content_type_automation_status3_normalized(data, selected_company, selected_harm)
        fig23 = generate_moderation_action_automation_status_figure3(data, selected_company, selected_harm)
        fig24 = sum_reports_per_moderation_action_per_company3(data, selected_company, selected_harm)

        with col1:
            with st.expander("Total ACC detections per moderation action", expanded=False):
                st.dataframe(fig3, use_container_width=True)
        with col2:
            with st.expander("Total ACC detections per automation decision status", expanded=False):
                st.dataframe(fig4, use_container_width=True)
        with col1:
            with st.expander("Total ACC detections per content type", expanded=False):
                st.dataframe(fig5, use_container_width=True)
        with col2:
            with st.expander("Total ACC detections per company", expanded=False):
                st.dataframe(fig1, use_container_width=True)
        with col1:
            with st.expander("Total ACC detections per harm", expanded=False):
                st.dataframe(fig2, use_container_width=True)
        with col2:
            with st.expander("Total count for manual vs automated detection", expanded=False):
                st.dataframe(fig6, use_container_width=True)
        with col1:
            with st.expander("Total number of Moderation actions for selected harm and company", expanded=False):
                st.dataframe(fig7, use_container_width=True)
        with col2:
            with st.expander("Total number of Moderation Actions normalized for selected harm and company", expanded=False):
                st.dataframe(fig9, use_container_width=True)
        with col1:
            with st.expander("Total number of Moderation Actions per Type of Content for selected harm and company", expanded=False):
                st.dataframe(fig10, use_container_width=True)
        with col2:
            with st.expander("Total number of Moderation Actions per Type of moderation action for selected harm and company", expanded=False):
                st.dataframe(fig11, use_container_width=True)
        with col1:
            with st.expander("Total number of Moderation Actions per Type of Moderation Decision for selected harm and company", expanded=False):
                st.dataframe(fig12, use_container_width=True)
        with col2:
            with st.expander("Number of reported Harms for selected harm and company", expanded=False):
                st.dataframe(fig13, use_container_width=True)
        with col1:
            with st.expander("Number of reported content type for selected harm and company", expanded=False):
                st.dataframe(fig14, use_container_width=True)
        with col2:
            with st.expander("Number of Automation Status type for selected harm and company", expanded=False):
                st.dataframe(fig15, use_container_width=True)
        with col1:
            with st.expander("Normalized counts of each automation status for selected harm and company", expanded=False):
                st.dataframe(fig16, use_container_width=True)
        with col2:
            with st.expander("Number of reported content type normalized for selected harm and company", expanded=False):
                st.dataframe(fig17, use_container_width=True)
        with col1:
            with st.expander("Count for each harm per automation status normalized for selected harm and company", expanded=False):
                st.dataframe(fig20, use_container_width=True)
        with col2:
            with st.expander("Count for each content type per automation status for selected harm and company", expanded=False):
                st.dataframe(fig21, use_container_width=True)
        with col1:
            with st.expander("Count for each content type per automation status normalized for selected harm and company", expanded=False):
                st.dataframe(fig22, use_container_width=True)
        with col2:
            with st.expander("Count of moderation decision per automation status for selected harm and company", expanded=False):
                st.dataframe(fig23, use_container_width=True)
        with col1:
            with st.expander("Number of reported moderation decision for selected harm and company", expanded=False):
                st.dataframe(fig24, use_container_width=True)
    ########################################################################################################
 
    ########################################################################################################
    elif selected_company:
        st.markdown("---")
        st.subheader(f"Analysis for {selected_company}")
        col1, col2 = st.columns(2)

        fig1 = plot_acc_totals_per_company_company(data, selected_company)
        fig2 = plot_acc_totals_per_harm_company(data, selected_company)
        fig3 = plot_acc_totals_per_moderation_action_company(data, selected_company)
        fig4 = plot_acc_totals_per_automation_status_company(data, selected_company)
        fig5 = plot_acc_totals_per_content_type_company(data, selected_company)
        fig6 = plot_acc_totals_company(data, selected_company)
        fig7 = sum_harm1(data, selected_company)
        fig8 = plot_company_dataxxz1(data, selected_company)
        fig9 = plot_company_dataxxz1_normalized(data, selected_company)
        fig10 = plot_content_type_totals1(data, selected_company)
        fig11 = plot_moderation_action_totals1(data, selected_company)
        fig12 = plot_automation_status_totals1(data, selected_company)
        fig13 = plot_harm_totals_per_company1(data, selected_company)
        fig14 = plot_content_type_totals_per_company1(data, selected_company)
        fig15 = plot_automation_status_table_general1(data, selected_company)
        fig16 = plot_normalized_automation_status1(data, selected_company)
        fig17 = plot_harm_content_type_1(data, selected_company)
        fig18 = plot_harm_content_type_normalized1(data, selected_company)
        fig19 = plot_harm_automation_status1(data, selected_company)
        fig20 = plot_harm_automation_status1_normalized(data, selected_company)
        fig21 = plot_content_type_automation_status1(data, selected_company)
        fig22 = plot_content_type_automation_status1_normalized(data, selected_company)
        fig23 = generate_moderation_action_automation_status_figure1(data, selected_company)
        fig24 = sum_reports_per_moderation_action_per_company1(data, selected_company)


        with col1:
            with st.expander("Total ACC detections per moderation action", expanded=False):
                st.dataframe(fig3, use_container_width=True)
        with col2:
            with st.expander("Total ACC detections per automation decision status", expanded=False):
                st.dataframe(fig4, use_container_width=True)
        with col1:
            with st.expander("Total ACC detections per content type", expanded=False):
                st.dataframe(fig5, use_container_width=True)
        with col2:
            with st.expander("Total ACC detections per company", expanded=False):
                st.dataframe(fig1, use_container_width=True)
        with col1:
            with st.expander("Total ACC detections per harm", expanded=False):
                st.dataframe(fig2, use_container_width=True)
        with col2:
            with st.expander("Total count for manual vs automated detection", expanded=False):
                st.dataframe(fig6, use_container_width=True)
        with col1:
            with st.expander("Total number of Moderation actions per harm", expanded=False):
                st.dataframe(fig7, use_container_width=True)
        with col2:
            with st.expander("Total number of Moderation Actions per Type of Automation Status", expanded=False):
                st.dataframe(fig12, use_container_width=True)
        with col1:
            with st.expander("Total number of Moderation Actions per Company", expanded=False):
                st.dataframe(fig8, use_container_width=True)
        with col2:
            with st.expander("Total number of Moderation Actions per Company normalized", expanded=False):
                st.dataframe(fig9, use_container_width=True)
        with col1:
            with st.expander("Total number of Moderation Actions per Type of Moderation Decision", expanded=False):
                st.dataframe(fig11, use_container_width=True)
        with col2:
            with st.expander("Number of reported Harms per Company", expanded=False):
                st.dataframe(fig13, use_container_width=True)
        with col1:
            with st.expander("Number of reported content type per Company", expanded=False):
                st.dataframe(fig14, use_container_width=True)
        with col2:
            with st.expander("Number of Automation Status type per Company", expanded=False):
                st.dataframe(fig15, use_container_width=True)
        with col1:
            with st.expander("Normalized counts of each automation status per company", expanded=False):
                st.dataframe(fig16, use_container_width=True)
        with col2:
            with st.expander("Count for each harm per content type", expanded=False):
                st.dataframe(fig17, use_container_width=True)
        with col1:
            with st.expander("Count for each harm per content type Normalized", expanded=False):
                st.dataframe(fig18, use_container_width=True)
        with col2:
            with st.expander("Count for each harm per automation status", expanded=False):
                st.dataframe(fig19, use_container_width=True)
        with col1:
            with st.expander("Count for each harm per automation status normalized", expanded=False):
                st.dataframe(fig20, use_container_width=True)
        with col2:
            with st.expander("Count for each content type per automation status", expanded=False):
                st.dataframe(fig21, use_container_width=True)
        with col1:
            with st.expander("Count for each content type per automation status Normalized", expanded=False):
                st.dataframe(fig22, use_container_width=True)
        with col2:
            with st.expander("Count of moderation decision per automation status", expanded=False):
                st.dataframe(fig23, use_container_width=True)
        with col1:
            with st.expander("Number of reported moderation decision per company", expanded=False):
                st.dataframe(fig24, use_container_width=True)
    ########################################################################################################

    ########################################################################################################
    elif selected_harm:
        st.markdown("---")
        st.subheader(f"Analysis for {selected_harm}")
        col1, col2 = st.columns(2)

        fig1 = plot_acc_totals_per_company_harm(data, selected_harm)
        fig2 = plot_acc_totals_per_harm_harm(data, selected_harm)
        fig3 = plot_acc_totals_per_moderation_action_harm(data, selected_harm)
        fig4 = plot_acc_totals_per_automation_status_harm(data, selected_harm)
        fig5 = plot_acc_totals_per_content_type_harm(data, selected_harm)
        fig6 = plot_acc_totals_harm(data, selected_harm)
        fig7 = sum_harm2(data, selected_harm)
        fig8 = plot_content_type_totals2(data, selected_harm)
        fig9 = plot_moderation_action_totals2(data, selected_harm)
        fig10 = plot_automation_status_totals2(data, selected_harm)
        fig11 = plot_harm_totals_per_company2(data, selected_harm)
        fig12 = plot_automation_status_table_general2(data, selected_harm)
        fig13 = plot_normalized_automation_status2(data, selected_harm)
        fig14 = plot_harm_content_type_normalized2(data, selected_harm)
        fig15 = plot_harm_automation_status2(data, selected_harm)
        fig16 = plot_harm_automation_status2_normalized(data, selected_harm)
        fig17 = plot_content_type_automation_status2(data, selected_harm)
        fig18 = plot_content_type_automation_status2_normalized(data, selected_harm)
        fig19 = generate_moderation_action_automation_status_figure2(data, selected_harm)

        with col1:
            with st.expander("Total ACC detections per moderation action", expanded=False):
                st.dataframe(fig3, use_container_width=True)
        with col2:
            with st.expander("Total ACC detections per automation decision status", expanded=False):
                st.dataframe(fig4, use_container_width=True)
        with col1:
            with st.expander("Total ACC detections per content type", expanded=False):
                st.dataframe(fig5, use_container_width=True)
        with col2:
            st.dataframe(fig1, use_container_width=True)
        with col1:
            with st.expander("Total ACC detections per harm", expanded=False):
                st.dataframe(fig2, use_container_width=True)
        with col2:
            with st.expander("Total count for manual vs automated detection", expanded=False):
                st.dataframe(fig6, use_container_width=True)
        with col1:
            with st.expander("Total number of Moderation actions per harm", expanded=False):
                st.dataframe(fig7, use_container_width=True)
        with col2:
            with st.expander("Total number of Moderation Actions per Type of Content for harm", expanded=False):
                st.dataframe(fig8, use_container_width=True)
        with col1:
            with st.expander("Total number of Moderation Actions per Type of Automation Status for harm", expanded=False):
                st.dataframe(fig9, use_container_width=True)
        with col2:
            with st.expander("Total number of Automation status for harm", expanded=False):
                st.dataframe(fig10, use_container_width=True)
        with col1:
            with st.expander("Number of reported Harms per Company for harm", expanded=False):
                st.dataframe(fig11, use_container_width=True)
        with col2:
            with st.expander("Number of Automation Status type per Company for harm", expanded=False):
                st.dataframe(fig12, use_container_width=True)
        with col1:
            with st.expander("Normalized counts of each automation status per company for harm", expanded=False):
                st.dataframe(fig13, use_container_width=True)
        with col2:
            with st.expander("Number of reported content type per Harm Normalized for harm", expanded=False):
                st.dataframe(fig14, use_container_width=True)
        with col1:
            with st.expander("Count for each harm per automation status for harm", expanded=False):
                st.dataframe(fig15, use_container_width=True)
        with col2:
            with st.expander("Count for each harm per automation status normalized for harm", expanded=False):
                st.dataframe(fig16, use_container_width=True)
        with col1:
            with st.expander("Count for each content type per automation status for harm", expanded=False):
                st.dataframe(fig17, use_container_width=True)
        with col2:
            with st.expander("Count for each content type per automation status normalized for harm", expanded=False):
                st.dataframe(fig18, use_container_width=True)
        with col1:
            with st.expander("Count of moderation decision per automation status for harm", expanded=False):
                st.dataframe(fig19, use_container_width=True)
            

    else:
        st.write("No dataset selected.")
    ########################################################################################################

########################################################################################################
if __name__ == "__main__":
    main()
########################################################################################################
