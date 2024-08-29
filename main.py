import pickle
import zipfile
from pathlib import Path
from collections import defaultdict
import os
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from azure.storage.blob import BlobServiceClient, BlobClient, ContentSettings
import requests
import os
import time
from tqdm import tqdm
from datetime import datetime, timedelta
import io
import zipfile
from azure.storage.blob import BlobServiceClient, ContentSettings  # Import ContentSettings
from tqdm import tqdm
import os
from io import BytesIO
import pandas as pd
from collections import defaultdict
import pickle
import concurrent.futures
import re
import altair as alt
import numpy as np


from general import *
from general_temp_for_company import *
from general_temp_for_harm import *
from general_temp_for_harm_and_comp import *

#st.set_option('deprecation.showPyplotGlobalUse', False)


# Azure Blob Storage configuration
connection_string = 'DefaultEndpointsProtocol=https;AccountName=asatrustandsafetycv;AccountKey=HrJteCB33VFGftZQQFcp0AL1oiv6XOYtUD7FHosKK67v6+KLTmYLrQSrEL0Np+ODbZrCUNvvZ2Zd+AStGD1jPw==;EndpointSuffix=core.windows.net'
container_name = 'dsa'

# Define global variables to hold loaded data
data_ACC = None
List_of_companies = []
List_of_harms = []
List_of_content_type = []
List_of_moderation_action = []
List_of_automation_status = []

# Initialize the BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Get the container client
container_client = blob_service_client.get_container_client(container_name)

# List all blobs in the container
blobs_list = container_client.list_blobs()

# Create a list with blob names
datasets_orig = [blob.name for blob in blobs_list]
datasets = [filename for filename in datasets_orig if re.match(r'^\d', filename)]


# Function to load data from selected dataset
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

    return data, List_of_companies, List_of_harms, List_of_content_type, List_of_moderation_action, List_of_automation_status

def load_data(selected_dataset):
    """Load data from the blob storage."""
    blob_name = f"{selected_dataset}.pkl"
    print("BLOB NAME: ", blob_name)
    blob_client = container_client.get_blob_client(blob_name)
    
    # Download the blob content to bytes and load it as a dictionary
    download_stream = blob_client.download_blob()
    blob_data = download_stream.readall()
    
    return pickle.load(io.BytesIO(blob_data))


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


def plot_acc_totals_per_harm_company_harm_historical(data, company_selected, harm_selected):
    """ Sum all numbers for acc per harm and return the results. """
    acc_totals_per_harm = 0
    manual_totals_per_harm = 0

    if company_selected in data:
        harm_dic = data[company_selected]
        if harm_selected in harm_dic:
            # Collect all automation_status dictionaries for parallel processing
            automation_statuses = []
            for content_type_data in harm_dic[harm_selected].values():
                for moderation_action in content_type_data.values():
                    for automation_status in moderation_action.values():
                        automation_statuses.append(automation_status)
            
            # Process automation statuses in parallel
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = list(executor.map(process_automation_status, automation_statuses))
                
            # Aggregate results from all threads/processes
            for acc_total, manual_total in results:
                acc_totals_per_harm += acc_total
                manual_totals_per_harm += manual_total

    return acc_totals_per_harm, manual_totals_per_harm
################################################################################################################

def main():

    for i in range(len(datasets)):
        if datasets[i].endswith(".pkl"):
            datasets[i] = datasets[i][:-4]

    datasets.reverse()


    st.set_page_config(layout="wide")
    st.write('<h1 style="text-align: center; text-decoration: underline;">Content moderation daily monitor</h1>', unsafe_allow_html=True)

    st.markdown(
    '<p style="color:green; font-size:17px; text-align:center;">THIS IS A PROOF OF CONCEPT IDEA</p>',
    unsafe_allow_html=True
)
    
    st.write('<h4 style="text-align: center;">This dashboard presents the daily count of moderation actions categorized by harm and platform provided by the DSA Transparency Database.</h4>', unsafe_allow_html=True)
    #st.write('<h6 style="text-align: center;">If any issues arise please contact either Umar (umar.saad@ofcom.org.uk) Or Pedro Freire (pedro.freire@ofcom.org.uk)</h6>', unsafe_allow_html=True)

    st.markdown("---")
    # Main category 1
    with st.expander("Harm definition according to the DSA documentation", expanded=True):
        question = st.selectbox(
            "Select a Harm",
            ["Animal welfare", "Data protection and privacy violations", "Illegal or harmful speech", "Intellectual property infringements", "Negative effects on civic discourse or elections", "Non-consensual behaviour", "Online bullying/intimidation", "Pornography or sexualized content", "Protection of minors", "Risk for public security", "Scams and/or fraud", "Self-harm", "Scope of platform service", "Unsafe and/or illegal products", "Violence"]
        )
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

        
    
    
#####################################################################################################################################

    st.write('<h2 style="text-align: center; text-decoration: underline;">Historical Analysis</h2>', unsafe_allow_html=True)

    date_initial, date_final, company_intial, harm_intial = st.columns(4)

    with date_initial:
        st.markdown("<h4 style=' text-decoration: underline;'>Select an inital date:</h4>", unsafe_allow_html=True)
        # Calculate today's date
        today = datetime.now().date()

        # Calculate the initial date, which is five days before today
        initial_date = today - timedelta(days=5)

        # Format the initial date in 'YYYY-MM-DD'
        initial_date_str = initial_date.strftime("%Y-%m-%d")

        # Filter the datasets to only include dates that are less than or equal to the initial date
        filtered_datasetss = [date for date in datasets if date <= initial_date_str]

        # Use the filtered list in the selectbox
        date_initial_str = st.selectbox(
            "Choose a date from the dropdown below:",
            filtered_datasetss,
            index=filtered_datasetss.index(initial_date_str) if initial_date_str in filtered_datasetss else 0
)
        
        # date_initial_str = st.selectbox("Choose a date from the dropdown below:", datasets, index=datasets.index(initial_date_str))
        # #disable_others = selected_option == "General Data"


    data = [datetime.strptime(d, "%Y-%m-%d") for d in datasets]
    date_initial = datetime.strptime(date_initial_str, "%Y-%m-%d")
    filtered_dates = [date for date in data if date > date_initial]
    datasets_final = [date.strftime('%Y-%m-%d') for date in filtered_dates]



#list_of_companies_orig = ['TikTok', 'Pinterest', 'App Store', 'Snapchat', 'Booking.com', 'LinkedIn', 'X', 'Google Play', 'Google Maps', 'Facebook', 'Instagram', 'Amazon', 'YouTube', 'AliExpress', 'Google Shopping']
    list_of_companies_orig = ['TikTok', 'Pinterest', 'Snapchat', 'LinkedIn', 'X', 'Facebook', 'Instagram','YouTube','Reddit','Bumble']    
    list_of_harms_orig = ['STATEMENT_CATEGORY_ILLEGAL_OR_HARMFUL_SPEECH', 'STATEMENT_CATEGORY_SCOPE_OF_PLATFORM_SERVICE', 'STATEMENT_CATEGORY_PROTECTION_OF_MINORS', 'STATEMENT_CATEGORY_VIOLENCE', 'STATEMENT_CATEGORY_PORNOGRAPHY_OR_SEXUALIZED_CONTENT', 'STATEMENT_CATEGORY_DATA_PROTECTION_AND_PRIVACY_VIOLATIONS', 'STATEMENT_CATEGORY_SCAMS_AND_FRAUD', 'STATEMENT_CATEGORY_SELF_HARM', 'STATEMENT_CATEGORY_NEGATIVE_EFFECTS_ON_CIVIC_DISCOURSE_OR_ELECTIONS', 'STATEMENT_CATEGORY_INTELLECTUAL_PROPERTY_INFRINGEMENTS', 'STATEMENT_CATEGORY_UNSAFE_AND_ILLEGAL_PRODUCTS', 'STATEMENT_CATEGORY_NON_CONSENSUAL_BEHAVIOUR', 'STATEMENT_CATEGORY_ANIMAL_WELFARE', 'STATEMENT_CATEGORY_RISK_FOR_PUBLIC_SECURITY']

    with date_final:
        st.markdown("<h4 style=' text-decoration: underline;'>Select a final date:</h4>", unsafe_allow_html=True)
        date_final_str = st.selectbox("Choose a final date from the dropdown below:",datasets_final)
        
    with company_intial:
        st.markdown("<h4 style=' text-decoration: underline;'>Select a Company:</h4>", unsafe_allow_html=True)
        company_selected = st.selectbox("Choose a Company from the dropdown below:",list_of_companies_orig)

    with harm_intial:
        st.markdown("<h4 style=' text-decoration: underline;'>Select a Specific Harm:</h4>", unsafe_allow_html=True)
        harm_selected = st.selectbox("Choose a Harm from the dropdown below:",list_of_harms_orig)

            
        
  

    date_final = datetime.strptime(date_final_str, "%Y-%m-%d")

    # list_of_available_dates = [date_initial + timedelta(days=i) 
    #                             for i in range((date_final - date_initial).days + 1)
    #                             if (date_initial + timedelta(days=i)).strftime("%Y-%m-%d") in datasets]
    list_of_available_dates = [
    (date_initial + timedelta(days=i)).strftime("%Y-%m-%d")
    for i in range((date_final - date_initial).days + 1)
    if (date_initial + timedelta(days=i)).strftime("%Y-%m-%d") in datasets
]

    start_time = time.time()

    #Load all data in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
        datasets_loaded = list(executor.map(load_data, list_of_available_dates))

    # Record the end time
    end_time = time.time()

    # Calculate the duration
    duration = end_time - start_time

    print(f"Function load_data ran in {duration:.4f} seconds")

    # with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
    #     datasets_loaded = list(executor.map(load_data, list_of_available_dates))

    # Process the loaded data
    # Record the start time
    start_time = time.time()

    def process_data(data):
        return plot_acc_totals_per_harm_company_harm_historical(data, company_selected, harm_selected)

    # Using ThreadPoolExecutor to parallelize the function calls
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_data, datasets_loaded))
        

        # Record the end time
    end_time = time.time()

    # Calculate the duration
    duration = end_time - start_time

    print(f"Function (plot_acc_totals_per_harm_company_harm_historical) ran in {duration:.4f} seconds")


        


    # Create DataFrame
    df = pd.DataFrame({
        # 'Date': [date.strftime('%Y-%m-%d') for date in list_of_available_dates],
        'Dates': list_of_available_dates,
        'Automated': [result[0] for result in results],
        'Manual': [result[1] for result in results]
    })

   # df.set_index('Dates', inplace=True)
    col1, col2 = st.columns(2)

    # Melt the dataframe to a long format for Altair
    df_long = df.melt('Dates', var_name='Type', value_name='DAILY FLAGGED CONTENT')

    # Create an Altair chart
    chart = alt.Chart(df_long).mark_line().encode(
        x='Dates',
        y='DAILY FLAGGED CONTENT',
        color=alt.Color('Type', scale=alt.Scale(domain=['Automated', 'Manual'], range=['red', 'green'])),
        strokeDash='Type'
    ).properties(
        title='ACC Flag count vs User Flag count'
    )



    #PLOTTING THE GRAPHS BELOW and MAKING #2 PLOT

    with col1:
        # Display the chart in Streamlit
        st.altair_chart(chart, use_container_width=True)

    with col2:
            acc_total = df['Automated'].sum()
            user_total = df["Manual"].sum()

            # Create a DataFrame with the totals
            data_a = {
                'Category': ['Automated', 'Manual'],
                'Total Harm Count': [acc_total, user_total]
            }
            df_xx = pd.DataFrame(data_a)

            # Display total harm counts
            # st.write(f"Automated (ACC): {acc_total} ┃ Manual (User reported): {user_total}")
            #st.write(f"Manual (User reported): {user_total}")

            st.write(
                f"<span style='color:red; font-weight:bold;'>Automated</span> (ACC): {acc_total} ┃ "
                f"<span style='color:green; font-weight:bold;'>Manual</span> (User reported): {user_total}",
                unsafe_allow_html=True
)

            # Plotting the bar chart
            # st.bar_chart(df_xx.set_index('Category'), use_container_width=True)

            color_scale = alt.Scale(domain=['Automated', 'Manual'], range=['red', 'green'])

            # Create an Altair bar chart
            chart = alt.Chart(df_xx).mark_bar().encode(
                x=alt.X('Category:N', title=''),
                y=alt.Y('Total Harm Count:Q', title='TOTAL FLAGGED CONTENT'),
                color=alt.Color('Category:N', scale=color_scale, legend=None)
            ).properties(
                width=alt.Step(80)  # controls the width of the bars
            )

            # Display the chart
            st.altair_chart(chart, use_container_width=True)
    st.markdown("---")





##########################################################################################################
    st.write('<h2 style="text-align: center; text-decoration: underline;">Daily Live Analysis</h2>', unsafe_allow_html=True)

    # Dropdown for selecting dataset
    st.markdown("<h3 style=' text-decoration: underline;'>Select a Specific Date:</h3>", unsafe_allow_html=True)



    

    selected_dataset = st.selectbox("Choose a Date from the dropdown below:", datasets)


    # Load data and extract lists from selected dataset
    if selected_dataset:
        data, List_of_companies, List_of_harms, List_of_content_type, List_of_moderation_action, List_of_automation_status = load_data_from_dataset(selected_dataset + ".pkl")



    # Create columns for the dropdowns
    general_data_col, company_col, harm_col = st.columns(3)

    with general_data_col:
        st.markdown("<h3 style=' text-decoration: underline;'>Overall info for all companies</h3>", unsafe_allow_html=True)
        selected_option_gen = st.checkbox("General data")

        disable_others = selected_option_gen

    with company_col:
        st.markdown("<h3 style=' text-decoration: underline;'>Select a Specific Company:</h3>", unsafe_allow_html=True)
        selected_company = st.selectbox("Choose a Company from the dropdown below:", [None] + List_of_companies, disabled=disable_others)
        
    with harm_col:
        st.markdown("<h3 style=' text-decoration: underline;'>Select a Specific Harm:</h3>", unsafe_allow_html=True)
        selected_harm = st.selectbox("Choose a Harm from the dropdown below:", [None] + List_of_harms, disabled=disable_others)




    if selected_option_gen:
        st.markdown("---")
        st.subheader("Analysis for General Overview")
   
        x = plot_acc_totals_per_company(data)
        y = plot_acc_totals_per_harm(data)
        l = plot_acc_totals_per_moderation_action(data)
        n = plot_acc_totals_per_automation_status(data)
        b = plot_acc_totals_per_content_type(data)
        fig16 = plot_acc_totals(data)
        

        figtest = sum_harm(data)
        fig0 = plot_company_dataxxz(data, List_of_companies)
        fig0two = plot_company_dataxxz_normalized(data, List_of_companies)
        fig1 = plot_content_type_totals(data)
        fig2 = plot_moderation_action_totals(data)
        fig3 = plot_automation_status_totals(data)
        fig4 = plot_harm_totals_per_company(data)
        fig5 = plot_content_type_totals_per_company(data)
        fig6 = plot_automation_status_table_general(data)
        fig7 = plot_normalized_automation_status(data)
        fig8 = plot_harm_content_type(data)
        fig9 = plot_harm_content_type_normalized(data)
        fig10 = plot_harm_automation_status(data)
        fig10two = plot_harm_automation_status_two(data)
        fig11 = plot_content_type_automation_status(data)
        fig11two = plot_content_type_automation_status_two(data)

        #put fig 12 in own column
        fig12 = sum_reports_per_harm_per_moderation_action(data)
        #fig13 = generate_content_type_moderation_action_figure(data)
        fig14 = generate_moderation_action_automation_status_figure(data)
        fig15 = sum_reports_per_moderation_action_per_company(data)
    

        col1, col2 = st.columns(2)



        def change_label_style(label, font_size='12px', font_color='black', font_family='sans-serif'):
            html = f"""
            <script>
            var elems = window.parent.document.querySelectorAll('p');
            var elem = Array.from(elems).find(x => x.innerText == '{label}');
            elem.style.fontSize = '{font_size}';
            elem.style.color = '{font_color}';
            elem.style.fontFamily = '{font_family}';
            </script>
            """
            st.components.v1.html(html)

      #  Your plot generation code (assuming fig0 is defined somewhere)

        with col1:
              with st.expander("Total ACC detections per moderation action", expanded=False):
                  # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
                  #change_label_style("Total number of Moderation actions per harm", font_size='30px')
                  #st.pyplot(l)
                  st.dataframe(l, use_container_width=True)

        with col2:
              with st.expander("Total ACC detections per automation decision status", expanded=False):
                  # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
                  #change_label_style("Total number of Moderation actions per harm", font_size='30px')
                  #st.pyplot(n)
                  st.dataframe(n, use_container_width=True)

        with col1:
              with st.expander("Total ACC detections per content type", expanded=False):
                  # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
                  #change_label_style("Total number of Moderation actions per harm", font_size='30px')
                  #st.pyplot(b)
                  st.dataframe(b, use_container_width=True)

        with col2:
              with st.expander("Total ACC detections per company", expanded=False):
                  # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
                  #change_label_style("Total number of Moderation actions per harm", font_size='30px')
                  st.dataframe(x, use_container_width=True)

        with col1:
              with st.expander("Total ACC detections per harm", expanded=False):
                  # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
                  #change_label_style("Total number of Moderation actions per harm", font_size='30px')
                  st.dataframe(y, use_container_width=True)

        with col2:
            with st.expander("Total count for manual vs automated detection", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
                #change_label_style("Total number of Moderation actions per harm", font_size='30px')
                st.dataframe(fig16, use_container_width=True)

        with col1:
            with st.expander("Total number of Moderation actions per harm", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
                #change_label_style("Total number of Moderation actions per harm", font_size='30px')
                st.dataframe(figtest, use_container_width=True)


        with col2:
            with st.expander("Total number of Moderation Actions per Company", expanded=False):
              #  change_label_style("Total number of Moderation Actions per Company", font_size='30px')
                st.dataframe(fig0, use_container_width=True)
            
        with col1:
            with st.expander("Total number of Moderation Actions per Company Normalized", expanded=False):
              #  change_label_style("Total number of Moderation Actions per Company", font_size='30px')
                st.dataframe(fig0two, use_container_width=True)
                   

        with col2:
            with st.expander("Total number of Moderation Actions per Type of Content", expanded=False):
               # change_label_style("Total number of Moderation Actions per Type of Content", font_size='30px')
                #st.image(fig_to_png(fig3), use_column_width=True, width = 500)
                st.dataframe(fig1, use_container_width=True)

        with col1:
            with st.expander("Total number of Moderation Actions per Type of Automation Status", expanded=False):
               # st.image(fig_to_png(fig4), use_column_width=True, width = 500)
              # change_label_style("Total number of Moderation Actions per Type of Automation Status", font_size='30px')
               st.dataframe(fig3, use_container_width=True)

        with col2:
            with st.expander("Total number of Moderation Actions per Type of Moderation Decision", expanded=False):
               # change_label_style("Total number of Moderation Actions per Type of Moderation Decision", font_size='30px')
                #st.image(fig_to_png(fig5), use_column_width=True, width = 500)
                st.dataframe(fig2, use_container_width=True)

        with col1:
            with st.expander("Number of reported Harms per Company", expanded=False):
              #  change_label_style("Number of reported Harms per Company", font_size='30px')
              #  st.image(fig_to_png(fig6), use_column_width=True, width = 500)
                st.dataframe(fig4, use_container_width=True)

        with col2:
            with st.expander("Number of reported content type per Company", expanded=False):
              #  change_label_style("Number of reported content type per Company", font_size='30px')
               # st.image(fig_to_png(fig8), use_column_width=True, width = 1100)
                st.dataframe(fig5, use_container_width=True)



        with col1:
            with st.expander("Normalized counts of each automation status per company", expanded=False):
              #  st.image(fig_to_png(fig9), use_column_width=True, width = 1100)
               # change_label_style("Normalized counts of each automation status per company", font_size='30px')
                st.dataframe(fig7, use_container_width=True)

        with col2:
            with st.expander("Number of reported content type per Harm", expanded=False):
              #  change_label_style("Number of reported content type per Harm", font_size='30px')
                st.dataframe(fig8, use_container_width=True)


        with col1:
            with st.expander("Number of reported content type per Harm Normalized", expanded=False):
               # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
               # change_label_style("Count for each harm per content type", font_size='30px')
                st.dataframe(fig9, use_container_width=True)
                
        with col2:
            with st.expander("Count for each harm per automation status", expanded=False):
               # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
                #change_label_style("Count for each harm per automation status", font_size='30px')
                st.dataframe(fig10, use_container_width=True)
     
        with col1:
            with st.expander("Count for each harm per automation status normalized", expanded=False):
               # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
               # change_label_style("Count for each harm per automation status normalized", font_size='30px')
                st.dataframe(fig10two, use_container_width=True)

        with col2:
            with st.expander("Count of each Harm per Moderation decision", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
                #change_label_style("Count of each Harm per Moderation decision", font_size='30px')
                st.dataframe(fig12, use_container_width=True)

        #with col2:
           # with st.expander("Count of each content type per Moderation decision", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
             #   change_label_style("Count of each content type per Moderation decision", font_size='30px')
               # st.pyplot(fig13)

        with col1:
            with st.expander("Count for each content type per automation status", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
              #  change_label_style("Count for each content type per automation status", font_size='30px')
                st.dataframe(fig11, use_container_width=True)

        with col2:
            with st.expander("Count for each content type per automation status Normalized", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
               # change_label_style("Count for each content type per automation status Normalized", font_size='30px')
                st.dataframe(fig11two, use_container_width=True)
        with col1:
            with st.expander("Number of reported moderation decision per company", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
               # change_label_style("Number of reported moderation decision per company", font_size='30px')
                st.dataframe(fig15, use_container_width=True)


    elif selected_company and selected_harm:
        st.markdown("---")
        st.subheader(f"Analysis for {selected_company} and {selected_harm}")


        x = plot_acc_totals_per_company_company_harm(data, selected_company, selected_harm)
        y = plot_acc_totals_per_harm_company_harm(data, selected_company, selected_harm)
        l = plot_acc_totals_per_moderation_action_company_harm(data, selected_company, selected_harm)
        n = plot_acc_totals_per_automation_status_company_harm(data, selected_company, selected_harm)
        b = plot_acc_totals_per_content_type_company_harm(data, selected_company, selected_harm)
        fig16 = plot_acc_totals_company_harm(data, selected_company, selected_harm)

        figtest = sum_harm3(data, selected_company,  selected_harm)
        fig0 = plot_company_dataxxz3(data, selected_company,  selected_harm)
        fig0two = plot_company_dataxxz3_normalized(data, selected_company,  selected_harm)
        fig1 = plot_content_type_totals3(data, selected_company,  selected_harm)
        fig2 = plot_moderation_action_totals3(data, selected_company,  selected_harm)
        fig3 = plot_automation_status_totals3(data, selected_company,  selected_harm)
        fig4 = plot_harm_totals_per_company3(data, selected_company,  selected_harm)
        fig5 = plot_content_type_totals_per_company3(data, selected_company,  selected_harm)
        fig6 = plot_automation_status_table_general3(data, selected_company,  selected_harm)
        fig7 = plot_normalized_automation_status3(data, selected_company,  selected_harm)
        #fig8 = plot_harm_content_type3(data, selected_company,  selected_harm)
        fig9 = plot_harm_content_type3_normalized(data, selected_company,  selected_harm)
        fig9two = plot_harm_content_type_normalized3(data, selected_company,  selected_harm)
        fig10 = plot_harm_automation_status3(data, selected_company,  selected_harm)
        fig10two = plot_harm_automation_status3_normalized(data, selected_company,  selected_harm)
        fig11 = plot_content_type_automation_status3(data, selected_company,  selected_harm)
        fig11two = plot_content_type_automation_status3_normalized(data, selected_company,  selected_harm)
       # fig12 = sum_reports_per_harm_per_moderation_action3(data, selected_company,  selected_harm)
       # fig13 = generate_content_type_moderation_action_figure3(data, selected_company,  selected_harm)
        fig14 = generate_moderation_action_automation_status_figure3(data, selected_company,  selected_harm)
        fig15 = sum_reports_per_moderation_action_per_company3(data, selected_company,  selected_harm)

        col1, col2 = st.columns(2)



        def change_label_style(label, font_size='12px', font_color='black', font_family='sans-serif'):
            html = f"""
            <script>
            var elems = window.parent.document.querySelectorAll('p');
            var elem = Array.from(elems).find(x => x.innerText == '{label}');
            elem.style.fontSize = '{font_size}';
            elem.style.color = '{font_color}';
            elem.style.fontFamily = '{font_family}';
            </script>
            """
            st.components.v1.html(html)

      #  Your plot generation code (assuming fig0 is defined somewhere)

        with col1:
              with st.expander("Total ACC detections per moderation action", expanded=False):
                  # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
                  #change_label_style("Total number of Moderation actions per harm", font_size='30px')
                  #st.pyplot(l)
                  st.dataframe(l, use_container_width=True)

        with col2:
              with st.expander("Total ACC detections per automation decision status", expanded=False):
                  # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
                  #change_label_style("Total number of Moderation actions per harm", font_size='30px')
                  #st.pyplot(n)
                  st.dataframe(n, use_container_width=True)

        with col1:
              with st.expander("Total ACC detections per content type", expanded=False):
                  # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
                  #change_label_style("Total number of Moderation actions per harm", font_size='30px')
                  #st.pyplot(b)
                  st.dataframe(b, use_container_width=True)

        with col2:
              with st.expander("Total ACC detections per company", expanded=False):
                  # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
                  #change_label_style("Total number of Moderation actions per harm", font_size='30px')
                 # st.pyplot(x)
                  st.dataframe(x, use_container_width=True)

        with col1:
              with st.expander("Total ACC detections per harm", expanded=False):
                  # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
                  #change_label_style("Total number of Moderation actions per harm", font_size='30px')
                 # st.pyplot(y)
                  st.dataframe(y, use_container_width=True)

        with col2:
            with st.expander("Total count for manual vs automated detection", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
                #change_label_style("Total number of Moderation actions per harm", font_size='30px')
              #  st.pyplot(fig16)
                st.dataframe(fig16, use_container_width=True)

        with col1:
            with st.expander("Total number of Moderation actions for selected harm and company", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
             #   change_label_style("Total number of Moderation actions per harm", font_size='30px')
              #  st.pyplot(figtest)
                st.dataframe(figtest, use_container_width=True)

        with col2:
            with st.expander("Total number of Moderation Actions normalized  for selected harm and company", expanded=False):
              #  change_label_style("Total number of Moderation Actions per Company", font_size='30px')
                #st.pyplot(fig0two)
                st.dataframe(fig0two, use_container_width=True)
                   

        with col1:
            with st.expander("Total number of Moderation Actions per Type of Content for selected harm and company", expanded=False):
              #  change_label_style("Total number of Moderation Actions per Type of Content", font_size='30px')
                #st.image(fig_to_png(fig3), use_column_width=True, width = 500)
               #st.pyplot(fig1)
                st.dataframe(fig1, use_container_width=True)

        with col2:
            with st.expander("Total number of Moderation Actions per Type of moderation action for selected harm and company", expanded=False):
               # st.image(fig_to_png(fig4), use_column_width=True, width = 500)
             #  change_label_style("Total number of Moderation Actions per Type of Automation Status", font_size='30px')
              # st.pyplot(fig2)
               st.dataframe(fig2, use_container_width=True)

        with col1:
            with st.expander("Total number of Moderation Actions per Type of Moderation Decision for selected harm and company", expanded=False):
              #  change_label_style("Total number of Moderation Actions per Type of Moderation Decision", font_size='30px')
                #st.image(fig_to_png(fig5), use_column_width=True, width = 500)
               # st.pyplot(fig3)
                st.dataframe(fig3, use_container_width=True)

        with col2:
            with st.expander("Number of reported Harms for selected harm and company", expanded=False):
             #   change_label_style("Number of reported Harms per Company", font_size='30px')
                #st.pyplot(fig4)
                st.dataframe(fig4, use_container_width=True)

        with col1:
            with st.expander("Number of reported content type  for selected harm and company", expanded=False):
               # change_label_style("Number of reported content type per Company", font_size='30px')
               # st.image(fig_to_png(fig8), use_column_width=True, width = 1100)
               # st.pyplot(fig5)
                st.dataframe(fig5, use_container_width=True)

        with col2:
            with st.expander("Number of Automation Status type for selected harm and company", expanded=False):
             #   change_label_style("Number of Automation Status type per Company", font_size='30px')
                #st.image(fig_to_png(fig10), use_column_width=True, width = 1100)
               # st.pyplot(fig6)
                st.dataframe(fig6, use_container_width=True)


        with col1:
            with st.expander("Normalized counts of each automation status for selected harm and company", expanded=False):
              #  st.image(fig_to_png(fig9), use_column_width=True, width = 1100)
              #  change_label_style("Normalized counts of each automation status per company", font_size='30px')
               # st.pyplot(fig7)
                st.dataframe(fig7, use_container_width=True)

        with col2:
            with st.expander("Number of reported content type normalized for selected harm and company", expanded=False):
               # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
             #   change_label_style("Count for each harm per content type", font_size='30px')
                #st.pyplot(fig9)
                st.dataframe(fig9, use_container_width=True)
        
        with col1:
            with st.expander("Count for each harm per automation status normalized for selected harm and company", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
              ##  change_label_style("Count for each harm per automation status", font_size='30px')
               # st.pyplot(fig10two)
                st.dataframe(fig10two, use_container_width=True)

        with col2:
            with st.expander("Count for each content type per automation status for selected harm and company", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
              #  change_label_style("Count for each content type per automation status", font_size='30px')
             #   st.pyplot(fig11)
                st.dataframe(fig11, use_container_width=True)

        with col1:
            with st.expander("Count for each content type per automation status normalized for selected harm and company", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
              #  change_label_style("Count for each content type per automation status", font_size='30px')
              #  st.pyplot(fig11two)
                st.dataframe(fig11two, use_container_width=True)
        # with col1:
        #     with st.expander("Count of each Harm per Moderation decision", expanded=False):
        #         # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
        #      #   change_label_style("Count of each Harm per Moderation decision", font_size='30px')
        #         st.pyplot(fig12)
        # with col2:
        #     with st.expander("Count of each content type per Moderation decision", expanded=False):
        #         # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
        #      #   change_label_style("Count of each content type per Moderation decision", font_size='30px')
        #         st.pyplot(fig13)
        with col2:
            with st.expander("Count of moderation decision per automation status for selected harm and company", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
               # change_label_style("Count of moderation decision per automation status", font_size='30px')
              #  st.pyplot(fig14)
                st.dataframe(fig14, use_container_width=True)
        with col1:
            with st.expander("Number of reported moderation decision for selected harm and company", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
               # change_label_style("Number of reported moderation decision per company", font_size='30px')
             #   st.pyplot(fig15)
                st.dataframe(fig15, use_container_width=True)



   


    #fix last graph and rename expander titles
    elif selected_company:
        st.markdown("---")
        st.subheader(f"Analysis for {selected_company}")


        x = plot_acc_totals_per_company_company(data, selected_company)
        y = plot_acc_totals_per_harm_company(data, selected_company)
        l = plot_acc_totals_per_moderation_action_company(data, selected_company)
        n = plot_acc_totals_per_automation_status_company(data, selected_company)
        b = plot_acc_totals_per_content_type_company(data, selected_company)
        fig16 = plot_acc_totals_company(data, selected_company)

        figtest = sum_harm1(data, selected_company)
        fig0 = plot_company_dataxxz1(data, selected_company)
        fig0two = plot_company_dataxxz1_normalized(data, selected_company)
        fig1 = plot_content_type_totals1(data, selected_company)
        fig2 = plot_moderation_action_totals1(data, selected_company)
        fig3 = plot_automation_status_totals1(data, selected_company)
        fig4 = plot_harm_totals_per_company1(data, selected_company)
        fig5 = plot_content_type_totals_per_company1(data, selected_company)
        fig6 = plot_automation_status_table_general1(data, selected_company)
        fig7 = plot_normalized_automation_status1(data, selected_company)
      #  fig8 = plot_harm_content_type1(data, selected_company)
        fig9 = plot_harm_content_type_1(data, selected_company)
        fig9two = plot_harm_content_type_normalized1(data, selected_company)
        fig10 = plot_harm_automation_status1(data, selected_company)
        fig10two = plot_harm_automation_status1_normalized(data, selected_company)
        fig11 = plot_content_type_automation_status1(data, selected_company)
        fig11two = plot_content_type_automation_status1_normalized(data, selected_company)
        #fig12 = sum_reports_per_harm_per_moderation_action1(data, selected_company)
     #   fig13 = generate_content_type_moderation_action_figure1(data, selected_company)
        fig14 = generate_moderation_action_automation_status_figure1(data, selected_company)
        fig15 = sum_reports_per_moderation_action_per_company1(data, selected_company)

        col1, col2 = st.columns(2)



        def change_label_style(label, font_size='12px', font_color='black', font_family='sans-serif'):
            html = f"""
            <script>
            var elems = window.parent.document.querySelectorAll('p');
            var elem = Array.from(elems).find(x => x.innerText == '{label}');
            elem.style.fontSize = '{font_size}';
            elem.style.color = '{font_color}';
            elem.style.fontFamily = '{font_family}';
            </script>
            """
            st.components.v1.html(html)

      #  Your plot generation code (assuming fig0 is defined somewhere)

        with col1:
              with st.expander("Total ACC detections per moderation action", expanded=False):
                  # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
                  #change_label_style("Total number of Moderation actions per harm", font_size='30px')
                 #st.pyplot(l)
                  st.dataframe(l, use_container_width=True)

        with col2:
              with st.expander("Total ACC detections per automation decision status", expanded=False):
                  # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
                  #change_label_style("Total number of Moderation actions per harm", font_size='30px')
                 # st.pyplot(n)
                  st.dataframe(n, use_container_width=True)

        with col1:
              with st.expander("Total ACC detections per content type", expanded=False):
                  # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
                  #change_label_style("Total number of Moderation actions per harm", font_size='30px')
                  #st.pyplot(b)
                  st.dataframe(b, use_container_width=True)

        with col2:
              with st.expander("Total ACC detections per company", expanded=False):
                  # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
                  #change_label_style("Total number of Moderation actions per harm", font_size='30px')
                #  st.pyplot(x)
                  st.dataframe(x, use_container_width=True)

        with col1:
              with st.expander("Total ACC detections per harm", expanded=False):
                  # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
                  #change_label_style("Total number of Moderation actions per harm", font_size='30px')
                #  st.pyplot(y)
                  st.dataframe(y, use_container_width=True)

        with col2:
            with st.expander("Total count for manual vs automated detection", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
                #change_label_style("Total number of Moderation actions per harm", font_size='30px')
              #  st.pyplot(fig16)
                st.dataframe(fig16, use_container_width=True)




        with col1:
            with st.expander("Total number of Moderation actions per harm", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
               # change_label_style("Total number of Moderation actions per harm", font_size='30px')
              #  st.pyplot(figtest)
                st.dataframe(figtest, use_container_width=True)


        with col2:
            with st.expander("Total number of Moderation Actions per Type of Automation Status", expanded=False):
               # st.image(fig_to_png(fig4), use_column_width=True, width = 500)
              # change_label_style("Total number of Moderation Actions per Type of Automation Status", font_size='30px')
             #  st.pyplot(fig3)
               st.dataframe(fig3, use_container_width=True)


        with col1:
            with st.expander("Total number of Moderation Actions per Company", expanded=False):
               # change_label_style("Total number of Moderation Actions per Company", font_size='30px')
               # st.pyplot(fig0)
                st.dataframe(fig0, use_container_width=True)

        with col2:
            with st.expander("Total number of Moderation Actions per Company normalized", expanded=False):
               # change_label_style("Total number of Moderation Actions per Company", font_size='30px')
               # st.pyplot(fig0two)
                st.dataframe(fig0two, use_container_width=True)
                

        with col1:
            with st.expander("Total number of Moderation Actions per Type of Moderation Decision", expanded=False):
              #  change_label_style("Total number of Moderation Actions per Type of Moderation Decision", font_size='30px')
                #st.image(fig_to_png(fig5), use_column_width=True, width = 500)
               # st.pyplot(fig2)
                st.dataframe(fig2, use_container_width=True)

        with col2:
            with st.expander("Number of reported Harms per Company", expanded=False):
             #   change_label_style("Number of reported Harms per Company", font_size='30px')
                #st.pyplot(fig4)
                st.dataframe(fig4, use_container_width=True)

        with col1:
            with st.expander("Number of reported content type  per Company", expanded=False):
             #   change_label_style("Number of reported content type per Company", font_size='30px')
               # st.image(fig_to_png(fig8), use_column_width=True, width = 1100)
               # st.pyplot(fig5)
                st.dataframe(fig5, use_container_width=True)

        with col2:
            with st.expander("Number of Automation Status type per Company", expanded=False):
             #   change_label_style("Number of Automation Status type per Company", font_size='30px')
                #st.image(fig_to_png(fig10), use_column_width=True, width = 1100)
               # st.pyplot(fig6)
                st.dataframe(fig6, use_container_width=True)


        with col1:
            with st.expander("Normalized counts of each automation status per company", expanded=False):
              #  st.image(fig_to_png(fig9), use_column_width=True, width = 1100)
              #  change_label_style("Normalized counts of each automation status per company", font_size='30px')
               # st.pyplot(fig7)
                st.dataframe(fig7, use_container_width=True)


        with col2:
            with st.expander("Count for each harm per content type", expanded=False):
               # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
             #   change_label_style("Count for each harm per content type", font_size='30px')
              #  st.pyplot(fig9)
                st.dataframe(fig9, use_container_width=True)

        with col1:
            with st.expander("Count for each harm per content type Normalized", expanded=False):
               # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
             #   change_label_style("Count for each harm per content type", font_size='30px')
               # st.pyplot(fig9two)
                st.dataframe(fig9two, use_container_width=True)
                
        with col2:
            with st.expander("Count for each harm per automation status", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
               # change_label_style("Count for each harm per automation status", font_size='30px')
              #  st.pyplot(fig10)
                st.dataframe(fig10, use_container_width=True)

        with col1:
            with st.expander("Count for each harm per automation status normalized", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
               # change_label_style("Count for each harm per automation status", font_size='30px')
            #    st.pyplot(fig10two)
                st.dataframe(fig10two, use_container_width=True)

        with col2:
            with st.expander("Count for each content type per automation status", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
             #   change_label_style("Count for each content type per automation status", font_size='30px')
              #  st.pyplot(fig11)
                st.dataframe(fig11, use_container_width=True)

        with col1:
            with st.expander("Count for each content type per automation status Normalized", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
             #   change_label_style("Count for each content type per automation status", font_size='30px')
               # st.pyplot(fig11two)
                st.dataframe(fig11two, use_container_width=True)

        with col2:
            with st.expander("Count of moderation decision per automation status", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
               # change_label_style("Count of moderation decision per automation status", font_size='30px')
              #  st.pyplot(fig14)
                st.dataframe(fig14, use_container_width=True)
        with col1:
            with st.expander("Number of reported moderation decision per company", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
               # change_label_style("Number of reported moderation decision per company", font_size='30px')
               # st.pyplot(fig15)
                st.dataframe(fig15, use_container_width=True)




    elif selected_harm:
        st.markdown("---")
        st.subheader(f"Analysis for {selected_harm}")

        x = plot_acc_totals_per_company_harm(data, selected_harm)
        y = plot_acc_totals_per_harm_harm(data, selected_harm)
        l = plot_acc_totals_per_moderation_action_harm(data, selected_harm)
        n = plot_acc_totals_per_automation_status_harm(data, selected_harm)
        b = plot_acc_totals_per_content_type_harm(data, selected_harm)
        fig16 = plot_acc_totals_harm(data, selected_harm)

        figtest = sum_harm2(data, selected_harm)
       # fig0 = plot_company_dataxxz2(data, selected_harm)
     #   fig0two = plot_company_dataxxz2_normalized(data, selected_harm)
        fig1 = plot_content_type_totals2(data, selected_harm)
        fig2 = plot_moderation_action_totals2(data, selected_harm)
        fig3 = plot_automation_status_totals2(data, selected_harm)
        fig4 = plot_harm_totals_per_company2(data, selected_harm)
      #  fig5 = plot_content_type_totals_per_company2(data, selected_harm)
        fig6 = plot_automation_status_table_general2(data, selected_harm)
        fig7 = plot_normalized_automation_status2(data, selected_harm)
    #    fig8 = plot_harm_content_type2(data, selected_harm)
        fig9 = plot_harm_content_type_normalized2(data, selected_harm)
        fig10 = plot_harm_automation_status2(data, selected_harm)
        fig10two = plot_harm_automation_status2_normalized(data, selected_harm)
        fig11 = plot_content_type_automation_status2(data, selected_harm)
        fig11two = plot_content_type_automation_status2_normalized(data, selected_harm)
        #fig12 = sum_reports_per_harm_per_moderation_action2(data, selected_harm)
      #  fig13 = generate_content_type_moderation_action_figure2(data, selected_harm)
        fig14 = generate_moderation_action_automation_status_figure2(data, selected_harm)
      #  fig15 = sum_reports_per_moderation_action_per_company2(data, selected_harm)

        col1, col2 = st.columns(2)



        def change_label_style(label, font_size='12px', font_color='black', font_family='sans-serif'):
            html = f"""
            <script>
            var elems = window.parent.document.querySelectorAll('p');
            var elem = Array.from(elems).find(x => x.innerText == '{label}');
            elem.style.fontSize = '{font_size}';
            elem.style.color = '{font_color}';
            elem.style.fontFamily = '{font_family}';
            </script>
            """
            st.components.v1.html(html)

      #  Your plot generation code (assuming fig0 is defined somewhere)
        

        with col1:
              with st.expander("Total ACC detections per moderation action", expanded=False):
                  # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
                  #change_label_style("Total number of Moderation actions per harm", font_size='30px')
                 # st.pyplot(l)
                  #st.pyplot(n)
                  st.dataframe(l, use_container_width=True)

        with col2:
              with st.expander("Total ACC detections per automation decision status", expanded=False):
                  # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
                  #change_label_style("Total number of Moderation actions per harm", font_size='30px')
                  #st.pyplot(n)
                  st.dataframe(n, use_container_width=True)

        with col1:
              with st.expander("Total ACC detections per content type", expanded=False):
                  # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
                  #change_label_style("Total number of Moderation actions per harm", font_size='30px')
                 # st.pyplot(b)
                  st.dataframe(b, use_container_width=True)

        with col2:
              with st.expander("Total ACC detections per company", expanded=False):
                  # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
                  #change_label_style("Total number of Moderation actions per harm", font_size='30px')
                 # st.pyplot(x)
                  st.dataframe(x, use_container_width=True)

        with col1:
              with st.expander("Total ACC detections per harm", expanded=False):
                  # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
                  #change_label_style("Total number of Moderation actions per harm", font_size='30px')
                 # st.pyplot(y)
                  st.dataframe(y, use_container_width=True)

        with col2:
            with st.expander("Total count for manual vs automated detection", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
                #change_label_style("Total number of Moderation actions per harm", font_size='30px')
                #st.pyplot(fig16)
                st.dataframe(fig16, use_container_width=True)

        with col1:
            with st.expander("Total number of Moderation actions per harm for harm", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
                #change_label_style("Total number of Moderation actions per harm", font_size='30px')
                #st.pyplot(figtest)
                st.dataframe(figtest, use_container_width=True)

                   

        with col2:
            with st.expander("Total number of Moderation Actions per Type of Content for harm", expanded=False):
               # change_label_style("Total number of Moderation Actions per Type of Content", font_size='30px')
                #st.image(fig_to_png(fig3), use_column_width=True, width = 500)
              #  st.pyplot(fig1)
                st.dataframe(fig1, use_container_width=True)

        with col1:
            with st.expander("Total number of Moderation Actions per Type of Automation Status for harm", expanded=False):
               # st.image(fig_to_png(fig4), use_column_width=True, width = 500)
              # change_label_style("Total number of Moderation Actions per Type of Automation Status", font_size='30px')
            #   st.pyplot(fig2)
               st.dataframe(fig2, use_container_width=True)

        with col2:
            with st.expander("Total number of Automation status for harm", expanded=False):
             #   change_label_style("Total number of Moderation Actions per Type of Moderation Decision", font_size='30px')
                #st.image(fig_to_png(fig5), use_column_width=True, width = 500)
              #  st.pyplot(fig3)
                st.dataframe(fig3, use_container_width=True)

        with col1:
            with st.expander("Number of reported Harms per Company for harm", expanded=False):
              #  change_label_style("Number of reported Harms per Company", font_size='30px')
            #    st.pyplot(fig4)
                st.dataframe(fig4, use_container_width=True)

        with col2:
            with st.expander("Number of Automation Status type per Company for harm", expanded=False):
               # change_label_style("Number of Automation Status type per Company", font_size='30px')
                #st.image(fig_to_png(fig10), use_column_width=True, width = 1100)
           #     st.pyplot(fig6)
                st.dataframe(fig6, use_container_width=True)


        with col1:
            with st.expander("Normalized counts of each automation status per company for harm", expanded=False):
              #  st.image(fig_to_png(fig9), use_column_width=True, width = 1100)
              #  change_label_style("Normalized counts of each automation status per company", font_size='30px')
                #st.pyplot(fig7)
                st.dataframe(fig7, use_container_width=True)

        with col2:
            with st.expander("Number of reported content type per Harm Normalized for harm", expanded=False):
               # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
               # change_label_style("Count for each harm per content type", font_size='30px')
               # st.pyplot(fig9)
                st.dataframe(fig9, use_container_width=True)
                
        with col1:
            with st.expander("Count for each harm per automation status for harm", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
               # change_label_style("Count for each harm per automation status", font_size='30px')
               # st.pyplot(fig10)
                st.dataframe(fig10, use_container_width=True)

        with col2:
            with st.expander("Count for each harm per automation status normalized for harm", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
               # change_label_style("Count for each harm per automation status", font_size='30px')
               # st.pyplot(fig10two)
                st.dataframe(fig10two, use_container_width=True)

        with col1:
            with st.expander("Count for each content type per automation status for harm", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
               # change_label_style("Count for each content type per automation status", font_size='30px')
                #st.pyplot(fig11)
                st.dataframe(fig11, use_container_width=True)

        with col2:
            with st.expander("Count for each content type per automation status normalized for harm", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
               # change_label_style("Count for each content type per automation status", font_size='30px')
                st.dataframe(fig11two, use_container_width=True)

        with col1:
            with st.expander("Count of moderation decision per automation status for harm", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
              #  change_label_styl
              # e("Count of moderation decision per automation status", font_size='30px')
               # st.pyplot(fig14)
                st.dataframe(fig14, use_container_width=True)
            

    else:
        st.write("No dataset selected.")

if __name__ == "__main__":
    main()
