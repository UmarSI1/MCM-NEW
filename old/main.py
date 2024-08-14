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


from general import *
from general_temp_for_company import *
from general_temp_for_harm import *
from general_temp_for_harm_and_comp import *

st.set_option('deprecation.showPyplotGlobalUse', False)


# Azure Blob Storage configuration
connection_string = 'DefaultEndpointsProtocol=https;AccountName=asatrustandsafetycv;AccountKey=PRjiFgPwqYyUvO+hQwi5Olh/gxjPs+EqWH2t5YrwJH/fSThCYpxfyYgyFLDlXuWGt4nkYJaeZqJw+AStxQ0Bog==;EndpointSuffix=core.windows.net'
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
datasets = [blob.name for blob in blobs_list]

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

################################################################################################################

# Main function to run the Streamlit app
def main():


    st.set_page_config(layout="wide")
    st.write('<h1 style="text-align: center;">Content moderation daily monitor</h1>', unsafe_allow_html=True)
    st.markdown("---")
    st.write('<h4 style="text-align: center;">This dashboard presents the daily count of moderation actions categorized by harm and platform.</h4>', unsafe_allow_html=True)
    st.markdown("---")


    # Dropdown for selecting dataset
    # Dropdown for selecting dataset
    selected_dataset = st.selectbox("Select a dataset", datasets)

    # Load data and extract lists from selected dataset
    if selected_dataset:
        data, List_of_companies, List_of_harms, List_of_content_type, List_of_moderation_action, List_of_automation_status = load_data_from_dataset(selected_dataset)
    else:
        st.write("No dataset selected.")

    # Create columns for the dropdowns
    company_col, harm_col, general_data_col = st.columns(3)

    
    with company_col:
        st.subheader("Select a Specific Company:")
        selected_company = st.selectbox("Company:", [None] + list(List_of_companies))
        # Dropdown for selecting a specific harm


    with harm_col:
        st.subheader("Select a Specific Harm:")
        harm_options = [None] + list(List_of_harms)
        selected_harm = st.selectbox("Harm:", harm_options)

    # Dropdown for selecting general data
    with general_data_col:
        st.subheader("Select General Data:")
        selected_option = st.selectbox("Option:", ["None", "General Data"])
       # selected_option = st.selectbox("Option:", ["General Data"])





    if selected_option == "General Data":
        st.markdown("---")
        st.subheader("Analysis for General Data")

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
       # fig13 = generate_content_type_moderation_action_figure(data)
       # fig14 = generate_moderation_action_automation_status_figure(data)
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
            with st.expander("Total number of Moderation actions per harm", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
                #change_label_style("Total number of Moderation actions per harm", font_size='30px')
                st.pyplot(figtest)


        with col2:
            with st.expander("Total number of Moderation Actions per Company", expanded=False):
              #  change_label_style("Total number of Moderation Actions per Company", font_size='30px')
                st.pyplot(fig0)
            
        with col1:
            with st.expander("Total number of Moderation Actions per Company Normalized", expanded=False):
              #  change_label_style("Total number of Moderation Actions per Company", font_size='30px')
                st.pyplot(fig0two)
                   

        with col2:
            with st.expander("Total number of Moderation Actions per Type of Content", expanded=False):
               # change_label_style("Total number of Moderation Actions per Type of Content", font_size='30px')
                #st.image(fig_to_png(fig3), use_column_width=True, width = 500)
                st.pyplot(fig1)

        with col1:
            with st.expander("Total number of Moderation Actions per Type of Automation Status", expanded=False):
               # st.image(fig_to_png(fig4), use_column_width=True, width = 500)
              # change_label_style("Total number of Moderation Actions per Type of Automation Status", font_size='30px')
               st.pyplot(fig3)

        with col2:
            with st.expander("Total number of Moderation Actions per Type of Moderation Decision", expanded=False):
               # change_label_style("Total number of Moderation Actions per Type of Moderation Decision", font_size='30px')
                #st.image(fig_to_png(fig5), use_column_width=True, width = 500)
                st.pyplot(fig2)

        with col1:
            with st.expander("Number of reported Harms per Company", expanded=False):
              #  change_label_style("Number of reported Harms per Company", font_size='30px')
              #  st.image(fig_to_png(fig6), use_column_width=True, width = 500)
                st.pyplot(fig4)

        with col2:
            with st.expander("Number of reported content type per Company", expanded=False):
              #  change_label_style("Number of reported content type per Company", font_size='30px')
               # st.image(fig_to_png(fig8), use_column_width=True, width = 1100)
                st.pyplot(fig5)

        #with col1:
         #   with st.expander("Count of each type of Automation Status", expanded=False):
              #  change_label_style("Number of Automation Status type per Company", font_size='30px')
                #st.image(fig_to_png(fig10), use_column_width=True, width = 1100)
               # st.pyplot(fig6)


        with col2:
            with st.expander("Normalized counts of each automation status per company", expanded=False):
              #  st.image(fig_to_png(fig9), use_column_width=True, width = 1100)
               # change_label_style("Normalized counts of each automation status per company", font_size='30px')
                st.pyplot(fig7)

        with col1:
            with st.expander("Number of reported content type per Harm", expanded=False):
              #  change_label_style("Number of reported content type per Harm", font_size='30px')
                st.pyplot(fig8)


        with col2:
            with st.expander("Number of reported content type per Harm Normalized", expanded=False):
               # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
               # change_label_style("Count for each harm per content type", font_size='30px')
                st.pyplot(fig9)
                
        with col1:
            with st.expander("Count for each harm per automation status", expanded=False):
               # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
                #change_label_style("Count for each harm per automation status", font_size='30px')
                st.pyplot(fig10)
     
        with col2:
            with st.expander("Count for each harm per automation status normalized", expanded=False):
               # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
               # change_label_style("Count for each harm per automation status normalized", font_size='30px')
                st.pyplot(fig10two)

        with col1:
            with st.expander("Count of each Harm per Moderation decision", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
                #change_label_style("Count of each Harm per Moderation decision", font_size='30px')
                st.pyplot(fig12)

        #with col2:
           # with st.expander("Count of each content type per Moderation decision", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
             #   change_label_style("Count of each content type per Moderation decision", font_size='30px')
               # st.pyplot(fig13)

        with col1:
            with st.expander("Count for each content type per automation status", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
              #  change_label_style("Count for each content type per automation status", font_size='30px')
                st.pyplot(fig11)

        with col2:
            with st.expander("Count for each content type per automation status Normalized", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
               # change_label_style("Count for each content type per automation status Normalized", font_size='30px')
                st.pyplot(fig11two)
  
        
       # with col1:
         #   with st.expander("Count of moderation decision per automation status", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
                #change_label_style("Count of moderation decision per automation status", font_size='30px')
              #  st.pyplot(fig14)
        with col2:
            with st.expander("Number of reported moderation decision per company", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
               # change_label_style("Number of reported moderation decision per company", font_size='30px')
                st.pyplot(fig15)


    elif selected_company and selected_harm:
        st.markdown("---")
        st.subheader("Analysis for company and harm")

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
            with st.expander("Total number of Moderation actions for selected harm and company", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
             #   change_label_style("Total number of Moderation actions per harm", font_size='30px')
                st.pyplot(figtest)


       # with col1:
           # with st.expander("Total number of Moderation Actions  for selected harm and company", expanded=False):
              #  change_label_style("Total number of Moderation Actions per Company", font_size='30px')
               # st.pyplot(fig0)

        with col2:
            with st.expander("Total number of Moderation Actions normalized  for selected harm and company", expanded=False):
              #  change_label_style("Total number of Moderation Actions per Company", font_size='30px')
                st.pyplot(fig0two)
                   

        with col2:
            with st.expander("Total number of Moderation Actions per Type of Content for selected harm and company", expanded=False):
              #  change_label_style("Total number of Moderation Actions per Type of Content", font_size='30px')
                #st.image(fig_to_png(fig3), use_column_width=True, width = 500)
                st.pyplot(fig1)

        with col1:
            with st.expander("Total number of Moderation Actions per Type of moderation action for selected harm and company", expanded=False):
               # st.image(fig_to_png(fig4), use_column_width=True, width = 500)
             #  change_label_style("Total number of Moderation Actions per Type of Automation Status", font_size='30px')
               st.pyplot(fig2)

        with col2:
            with st.expander("Total number of Moderation Actions per Type of Moderation Decision for selected harm and company", expanded=False):
              #  change_label_style("Total number of Moderation Actions per Type of Moderation Decision", font_size='30px')
                #st.image(fig_to_png(fig5), use_column_width=True, width = 500)
                st.pyplot(fig3)

        with col1:
            with st.expander("Number of reported Harms for selected harm and company", expanded=False):
             #   change_label_style("Number of reported Harms per Company", font_size='30px')
                st.pyplot(fig4)

        with col2:
            with st.expander("Number of reported content type  for selected harm and company", expanded=False):
               # change_label_style("Number of reported content type per Company", font_size='30px')
               # st.image(fig_to_png(fig8), use_column_width=True, width = 1100)
                st.pyplot(fig5)

        with col1:
            with st.expander("Number of Automation Status type for selected harm and company", expanded=False):
             #   change_label_style("Number of Automation Status type per Company", font_size='30px')
                #st.image(fig_to_png(fig10), use_column_width=True, width = 1100)
                st.pyplot(fig6)


        with col2:
            with st.expander("Normalized counts of each automation status for selected harm and company", expanded=False):
              #  st.image(fig_to_png(fig9), use_column_width=True, width = 1100)
              #  change_label_style("Normalized counts of each automation status per company", font_size='30px')
                st.pyplot(fig7)

      #  with col1:
       #     with st.expander("Number of reported content type for selected harm and company", expanded=False):
              #  change_label_style("Number of reported content type per Harm", font_size='30px')
              #  st.pyplot(fig8)


        with col2:
            with st.expander("Number of reported content type normalized for selected harm and company", expanded=False):
               # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
             #   change_label_style("Count for each harm per content type", font_size='30px')
                st.pyplot(fig9)

        # with col2:
        #     with st.expander("Count for each harm per content type normalized for selected harm and company", expanded=False):
        #        # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
        #      #   change_label_style("Count for each harm per content type", font_size='30px')
        #         st.pyplot(fig9two)
                
       # with col1:
           # with st.expander("Count for each harm per automation status for selected harm and company", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
              ##  change_label_style("Count for each harm per automation status", font_size='30px')
              #  st.pyplot(fig10)
        
        with col2:
            with st.expander("Count for each harm per automation status normalized for selected harm and company", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
              ##  change_label_style("Count for each harm per automation status", font_size='30px')
                st.pyplot(fig10two)

        with col2:
            with st.expander("Count for each content type per automation status for selected harm and company", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
              #  change_label_style("Count for each content type per automation status", font_size='30px')
                st.pyplot(fig11)

        with col1:
            with st.expander("Count for each content type per automation status normalized for selected harm and company", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
              #  change_label_style("Count for each content type per automation status", font_size='30px')
                st.pyplot(fig11two)
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
        with col1:
            with st.expander("Count of moderation decision per automation status for selected harm and company", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
               # change_label_style("Count of moderation decision per automation status", font_size='30px')
                st.pyplot(fig14)
        with col1:
            with st.expander("Number of reported moderation decision for selected harm and company", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
               # change_label_style("Number of reported moderation decision per company", font_size='30px')
                st.pyplot(fig15)



   


    #fix last graph and rename expander titles
    elif selected_company:
        st.markdown("---")
        st.subheader("Analysis for Company Data")

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
            with st.expander("Total number of Moderation actions per harm", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
               # change_label_style("Total number of Moderation actions per harm", font_size='30px')
                st.pyplot(figtest)


        with col2:
            with st.expander("Total number of Moderation Actions per Type of Automation Status", expanded=False):
               # st.image(fig_to_png(fig4), use_column_width=True, width = 500)
              # change_label_style("Total number of Moderation Actions per Type of Automation Status", font_size='30px')
               st.pyplot(fig3)


        with col1:
            with st.expander("Total number of Moderation Actions per Company", expanded=False):
               # change_label_style("Total number of Moderation Actions per Company", font_size='30px')
                st.pyplot(fig0)

        with col2:
            with st.expander("Total number of Moderation Actions per Company normalized", expanded=False):
               # change_label_style("Total number of Moderation Actions per Company", font_size='30px')
                st.pyplot(fig0two)
                

        with col2:
            with st.expander("Total number of Moderation Actions per Type of Moderation Decision", expanded=False):
              #  change_label_style("Total number of Moderation Actions per Type of Moderation Decision", font_size='30px')
                #st.image(fig_to_png(fig5), use_column_width=True, width = 500)
                st.pyplot(fig2)

        with col1:
            with st.expander("Number of reported Harms per Company", expanded=False):
             #   change_label_style("Number of reported Harms per Company", font_size='30px')
                st.pyplot(fig4)

        with col2:
            with st.expander("Number of reported content type  per Company", expanded=False):
             #   change_label_style("Number of reported content type per Company", font_size='30px')
               # st.image(fig_to_png(fig8), use_column_width=True, width = 1100)
                st.pyplot(fig5)

        with col1:
            with st.expander("Number of Automation Status type per Company", expanded=False):
             #   change_label_style("Number of Automation Status type per Company", font_size='30px')
                #st.image(fig_to_png(fig10), use_column_width=True, width = 1100)
                st.pyplot(fig6)


        with col2:
            with st.expander("Normalized counts of each automation status per company", expanded=False):
              #  st.image(fig_to_png(fig9), use_column_width=True, width = 1100)
              #  change_label_style("Normalized counts of each automation status per company", font_size='30px')
                st.pyplot(fig7)

        # with col1:
        #     with st.expander("Number of reported content type per Harm", expanded=False):
        #      #   change_label_style("Number of reported content type per Harm", font_size='30px')
        #         st.pyplot(fig8)


        with col1:
            with st.expander("Count for each harm per content type", expanded=False):
               # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
             #   change_label_style("Count for each harm per content type", font_size='30px')
                st.pyplot(fig9)

        with col2:
            with st.expander("Count for each harm per content type Normalized", expanded=False):
               # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
             #   change_label_style("Count for each harm per content type", font_size='30px')
                st.pyplot(fig9two)
                
        with col1:
            with st.expander("Count for each harm per automation status", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
               # change_label_style("Count for each harm per automation status", font_size='30px')
                st.pyplot(fig10)

        with col2:
            with st.expander("Count for each harm per automation status normalized", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
               # change_label_style("Count for each harm per automation status", font_size='30px')
                st.pyplot(fig10two)

        with col1:
            with st.expander("Count for each content type per automation status", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
             #   change_label_style("Count for each content type per automation status", font_size='30px')
                st.pyplot(fig11)

        with col2:
            with st.expander("Count for each content type per automation status Normalized", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
             #   change_label_style("Count for each content type per automation status", font_size='30px')
                st.pyplot(fig11two)
        # with col1:
        #     with st.expander("Count of each Harm per Moderation decision", expanded=False):
        #         # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
        #     #    change_label_style("Count of each Harm per Moderation decision", font_size='30px')
        #         st.pyplot(fig12)
       # with col2:
        #    with st.expander("Count of each content type per Moderation decision", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
             #   change_label_style("Count of each content type per Moderation decision", font_size='30px')
          #      st.pyplot(fig13)
        with col1:
            with st.expander("Count of moderation decision per automation status", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
               # change_label_style("Count of moderation decision per automation status", font_size='30px')
                st.pyplot(fig14)
        with col1:
            with st.expander("Number of reported moderation decision per company", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
               # change_label_style("Number of reported moderation decision per company", font_size='30px')
                st.pyplot(fig15)




    elif selected_harm:
        st.markdown("---")
        st.subheader("Analysis for Harm Data")

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
            with st.expander("Total number of Moderation actions per harm for harm", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
                #change_label_style("Total number of Moderation actions per harm", font_size='30px')
                st.pyplot(figtest)


     #   with col1:
      #      with st.expander("Total number of Moderation Actions per Company for harm", expanded=False):
               # change_label_style("Total number of Moderation Actions per Company", font_size='30px')
      #          st.pyplot(fig0)

     #   with col2:
       #     with st.expander("Total number of Moderation Actions per Company normalized for harm", expanded=False):
               # change_label_style("Total number of Moderation Actions per Company", font_size='30px')
           #     st.pyplot(fig0two)
                   

        with col2:
            with st.expander("Total number of Moderation Actions per Type of Content for harm", expanded=False):
               # change_label_style("Total number of Moderation Actions per Type of Content", font_size='30px')
                #st.image(fig_to_png(fig3), use_column_width=True, width = 500)
                st.pyplot(fig1)

        with col1:
            with st.expander("Total number of Moderation Actions per Type of Automation Status for harm", expanded=False):
               # st.image(fig_to_png(fig4), use_column_width=True, width = 500)
              # change_label_style("Total number of Moderation Actions per Type of Automation Status", font_size='30px')
               st.pyplot(fig2)

        with col2:
            with st.expander("Total number of Automation status for harm", expanded=False):
             #   change_label_style("Total number of Moderation Actions per Type of Moderation Decision", font_size='30px')
                #st.image(fig_to_png(fig5), use_column_width=True, width = 500)
                st.pyplot(fig3)

        with col1:
            with st.expander("Number of reported Harms per Company for harm", expanded=False):
              #  change_label_style("Number of reported Harms per Company", font_size='30px')
                st.pyplot(fig4)

       # with col2:
       #     with st.expander("Number of reported content type per Company for harm", expanded=False):
             #   change_label_style("Number of reported content type per Company", font_size='30px')
               # st.image(fig_to_png(fig8), use_column_width=True, width = 1100)
        #        st.pyplot(fig5)

        with col1:
            with st.expander("Number of Automation Status type per Company for harm", expanded=False):
               # change_label_style("Number of Automation Status type per Company", font_size='30px')
                #st.image(fig_to_png(fig10), use_column_width=True, width = 1100)
                st.pyplot(fig6)


        with col2:
            with st.expander("Normalized counts of each automation status per company for harm", expanded=False):
              #  st.image(fig_to_png(fig9), use_column_width=True, width = 1100)
              #  change_label_style("Normalized counts of each automation status per company", font_size='30px')
                st.pyplot(fig7)

      #  with col1:
        #    with st.expander("Number of reported content type per Harm for harm", expanded=False):
               # change_label_style("Number of reported content type per Harm", font_size='30px')
         #       st.pyplot(fig8)


        with col2:
            with st.expander("Number of reported content type per Harm Normalized for harm", expanded=False):
               # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
               # change_label_style("Count for each harm per content type", font_size='30px')
                st.pyplot(fig9)
                
        with col1:
            with st.expander("Count for each harm per automation status for harm", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
               # change_label_style("Count for each harm per automation status", font_size='30px')
                st.pyplot(fig10)

        with col2:
            with st.expander("Count for each harm per automation status normalized for harm", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
               # change_label_style("Count for each harm per automation status", font_size='30px')
                st.pyplot(fig10two)

        with col1:
            with st.expander("Count for each content type per automation status for harm", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
               # change_label_style("Count for each content type per automation status", font_size='30px')
                st.pyplot(fig11)

        with col2:
            with st.expander("Count for each content type per automation status normalized for harm", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
               # change_label_style("Count for each content type per automation status", font_size='30px')
                st.pyplot(fig11two)

        # with col1:
        #     with st.expander("Count of each Harm per Moderation decision for harm", expanded=False):
        #             st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
        #       #  change_label_style("Count of each Harm per Moderation decision", font_size='30px')
        #          st.pyplot(fig12)
        # with col2:
        #     with st.expander("Count of each content type per Moderation decision for harm", expanded=False):
        #         # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
        #        # change_label_style("Count of each content type per Moderation decision", font_size='30px')
        #         st.pyplot(fig13)
        with col1:
            with st.expander("Count of moderation decision per automation status for harm", expanded=False):
                # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
              #  change_label_style("Count of moderation decision per automation status", font_size='30px')
                st.pyplot(fig14)
        # with col2:
        #     with st.expander("Number of reported moderation decision per company for harm", expanded=False):
        #         # st.image(fig_to_png(fig11), use_column_width=True, width = 1100)
        #         #change_label_style("Number of reported moderation decision per company", font_size='30px')
        #         st.pyplot(fig15)

if __name__ == "__main__":
    main()
