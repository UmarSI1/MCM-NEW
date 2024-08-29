import pickle
import zipfile
from pathlib import Path
from collections import defaultdict
import os
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import textwrap
import matplotlib.ticker as mtick
# file_path = 'data_ACC.pkl'

# # Open the file in binary read mode
# with open(file_path, 'rb') as f:
#     # Load the pickled data back into a dictionary variable
#     data_ACC = pickle.load(f)

# # Assuming data_ACC is the loaded dictionary from the pickle file
# List_of_companies = list(data_ACC.keys())
# harm_dic = data_ACC[List_of_companies[0]]
# List_of_harms = list(harm_dic.keys())
#a
# content_dic = harm_dic[List_of_harms[0]]
# List_of_content_type = list(content_dic.keys())
# action_dic = content_dic[List_of_content_type[0]]
# List_of_moderation_action = list(action_dic.keys())
# automation_dic = action_dic[List_of_moderation_action[0]]
# List_of_automation_status = list(automation_dic.keys())


###########################################################################################


def plot_acc_totals_harm(data, harm):
    """ Sum all numbers for each ACC automation detection status """
    automation_detection_totals = {}
    total_sum_all_automation_detection = 0  # Initialize the total sum for all automation statuses

    # Sum all numbers for each automation status across all companies
    for company in data.values():
        for content_type in company[harm].values():
                for moderation_action in content_type.values():
                    for automation_status in moderation_action.values():
                        for automation_detection, count in automation_status.items():
                            
                            if automation_detection not in automation_detection_totals:
                                 
                                automation_detection_totals[automation_detection] = 0
                        
                            automation_detection_totals[automation_detection] += count
                            total_sum_all_automation_detection += count  # Accumulate the total sum

    # Add the total to the data after the loop
    automation_detection_totals['Total'] = total_sum_all_automation_detection  # Append the total sum

    # Create a DataFrame for the automation status totals
    automation_detection_data = {'Automation Detection': list(automation_detection_totals.keys()),
                              'N# Actions': list(automation_detection_totals.values())}
    df_automation_detection = pd.DataFrame(automation_detection_data).dropna()

    # Define automation decision descriptions for mapping
    automated_detection_cleaned = {
        'Yes': 'ACC flag',
        'No': 'User flag'
    }

    # Rename the automation statuses in the DataFrame
    df_automation_detection['Automation Detection'] = df_automation_detection['Automation Detection'].map(automated_detection_cleaned).fillna('Total')

    # Plotting the table
    fig3, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')

    # Create the table
    table = ax.table(cellText=df_automation_detection.values, colLabels=df_automation_detection.columns, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])

    formatter = mtick.FuncFormatter(lambda x, _: f'{x:,.0f}')
    for i in range(1, len(df_automation_detection.columns)):
        for key, cell in table.get_celld().items():
            if key[0] != 0 and key[1] == i:  # Exclude the header row
              #  cell.get_text().set_text(formatter(int(cell.get_text().get_text())))
                text = cell.get_text().get_text()
                try:
                    value = int(float(text))
                    cell.get_text().set_text(formatter(value))
                except ValueError:
                    pass  # or handle the error as needed


    # Set font size
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Display the table
    plt.show()

    #return fig3
    return df_automation_detection

######### Data General plot - Number of reported content type  per Company
def plot_acc_totals_per_company_harm(data, harm):
    """ Sum all numbers for acc per company and plot the results as a table. """
    acc_totals_per_company = {company: {} for company in data.keys()}
    total_acc_totals = {company: 0 for company in data.keys()}  # Initialize total sum for each company

    # Sum all numbers for each content type per company
    for company, harms in data.items():
            for content_data in harms[harm].values():
                for moderation_action in content_data.values():
                    for automation_status in moderation_action.values():
                        for acc, automation_detection in automation_status.items():
                            if acc not in acc_totals_per_company[company]:
                                acc_totals_per_company[company][acc] = 0

                            acc_totals_per_company[company][acc] += automation_detection
                            total_acc_totals[company] += automation_detection  # Accumulate the total sum for each company

    # Prepare data for DataFrame
    data_for_df = {'Company': [], 'ACC': [], 'N# Actions': []}
    for company, automation_detection in acc_totals_per_company.items():
        for content_type, total_actions in automation_detection.items():
            data_for_df['Company'].append(company)
            data_for_df['ACC'].append(content_type)
            data_for_df['N# Actions'].append(int(float(total_actions)))  # Convert to float first, then to int
        # Add total for each company
      #  data_for_df['Company'].append(company)
     #   data_for_df['ACC'].append('Total')
       # data_for_df['N# Actions'].append(total_acc_totals[company])

    df_content_type_per_company = pd.DataFrame(data_for_df).dropna()

    # Define content type descriptions for mapping
    acc_cleaned = {
        'Yes': 'ACC flag',
        'No': 'User flag'
    }

    # Rename the content type categories in the DataFrame
    df_content_type_per_company['ACC'] = df_content_type_per_company['ACC'].map(acc_cleaned) #.fillna('Total')

    # Handle duplicate entries by summing them
    df_content_type_per_company = df_content_type_per_company.groupby(['Company', 'ACC'], as_index=False).sum()



    # Pivot the DataFrame
    pivot_df = df_content_type_per_company.pivot(index='Company', columns='ACC', values='N# Actions').fillna(0).reset_index()

    

    wrapped_columns = [textwrap.fill(col, width=15) for col in pivot_df.columns]

    # Plotting the table
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjust the figsize as needed
    ax.axis('tight')
    ax.axis('off')
    ax.set_position([0, 0, 3, 3])

    # Create the table
    table = ax.table(cellText=pivot_df.values, colLabels=wrapped_columns, cellLoc='center', loc='center',  bbox=[0, 0, 1, 1])

    # Set cell height
    table.auto_set_column_width(col=list(range(len(pivot_df.columns))))
    table.scale(1, 3)  # Increase cell height (adjust as needed)

    # Set font size
    table.auto_set_font_size(False)
    table.set_fontsize(27)

    formatter = mtick.FuncFormatter(lambda x, _: f'{x:,.0f}')
    for i in range(1, len(pivot_df.columns)):
        for key, cell in table.get_celld().items():
            if key[0] != 0 and key[1] == i:  # Exclude the header row
                cell.get_text().set_text(formatter(int(float(cell.get_text().get_text()))))  # Convert to float first, then to int

    # Set cell height
    table.auto_set_column_width(col=list(range(len(pivot_df.columns))))
    table.scale(1, 1.5)  # Increase cell height (adjust as needed)

    # Set font size
    table.auto_set_font_size(False)
    table.set_fontsize(22)

    # Display the table
    plt.show()

    #return fig
    return pivot_df

##########################################################################################################################

def plot_acc_totals_per_harm_harm(data, harm):
    """ Sum all numbers for acc per harm and plot the results as a table. """
    acc_totals_per_harm = {}

    for company_data in data.values():
        for content_type_data in company_data[harm].values():
                for moderation_action in content_type_data.values():
                    for automation_status in moderation_action.values():
                        for acc, automation_detection in automation_status.items():
                            if (harm, acc) not in acc_totals_per_harm:
                                acc_totals_per_harm[(harm, acc)] = 0
                            if pd.notna(automation_detection):  # Check if the count is not NaN
                                acc_totals_per_harm[(harm, acc)] += automation_detection


                         

# Prepare data for DataFrame
    data_for_df = {'Harm': [], 'ACC': [], 'N# Actions': []}
    for (harm, acc), total_actions in acc_totals_per_harm.items():
        data_for_df['Harm'].append(harm)
        data_for_df['ACC'].append(acc)
        data_for_df['N# Actions'].append(int(total_actions))

    df_content_type_per_company = pd.DataFrame(data_for_df).dropna()

    # Define content type descriptions for mapping
    acc_cleaned = {
        'Yes': 'ACC flag',
        'No': 'User flag'
    }

    category_descriptions = {
        'STATEMENT_CATEGORY_SCOPE_OF_PLATFORM_SERVICE': 'PLATFORM SCOPE',
        'STATEMENT_CATEGORY_DATA_PROTECTION_AND_PRIVACY_VIOLATIONS': 'GDPR VIOLATION',
        'STATEMENT_CATEGORY_PORNOGRAPHY_OR_SEXUALIZED_CONTENT': 'PORN/SEX CONTENT',
        'STATEMENT_CATEGORY_ILLEGAL_OR_HARMFUL_SPEECH': 'ILLEGAL/HARMFULL SPEECH',
        'STATEMENT_CATEGORY_VIOLENCE': 'VIOLENCE',
        'STATEMENT_CATEGORY_SCAMS_AND_FRAUD': 'SCAMS/FRAUD',
        'STATEMENT_CATEGORY_UNSAFE_AND_ILLEGAL_PRODUCTS': 'ILLEGAL PRODUCTS',
        'STATEMENT_CATEGORY_NON_CONSENSUAL_BEHAVIOUR': 'NON CONSENSUAL BEHAVIOUR',
        'STATEMENT_CATEGORY_PROTECTION_OF_MINORS': 'PROTECT MINORS',
        'STATEMENT_CATEGORY_INTELLECTUAL_PROPERTY_INFRINGEMENTS': 'COPYRIGHT',
        'STATEMENT_CATEGORY_NEGATIVE_EFFECTS_ON_CIVIC_DISCOURSE_OR_ELECTIONS': 'NEGATIVE EFFECTS ELECTIONS',
        'STATEMENT_CATEGORY_RISK_FOR_PUBLIC_SECURITY': 'RISK PUBLIC SECURITY',
        'STATEMENT_CATEGORY_ANIMAL_WELFARE': 'ANIMAL WELFARE',
        'STATEMENT_CATEGORY_SELF_HARM': 'SELF HARM'
    }

    # Rename the content type categories in the DataFrame
    df_content_type_per_company['ACC'] = df_content_type_per_company['ACC'].map(acc_cleaned).fillna('ACC')
    df_content_type_per_company['Harm'] = df_content_type_per_company['Harm'].map(category_descriptions)

    # Handle duplicate entries by summing them
    df_content_type_per_company = df_content_type_per_company.groupby(['Harm', 'ACC'], as_index=False).sum()



    # Pivot the DataFrame
    pivot_df = df_content_type_per_company.pivot(index='Harm', columns='ACC', values='N# Actions').fillna(0).reset_index()

    

    wrapped_columns = [textwrap.fill(col, width=15) for col in pivot_df.columns]

    # Plotting the table
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjust the figsize as needed
    ax.axis('tight')
    ax.axis('off')
    ax.set_position([0, 0, 3, 3])

    # Create the table
    table = ax.table(cellText=pivot_df.values, colLabels=wrapped_columns, cellLoc='center', loc='center',  bbox=[0, 0, 1, 1])

    # Set cell height
    table.auto_set_column_width(col=list(range(len(pivot_df.columns))))
    table.scale(1, 3)  # Increase cell height (adjust as needed)

    # Set font size
    table.auto_set_font_size(False)
    table.set_fontsize(27)

    formatter = mtick.FuncFormatter(lambda x, _: f'{x:,.0f}')
    for i in range(1, len(pivot_df.columns)):
        for key, cell in table.get_celld().items():
            if key[0] != 0 and key[1] == i:  # Exclude the header row
                cell.get_text().set_text(formatter(int(float(cell.get_text().get_text()))))  # Convert to float first, then to int

    # Set cell height
    table.auto_set_column_width(col=list(range(len(pivot_df.columns))))
    table.scale(1, 1.5)  # Increase cell height (adjust as needed)

    # Set font size
    table.auto_set_font_size(False)
    table.set_fontsize(22)

    # Display the table
    plt.show()

    #return fig
    return pivot_df




##########################################################################################################################

def plot_acc_totals_per_moderation_action_harm(data, harm):
    """ Sum all numbers for acc per harm and plot the results as a table. """
    acc_totals_per_action = {}

    for company_data in data.values():
        for content_type_data in company_data[harm].values():
                for action, moderation_action in content_type_data.items():
                    for automation_status in moderation_action.values():
                        for acc, automation_detection in automation_status.items():
                            if (action, acc) not in acc_totals_per_action:
                                acc_totals_per_action[(action, acc)] = 0
                            if pd.notna(automation_detection):  # Check if the count is not NaN
                                acc_totals_per_action[(action, acc)] += automation_detection


                         

# Prepare data for DataFrame
    data_for_df = {'Action': [], 'ACC': [], 'N# Actions': []}
    for (action, acc), total_actions in acc_totals_per_action.items():
        data_for_df['Action'].append(action)
        data_for_df['ACC'].append(acc)
        data_for_df['N# Actions'].append(int(total_actions))

    df_content_type_per_company = pd.DataFrame(data_for_df).dropna()

    # Define content type descriptions for mapping
    acc_cleaned = {
        'Yes': 'ACC flag',
        'No': 'User flag'
    }

    category_descriptions = {
        '["DECISION_VISIBILITY_CONTENT_REMOVED"]': 'REMOVED',
        '["DECISION_VISIBILITY_CONTENT_INTERACTION_RESTRICTED","DECISION_VISIBILITY_CONTENT_REMOVED"]': 'RESTRICTED/REMOVED',
        '["DECISION_VISIBILITY_CONTENT_REMOVED","DECISION_VISIBILITY_CONTENT_AGE_RESTRICTED"]': 'REMOVED/AGE RESTRICTED',
        '["DECISION_VISIBILITY_OTHER","DECISION_VISIBILITY_CONTENT_AGE_RESTRICTED"]': 'OTHER/AGE RESTRICTED',
        '["DECISION_VISIBILITY_CONTENT_LABELLED"]': 'LABELLED',
        '["DECISION_VISIBILITY_OTHER"]': 'OTHER',
        '["DECISION_VISIBILITY_CONTENT_AGE_RESTRICTED"]': 'AGE RESTRICTED',
        '["DECISION_VISIBILITY_CONTENT_DISABLED"]': 'DISABLED',
        '["DECISION_VISIBILITY_CONTENT_INTERACTION_RESTRICTED"]': 'INTERACTION RESTRICTED',
        '[]': 'NOT DEFINED',
        '["DECISION_VISIBILITY_CONTENT_DEMOTED"]': 'DEMOTED'
    }

    # Rename the content type categories in the DataFrame
    df_content_type_per_company['ACC'] = df_content_type_per_company['ACC'].map(acc_cleaned).fillna('ACC')
    df_content_type_per_company['Action'] = df_content_type_per_company['Action'].map(category_descriptions)

    # Handle duplicate entries by summing them
    df_content_type_per_company = df_content_type_per_company.groupby(['Action', 'ACC'], as_index=False).sum()



    # Pivot the DataFrame
    pivot_df = df_content_type_per_company.pivot(index='Action', columns='ACC', values='N# Actions').fillna(0).reset_index()

    

    wrapped_columns = [textwrap.fill(col, width=15) for col in pivot_df.columns]

    # Plotting the table
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjust the figsize as needed
    ax.axis('tight')
    ax.axis('off')
    ax.set_position([0, 0, 3, 3])

    # Create the table
    table = ax.table(cellText=pivot_df.values, colLabels=wrapped_columns, cellLoc='center', loc='center',  bbox=[0, 0, 1, 1])

    # Set cell height
    table.auto_set_column_width(col=list(range(len(pivot_df.columns))))
    table.scale(1, 3)  # Increase cell height (adjust as needed)

    # Set font size
    table.auto_set_font_size(False)
    table.set_fontsize(27)

    formatter = mtick.FuncFormatter(lambda x, _: f'{x:,.0f}')
    for i in range(1, len(pivot_df.columns)):
        for key, cell in table.get_celld().items():
            if key[0] != 0 and key[1] == i:  # Exclude the header row
                cell.get_text().set_text(formatter(int(float(cell.get_text().get_text()))))  # Convert to float first, then to int

    # Set cell height
    table.auto_set_column_width(col=list(range(len(pivot_df.columns))))
    table.scale(1, 1.5)  # Increase cell height (adjust as needed)

    # Set font size
    table.auto_set_font_size(False)
    table.set_fontsize(22)

    # Display the table
    plt.show()

    #return fig
    return pivot_df



##########################################################################################################################

def plot_acc_totals_per_automation_status_harm(data, harm):
    """ Sum all numbers for acc per harm and plot the results as a table. """
    acc_totals_per_status = {}

    for company_data in data.values():
        for content_type_data in company_data[harm].values():
                for  moderation_action in content_type_data.values():
                    for status, automation_status in moderation_action.items():
                        for acc, automation_detection in automation_status.items():
                            if (status, acc) not in acc_totals_per_status:
                                acc_totals_per_status[(status, acc)] = 0
                            if pd.notna(automation_detection):  # Check if the count is not NaN
                                acc_totals_per_status[(status, acc)] += automation_detection


                         

# Prepare data for DataFrame
    data_for_df = {'Status': [], 'ACC': [], 'N# Actions': []}
    for (status, acc), total_actions in acc_totals_per_status.items():
        data_for_df['Status'].append(status)
        data_for_df['ACC'].append(acc)
        data_for_df['N# Actions'].append(int(total_actions))

    df_content_type_per_company = pd.DataFrame(data_for_df).dropna()

    # Define content type descriptions for mapping
    acc_cleaned = {
        'Yes': 'ACC flag',
        'No': 'User flag'
    }

    category_descriptions = {

        'AUTOMATED_DECISION_FULLY': 'Fully Automated',
        'AUTOMATED_DECISION_NOT_AUTOMATED': 'Not Automated',
        'AUTOMATED_DECISION_PARTIALLY': 'Partially Automated'

    }

    # Rename the content type categories in the DataFrame
    df_content_type_per_company['ACC'] = df_content_type_per_company['ACC'].map(acc_cleaned).fillna('ACC')
    df_content_type_per_company['Status'] = df_content_type_per_company['Status'].map(category_descriptions)

    # Handle duplicate entries by summing them
    df_content_type_per_company = df_content_type_per_company.groupby(['Status', 'ACC'], as_index=False).sum()



    # Pivot the DataFrame
    pivot_df = df_content_type_per_company.pivot(index='Status', columns='ACC', values='N# Actions').fillna(0).reset_index()

    

    wrapped_columns = [textwrap.fill(col, width=15) for col in pivot_df.columns]

    # Plotting the table
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjust the figsize as needed
    ax.axis('tight')
    ax.axis('off')
    ax.set_position([0, 0, 3, 3])

    # Create the table
    table = ax.table(cellText=pivot_df.values, colLabels=wrapped_columns, cellLoc='center', loc='center',  bbox=[0, 0, 1, 1])

    # Set cell height
    table.auto_set_column_width(col=list(range(len(pivot_df.columns))))
    table.scale(1, 3)  # Increase cell height (adjust as needed)

    # Set font size
    table.auto_set_font_size(False)
    table.set_fontsize(27)

    formatter = mtick.FuncFormatter(lambda x, _: f'{x:,.0f}')
    for i in range(1, len(pivot_df.columns)):
        for key, cell in table.get_celld().items():
            if key[0] != 0 and key[1] == i:  # Exclude the header row
                cell.get_text().set_text(formatter(int(float(cell.get_text().get_text()))))  # Convert to float first, then to int

    # Set cell height
    table.auto_set_column_width(col=list(range(len(pivot_df.columns))))
    table.scale(1, 1.5)  # Increase cell height (adjust as needed)

    # Set font size
    table.auto_set_font_size(False)
    table.set_fontsize(22)

    # Display the table
    plt.show()

    #return fig
    return pivot_df


##########################################################################################################################

def plot_acc_totals_per_content_type_harm(data, harm):
    """ Sum all numbers for acc per harm and plot the results as a table. """
    acc_totals_per_content = {}

    for company_data in data.values():
            for content,content_type_data in company_data[harm].items():
                for  moderation_action in content_type_data.values():
                    for automation_status in moderation_action.values():
                        for acc, automation_detection in automation_status.items():
                            if (content, acc) not in acc_totals_per_content:
                                acc_totals_per_content[(content, acc)] = 0
                            if pd.notna(automation_detection):  # Check if the count is not NaN
                                acc_totals_per_content[(content, acc)] += automation_detection


                         

# Prepare data for DataFrame
    data_for_df = {'Content': [], 'ACC': [], 'N# Actions': []}
    for (content, acc), total_actions in acc_totals_per_content.items():
        data_for_df['Content'].append(content)
        data_for_df['ACC'].append(acc)
        data_for_df['N# Actions'].append(int(total_actions))

    df_content_type_per_company = pd.DataFrame(data_for_df).dropna()

    # Define content type descriptions for mapping
    acc_cleaned = {
        'Yes': 'ACC flag',
        'No': 'User flag'
    }

    category_descriptions = {
        
        '["CONTENT_TYPE_OTHER"]': 'OTHER',
        '["CONTENT_TYPE_SYNTHETIC_MEDIA"]': 'SYNTHETIC MEDIA',
        '["CONTENT_TYPE_IMAGE"]': 'IMAGE',
        '["CONTENT_TYPE_TEXT"]': 'TEXT',
        '["CONTENT_TYPE_VIDEO"]': 'VIDEO',
        '["CONTENT_TYPE_PRODUCT"]': 'PRODUCT',
        '["CONTENT_TYPE_APP"]': 'APP',
        '["CONTENT_TYPE_AUDIO"]': 'AUDIO',
        '["CONTENT_TYPE_IMAGE","CONTENT_TYPE_TEXT"]': 'IMAGE/TEXT',
        '["CONTENT_TYPE_IMAGE","CONTENT_TYPE_TEXT","CONTENT_TYPE_VIDEO"]': 'IMAGE/TEXT/VIDEO',
        '["CONTENT_TYPE_AUDIO","CONTENT_TYPE_TEXT"],"CONTENT_TYPE_VIDEO"]': 'AUDIO/TEXT/VIDEO',
        '["CONTENT_TYPE_AUDIO","CONTENT_TYPE_IMAGE","CONTENT_TYPE_TEXT","CONTENT_TYPE_VIDEO"]': 'AUDIO/TEXT/VIDEO/IMAGE',
        '["CONTENT_TYPE_AUDIO","CONTENT_TYPE_IMAGE","CONTENT_TYPE_VIDEO"]': 'AUDIO/IMAGE/VIDEO'

    }

    # Rename the content type categories in the DataFrame
    df_content_type_per_company['ACC'] = df_content_type_per_company['ACC'].map(acc_cleaned).fillna('ACC')
    df_content_type_per_company['Content'] = df_content_type_per_company['Content'].map(category_descriptions)

    # Handle duplicate entries by summing them
    df_content_type_per_company = df_content_type_per_company.groupby(['Content', 'ACC'], as_index=False).sum()



    # Pivot the DataFrame
    pivot_df = df_content_type_per_company.pivot(index='Content', columns='ACC', values='N# Actions').fillna(0).reset_index()

    

    wrapped_columns = [textwrap.fill(col, width=15) for col in pivot_df.columns]

    # Plotting the table
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjust the figsize as needed
    ax.axis('tight')
    ax.axis('off')
    ax.set_position([0, 0, 3, 3])

    # Create the table
    table = ax.table(cellText=pivot_df.values, colLabels=wrapped_columns, cellLoc='center', loc='center',  bbox=[0, 0, 1, 1])

    # Set cell height
    table.auto_set_column_width(col=list(range(len(pivot_df.columns))))
    table.scale(1, 3)  # Increase cell height (adjust as needed)

    # Set font size
    table.auto_set_font_size(False)
    table.set_fontsize(27)

    formatter = mtick.FuncFormatter(lambda x, _: f'{x:,.0f}')
    for i in range(1, len(pivot_df.columns)):
        for key, cell in table.get_celld().items():
            if key[0] != 0 and key[1] == i:  # Exclude the header row
                cell.get_text().set_text(formatter(int(float(cell.get_text().get_text()))))  # Convert to float first, then to int

    # Set cell height
    table.auto_set_column_width(col=list(range(len(pivot_df.columns))))
    table.scale(1, 1.5)  # Increase cell height (adjust as needed)

    # Set font size
    table.auto_set_font_size(False)
    table.set_fontsize(22)

    # Display the table
    plt.show()

    #return fig
    return pivot_df


###########################################################################################
######## Data General plot - Total number of Moderation Actions per Company
def plot_company_dataxxz2(data, harm):
    """ Sum all numbers for a certain company key and plot the results as both a table and a bar chart. """
    def sum_company(data, company):
        """ Sum all numbers for a certain company key """
        total_sum = 0
        for content_type in data[company][harm].values():
            for moderation_action in content_type.values():
                for automation_status in moderation_action.values():
                    for automation_detection in automation_status.values():
                        total_sum += automation_detection
                    
        return total_sum

    company_data = {'Company': [], 'N# Actions': []}
    for company in List_of_companies:
        num_actions = sum_company(data, company)
        company_data['Company'].append(company)
        company_data['N# Actions'].append(num_actions)

    df_company = pd.DataFrame(company_data).dropna()

    # Plotting the table and graph
    fig, ax = plt.subplots(figsize=(5,5))

    # Table
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df_company.values, colLabels=df_company.columns, cellLoc='center', loc='center',  bbox=[0, 0, 1, 1])

    plt.tight_layout()
    return fig


def plot_company_dataxxz2_normalized(data, harm):
    """ Sum all numbers for a certain company key and plot the results as both a table and a bar chart. """
    def sum_company(data, company):
        """ Sum all numbers for a certain company key """
        total_sum = 0
        for content_type in data[company][harm].values():
            for moderation_action in content_type.values():
                for automation_status in moderation_action.values():
                    for automation_detection in automation_status.values():
                            total_sum += automation_detection
        return total_sum

    company_data = {'Company': [], 'N# Actions': []}
    for company in List_of_companies:
        num_actions = sum_company(data, company)
        company_data['Company'].append(company)
        company_data['N# Actions'].append(num_actions)

    df_company = pd.DataFrame(company_data).dropna()

    # Plotting the table and graph
    fig, ax = plt.subplots(figsize=(10, 12))

    # Bar chart
    df_company.plot(kind='bar', x='Company', y='N# Actions', ax=ax)
    ax.set_xlabel('Company')
    ax.set_ylabel('Total Number of Actions')
    ax.set_title('Total Number of Actions per Company')
    plt.xticks(rotation=45)

    plt.tight_layout()
    return fig




######### Data General plot - Total number of Moderation Actions per Harm

def sum_harm2(data_ACC, harm):
    category_descriptions = {
        'STATEMENT_CATEGORY_SCOPE_OF_PLATFORM_SERVICE': 'PLATFORM SCOPE',
        'STATEMENT_CATEGORY_DATA_PROTECTION_AND_PRIVACY_VIOLATIONS': 'GDPR VIOLATION',
        'STATEMENT_CATEGORY_PORNOGRAPHY_OR_SEXUALIZED_CONTENT': 'PORN/SEX CONTENT',
        'STATEMENT_CATEGORY_ILLEGAL_OR_HARMFUL_SPEECH': 'ILLEGAL/HARMFULL SPEECH',
        'STATEMENT_CATEGORY_VIOLENCE': 'VIOLENCE',
        'STATEMENT_CATEGORY_SCAMS_AND_FRAUD': 'SCAMS/FRAUD',
        'STATEMENT_CATEGORY_UNSAFE_AND_ILLEGAL_PRODUCTS': 'ILLEGAL PRODUCTS',
        'STATEMENT_CATEGORY_NON_CONSENSUAL_BEHAVIOUR': 'NON CONSENSUAL BEHAVIOUR',
        'STATEMENT_CATEGORY_PROTECTION_OF_MINORS': 'PROTECT MINORS',
        'STATEMENT_CATEGORY_INTELLECTUAL_PROPERTY_INFRINGEMENTS': 'COPYRIGHT',
        'STATEMENT_CATEGORY_NEGATIVE_EFFECTS_ON_CIVIC_DISCOURSE_OR_ELECTIONS': 'NEGATIVE EFFECTS ELECTIONS',
        'STATEMENT_CATEGORY_RISK_FOR_PUBLIC_SECURITY': 'RISK PUBLIC SECURITY',
        'STATEMENT_CATEGORY_ANIMAL_WELFARE': 'ANIMAL WELFARE',
        'STATEMENT_CATEGORY_SELF_HARM': 'SELF HARM'
    }
    
    # Sum all numbers for each harm type across all companies
    harm_totals = {}
    total_sum_all_harms = 0 
    for company in data_ACC.values():
        for content_type in company[harm].values():
            if harm not in harm_totals:
                    harm_totals[harm] = 0
            for moderation_action in content_type.values():
                for automation_status in moderation_action.values():
                    for automation_detection in automation_status.values():
                            harm_totals[harm] += automation_detection
                            total_sum_all_harms += automation_detection
    
    harm_data = {'Harm': list(harm_totals.keys()), 'N# Actions': list(harm_totals.values())}
    df_harm = pd.DataFrame(harm_data).dropna()

    # Renaming the harm categories in your DataFrame
    df_harm['Harm'] = df_harm['Harm'].map(category_descriptions)

    # Plotting the table graph
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')

    # Create the table
    table = ax.table(cellText=df_harm.values, colLabels=df_harm.columns, cellLoc='center', loc='center',  bbox=[0, 0, 1, 1])

    
    formatter = mtick.FuncFormatter(lambda x, _: f'{x:,.0f}')
    for i in range(1, len(df_harm.columns)):
        for key, cell in table.get_celld().items():
            if key[0] != 0 and key[1] == i:  # Exclude the header row
                cell.get_text().set_text(formatter(int(float(cell.get_text().get_text()))))  # Convert to float first, then to int

    # Set font size
    table.auto_set_font_size(False)
    table.set_fontsize(5)

    #return fig
    return df_harm


######### Data General plot - Total number of Moderation Actions per Type of Content

def plot_content_type_totals2(data, harm):
    """ Sum all numbers for each content type across all companies and plot the results as a table. """
    content_type_totals = {}
    total_sum_all_content_types = 0
    
    # Sum all numbers for each content type across all companies
    for company in data.values():
        #for harm in company.values():
        for content_type, content_data in company[harm].items():
            if content_type not in content_type_totals:
                content_type_totals[content_type] = 0
            for moderation_action in content_data.values():
                for automation_status in moderation_action.values():
                    for automation_detection in automation_status.values():
                            content_type_totals[content_type] += automation_detection
                            total_sum_all_content_types += automation_detection
    
    # Create a DataFrame for the content type totals

    # Add the total to the data after the loop
    content_type_totals['Total'] = total_sum_all_content_types  # Append the total sum

    content_type_data = {'Content Type': list(content_type_totals.keys()), 'N# Actions': list(content_type_totals.values())}
    df_content_type = pd.DataFrame(content_type_data).dropna()

    # Define content type descriptions for mapping
    content_type_descriptions = {
        '["CONTENT_TYPE_OTHER"]': 'OTHER',
        '["CONTENT_TYPE_SYNTHETIC_MEDIA"]': 'SYNTHETIC MEDIA',
        '["CONTENT_TYPE_IMAGE"]': 'IMAGE',
        '["CONTENT_TYPE_TEXT"]': 'TEXT',
        '["CONTENT_TYPE_VIDEO"]': 'VIDEO',
        '["CONTENT_TYPE_PRODUCT"]': 'PRODUCT',
        '["CONTENT_TYPE_APP"]': 'APP',
        '["CONTENT_TYPE_AUDIO"]': 'AUDIO',
        '["CONTENT_TYPE_IMAGE","CONTENT_TYPE_TEXT"]': 'IMAGE/TEXT',
        '["CONTENT_TYPE_IMAGE","CONTENT_TYPE_TEXT","CONTENT_TYPE_VIDEO"]': 'IMAGE/TEXT/VIDEO',
        '["CONTENT_TYPE_AUDIO","CONTENT_TYPE_TEXT"],"CONTENT_TYPE_VIDEO"]': 'AUDIO/TEXT/VIDEO',
        '["CONTENT_TYPE_AUDIO","CONTENT_TYPE_IMAGE","CONTENT_TYPE_TEXT","CONTENT_TYPE_VIDEO"]': 'AUDIO/TEXT/VIDEO/IMAGE',
        '["CONTENT_TYPE_AUDIO","CONTENT_TYPE_IMAGE","CONTENT_TYPE_VIDEO"]': 'AUDIO/IMAGE/VIDEO'
    }

    # Rename the content types in the DataFrame
    df_content_type['Content Type'] = df_content_type['Content Type'].map(content_type_descriptions)

    # Plotting the table
    fig1, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')

    # Create the table
    table = ax.table(cellText=df_content_type.values, colLabels=df_content_type.columns, cellLoc='center', loc='center',  bbox=[0, 0, 1, 1])


    formatter = mtick.FuncFormatter(lambda x, _: f'{x:,.0f}')
    for i in range(1, len(df_content_type.columns)):
        for key, cell in table.get_celld().items():
            if key[0] != 0 and key[1] == i:  # Exclude the header row
                cell.get_text().set_text(formatter(int(float(cell.get_text().get_text()))))  # Convert to float first, then to int

    # Set font size
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Display the table
    plt.show()

    #return fig1
    return df_content_type


######### Data General plot - Total number of Moderation Actions per Type of Moderation Actions

def plot_moderation_action_totals2(data, harm):
    """ Sum all numbers for each moderation action across all companies and plot the results as a table. """
    moderation_action_totals = {}
    total_sum_all_moderation_actions = 0
    
    # Sum all numbers for each moderation action across all companies
    for company in data.values():
        for content_type in company[harm].values():
            for moderation_action, action_data in content_type.items():
                if moderation_action not in moderation_action_totals:
                    moderation_action_totals[moderation_action] = 0
                for automation_status in action_data.values():
                    for automation_detection in automation_status.values():
                            moderation_action_totals[moderation_action] += automation_detection
                            total_sum_all_moderation_actions += automation_detection  # Accumulate the total sum

        # Add the total to the data after the loop
    moderation_action_totals['Total'] = total_sum_all_moderation_actions  # Append the total sum
    # Create a DataFrame for the moderation action totals
    moderation_action_data = {'Moderation Action': list(moderation_action_totals.keys()),
                              'N# Actions': list(moderation_action_totals.values())}
    df_moderation_action = pd.DataFrame(moderation_action_data).dropna()

    # Define visibility descriptions for mapping
    visibility_descriptions = {
    '["DECISION_VISIBILITY_CONTENT_REMOVED"]': 'REMOVED',
    '["DECISION_VISIBILITY_CONTENT_INTERACTION_RESTRICTED","DECISION_VISIBILITY_CONTENT_REMOVED"]': 'RESTRICTED/REMOVED',
    '["DECISION_VISIBILITY_CONTENT_REMOVED","DECISION_VISIBILITY_CONTENT_AGE_RESTRICTED"]': 'REMOVED/AGE RESTRICTED',
    '["DECISION_VISIBILITY_OTHER","DECISION_VISIBILITY_CONTENT_AGE_RESTRICTED"]': 'OTHER/AGE RESTRICTED',
    '["DECISION_VISIBILITY_CONTENT_LABELLED"]': 'LABELLED',
    '["DECISION_VISIBILITY_OTHER"]': 'OTHER',
    '["DECISION_VISIBILITY_CONTENT_AGE_RESTRICTED"]': 'AGE RESTRICTED',
    '["DECISION_VISIBILITY_CONTENT_DISABLED"]': 'DISABLED',
    '["DECISION_VISIBILITY_CONTENT_INTERACTION_RESTRICTED"]': 'INTERACTION RESTRICTED',
    '[]': 'NOT DEFINED',
    '["DECISION_VISIBILITY_CONTENT_DEMOTED"]': 'DEMOTED'
}

    # Rename the moderation actions in the DataFrame
    df_moderation_action['Moderation Action'] = df_moderation_action['Moderation Action'].map(visibility_descriptions)

    # Plotting the table
    fig2, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')

    # Create the table
    table = ax.table(cellText=df_moderation_action.values, colLabels=df_moderation_action.columns, cellLoc='center', loc='center',  bbox=[0, 0, 1, 1])

    formatter = mtick.FuncFormatter(lambda x, _: f'{x:,.0f}')
    for i in range(1, len(df_moderation_action.columns)):
        for key, cell in table.get_celld().items():
            if key[0] != 0 and key[1] == i:  # Exclude the header row
                cell.get_text().set_text(formatter(int(float(cell.get_text().get_text()))))  # Convert to float first, then to int

    # Set font size
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Display the table
    plt.show()

    #return fig2
    return df_moderation_action


######### Data General plot - Total number of Moderation Actions per Type of Moderation - ADD GRAPH
def plot_automation_status_totals2(data, harm):
    """ Sum all numbers for each automation status across all companies and plot the results as a table. """
    automation_status_totals = {}
    total_sum_all_automation_statuses = 0
    
    # Sum all numbers for each automation status across all companies
    for company in data.values():
        for content_type in company[harm].values():
            for moderation_action in content_type.values():
                    for automation_status, count in moderation_action.items():
                        if automation_status not in automation_status_totals:
                            automation_status_totals[automation_status] = 0
                        for automation_detection in count.values():
                            automation_status_totals[automation_status] += automation_detection
                            total_sum_all_automation_statuses += automation_detection  # Accumulate the total sum
    
    # Create a DataFrame for the automation status totals
    automation_status_data = {'Automation Status': list(automation_status_totals.keys()),
                              'N# Actions': list(automation_status_totals.values())}
    df_automation_status = pd.DataFrame(automation_status_data).dropna()

    # Define automation decision descriptions for mapping
    automated_decision_cleaned = {
        'AUTOMATED_DECISION_FULLY': 'Fully Automated',
        'AUTOMATED_DECISION_NOT_AUTOMATED': 'Not Automated',
        'AUTOMATED_DECISION_PARTIALLY': 'Partially Automated'
    }

    # Rename the automation statuses in the DataFrame
    df_automation_status['Automation Status'] = df_automation_status['Automation Status'].map(automated_decision_cleaned)

    # Plotting the table
    fig3, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')

    # Create the table
    table = ax.table(cellText=df_automation_status.values, colLabels=df_automation_status.columns, cellLoc='center', loc='center',  bbox=[0, 0, 1, 1])

    formatter = mtick.FuncFormatter(lambda x, _: f'{x:,.0f}')
    for i in range(1, len(df_automation_status.columns)):
        for key, cell in table.get_celld().items():
            if key[0] != 0 and key[1] == i:  # Exclude the header row
                cell.get_text().set_text(formatter(int(float(cell.get_text().get_text()))))  # Convert to float first, then to int

    # Set font size
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Display the table
    plt.show()

    # fig3
    return df_automation_status

# Example usage with the given data
# fig = plot_automation_status_totals(data_ACC)



######### Data General plot - Number of reported Harms per Company FIX DESIGN
def plot_harm_totals_per_company2(data, harm):
    """ Sum all numbers for each harm per company and plot the results as a table. """
    harm_totals_per_company = {company: {} for company in data.keys()}
    
    # Sum all numbers for each harm per company
    for company_name, company_data in data.items():
        if harm not in harm_totals_per_company:
            harm_totals_per_company[company_name] = {}
        if harm not in harm_totals_per_company[company_name]:
            harm_totals_per_company[company_name][harm] = 0
        for content_type, moderation_actions in company_data[harm].items():
            for moderation_action, counts in moderation_actions.items():
                for automation_status in counts.values():
                    for automation_detection in automation_status.values():
                        harm_totals_per_company[company_name][harm] += automation_detection

    # for company_name, company_data in data.items():
    #     if harm not in harm_totals_per_company:
    #         harm_totals_per_company[company_name] = {}
    #     if harm not in harm_totals_per_company[company_name]:
    #         harm_totals_per_company[company_name][harm] = 0
    #     for content_type, moderation_actions in company_data[harm].items():
    #         for moderation_action, counts in moderation_actions.items():
    #             for count in counts.values():
    #                 harm_totals_per_company[company_name][harm] += count
    
    # Prepare data for DataFrame
    data_for_df = {'Company': [], 'Harm': [], 'N# Actions': []}
    for company, harms in harm_totals_per_company.items():
        for harm, total_actions in harms.items():
            data_for_df['Company'].append(company)
            data_for_df['Harm'].append(harm)
            data_for_df['N# Actions'].append(total_actions)
    
    df_harm_per_company = pd.DataFrame(data_for_df).dropna()

    # Define harm category descriptions for mapping
    category_descriptions = {
        'STATEMENT_CATEGORY_SCOPE_OF_PLATFORM_SERVICE': 'PLATFORM SCOPE',
        'STATEMENT_CATEGORY_DATA_PROTECTION_AND_PRIVACY_VIOLATIONS': 'GDPR VIOLATION',
        'STATEMENT_CATEGORY_PORNOGRAPHY_OR_SEXUALIZED_CONTENT': 'PORN/SEX CONTENT',
        'STATEMENT_CATEGORY_ILLEGAL_OR_HARMFUL_SPEECH': 'ILLEGAL/HARMFUL SPEECH',
        'STATEMENT_CATEGORY_VIOLENCE': 'VIOLENCE',
        'STATEMENT_CATEGORY_SCAMS_AND_FRAUD': 'SCAMS/FRAUD',
        'STATEMENT_CATEGORY_UNSAFE_AND_ILLEGAL_PRODUCTS': 'ILLEGAL PRODUCTS',
        'STATEMENT_CATEGORY_NON_CONSENSUAL_BEHAVIOUR': 'NON CONSENSUAL BEHAVIOUR',
        'STATEMENT_CATEGORY_PROTECTION_OF_MINORS': 'PROTECT MINORS',
        'STATEMENT_CATEGORY_INTELLECTUAL_PROPERTY_INFRINGEMENTS': 'COPYRIGHT',
        'STATEMENT_CATEGORY_NEGATIVE_EFFECTS_ON_CIVIC_DISCOURSE_OR_ELECTIONS': 'NEGATIVE EFFECTS ELECTIONS',
        'STATEMENT_CATEGORY_RISK_FOR_PUBLIC_SECURITY': 'RISK PUBLIC SECURITY',
        'STATEMENT_CATEGORY_ANIMAL_WELFARE': 'ANIMAL WELFARE',
        'STATEMENT_CATEGORY_SELF_HARM': 'SELF HARM'
    }

    # Rename the harm categories in the DataFrame
    df_harm_per_company['Harm'] = df_harm_per_company['Harm'].map(category_descriptions)

    # Pivot the DataFrame
    pivot_df = df_harm_per_company.pivot(index='Company', columns='Harm', values='N# Actions').fillna(0).reset_index()

    # Plotting the table
    fig4, ax = plt.subplots(figsize=(8, 8))  # Adjust the figsize as needed
    ax.axis('tight')
    ax.axis('off')

    # Create the table
    table = ax.table(cellText=pivot_df.values, colLabels=pivot_df.columns, cellLoc='center', loc='center',  bbox=[0, 0, 1, 1])

    formatter = mtick.FuncFormatter(lambda x, _: f'{x:,.0f}')
    for i in range(1, len(pivot_df.columns)):
        for key, cell in table.get_celld().items():
            if key[0] != 0 and key[1] == i:  # Exclude the header row
                cell.get_text().set_text(formatter(int(float(cell.get_text().get_text()))))  # Convert to float first, then to int

    # Set cell height
    table.auto_set_column_width(col=list(range(len(pivot_df.columns))))
    table.scale(2, 2.5)  # Increase cell height (adjust as needed)

    # Set font size
    table.auto_set_font_size(False)
    table.set_fontsize(8)  # Adjust font size as needed

    # Display the table
    plt.show()

    #return fig4
    return pivot_df



######### Data General plot - Number of reported content type  per Company
def plot_content_type_totals_per_company2(data, harm):
    """ Sum all numbers for each content type per company and plot the results as a table. """
    content_type_totals_per_company = {company: {} for company in data.keys()}
    
    # Sum all numbers for each content type per company
    for company, harms in data.items():
        for content_type, content_data in harms[harm].items():
            if content_type not in content_type_totals_per_company[company]:
                content_type_totals_per_company[company][content_type] = 0
            for moderation_action in content_data.values():
                for count in moderation_action.values():
                    content_type_totals_per_company[company][content_type] += count
    
    # Prepare data for DataFrame
    data_for_df = {'Company': [], 'Content Type': [], 'N# Actions': []}
    for company, content_types in content_type_totals_per_company.items():
        for content_type, total_actions in content_types.items():
            data_for_df['Company'].append(company)
            data_for_df['Content Type'].append(content_type)
            data_for_df['N# Actions'].append(int(float(total_actions)))
    
    df_content_type_per_company = pd.DataFrame(data_for_df).dropna()

    # Define content type descriptions for mapping
    content_type_descriptions = {
        '["CONTENT_TYPE_OTHER"]': 'OTHER',
        '["CONTENT_TYPE_SYNTHETIC_MEDIA"]': 'SYNTHETIC MEDIA',
        '["CONTENT_TYPE_IMAGE"]': 'IMAGE',
        '["CONTENT_TYPE_TEXT"]': 'TEXT',
        '["CONTENT_TYPE_VIDEO"]': 'VIDEO',
        '["CONTENT_TYPE_PRODUCT"]': 'PRODUCT',
        '["CONTENT_TYPE_APP"]': 'APP',
        '["CONTENT_TYPE_AUDIO"]': 'AUDIO',
        '["CONTENT_TYPE_IMAGE","CONTENT_TYPE_TEXT"]': 'IMAGE/TEXT',
        '["CONTENT_TYPE_IMAGE","CONTENT_TYPE_TEXT","CONTENT_TYPE_VIDEO"]': 'IMAGE/TEXT/VIDEO',
        '["CONTENT_TYPE_AUDIO","CONTENT_TYPE_TEXT"],"CONTENT_TYPE_VIDEO"]': 'AUDIO/TEXT/VIDEO',
        '["CONTENT_TYPE_AUDIO","CONTENT_TYPE_IMAGE","CONTENT_TYPE_TEXT","CONTENT_TYPE_VIDEO"]': 'AUDIO/TEXT/VIDEO/IMAGE',
        '["CONTENT_TYPE_AUDIO","CONTENT_TYPE_IMAGE","CONTENT_TYPE_VIDEO"]': 'AUDIO/IMAGE/VIDEO'
    }

    # Rename the content type categories in the DataFrame
    df_content_type_per_company['Content Type'] = df_content_type_per_company['Content Type'].map(content_type_descriptions)

    # Pivot the DataFrame
    pivot_df = df_content_type_per_company.pivot(index='Company', columns='Content Type', values='N# Actions').fillna(0).reset_index()

    wrapped_columns = [textwrap.fill(col, width=20) for col in pivot_df.columns]

    # Plotting the table
    fig, ax = plt.subplots(figsize=(10, 7))  # Adjust the figsize as needed
    ax.axis('tight')
    ax.axis('off')
    ax.set_position([0, 0, 3, 3])

    # Create the table
    table = ax.table(cellText=pivot_df.values, colLabels=wrapped_columns, cellLoc='center', loc='center',  bbox=[0, 0, 1, 1])

    formatter = mtick.FuncFormatter(lambda x, _: f'{x:,.0f}')
    for i in range(1, len(pivot_df.columns)):
        for key, cell in table.get_celld().items():
            if key[0] != 0 and key[1] == i:  # Exclude the header row
                cell.get_text().set_text(formatter(int(float(cell.get_text().get_text()))))  # Convert to float first, then to int

    # Set cell height
    table.auto_set_column_width(col=list(range(len(pivot_df.columns))))
    table.scale(1, 1.5)  # Increase cell height (adjust as needed)

    # Set font size
    table.auto_set_font_size(False)
    table.set_fontsize(20)

    # Display the table
    plt.show()

    #return fig
    return pivot_df



######### Data General plot - Number of reported moderation action per Company

def sum_reports_per_moderation_action_per_company2(data, harm):
    """ 
    Sum all numbers for each moderation action per company
    """
    moderation_action_totals_per_company = {company: {} for company in data.keys()}
    for company, harms in data.items():
        for content_type in harms[harm].values():
            for moderation_action, action_data in content_type.items():
                if moderation_action not in moderation_action_totals_per_company[company]:
                    moderation_action_totals_per_company[company][moderation_action] = 0
                for count in action_data.values():
                    if pd.notna(count):  # Check if the count is not NaN
                        moderation_action_totals_per_company[company][moderation_action] += count
    
    data_for_df = {'Company': [], 'Moderation Action': [], 'N# Actions': []}
    for company, moderation_actions in moderation_action_totals_per_company.items():
        for moderation_action, total_actions in moderation_actions.items():
            data_for_df['Company'].append(company)
            data_for_df['Moderation Action'].append(moderation_action)
            data_for_df['N# Actions'].append(total_actions)
    
    df_moderation_action_per_company = pd.DataFrame(data_for_df).dropna()
    
    visibility_descriptions = {
        '["DECISION_VISIBILITY_CONTENT_REMOVED"]': 'REMOVED',
        '["DECISION_VISIBILITY_CONTENT_INTERACTION_RESTRICTED","DECISION_VISIBILITY_CONTENT_REMOVED"]': 'RESTRICTED/REMOVED',
        '["DECISION_VISIBILITY_CONTENT_REMOVED","DECISION_VISIBILITY_CONTENT_AGE_RESTRICTED"]': 'REMOVED/AGE RESTRICTED',
        '["DECISION_VISIBILITY_OTHER","DECISION_VISIBILITY_CONTENT_AGE_RESTRICTED"]': 'OTHER/AGE RESTRICTED',
        '["DECISION_VISIBILITY_CONTENT_LABELLED"]': 'LABELLED',
        '["DECISION_VISIBILITY_OTHER"]': 'OTHER',
        '["DECISION_VISIBILITY_CONTENT_AGE_RESTRICTED"]': 'AGE RESTRICTED',
        '["DECISION_VISIBILITY_CONTENT_DISABLED"]': 'DISABLED',
        '["DECISION_VISIBILITY_CONTENT_INTERACTION_RESTRICTED"]': 'INTERACTION RESTRICTED',
        '[]': 'NOT DEFINED',
        '["DECISION_VISIBILITY_CONTENT_DEMOTED"]': 'DEMOTED'
    }

    # Renaming the harm categories in your DataFrame
    df_moderation_action_per_company['Moderation Action'] = df_moderation_action_per_company['Moderation Action'].map(visibility_descriptions)

    # Pivot the DataFrame
    pivot_df = df_moderation_action_per_company.pivot(index='Company', columns='Moderation Action', values='N# Actions').fillna(0).reset_index()

    wrapped_columns = [textwrap.fill(col, width=15) for col in pivot_df.columns]


    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figsize as needed
    ax.axis('tight')
    ax.axis('off')
    ax.set_position([0, 0, 3, 3])
    # Create the table
    table = ax.table(cellText=pivot_df.values, colLabels=wrapped_columns, cellLoc='center', loc='center',  bbox=[0, 0, 1, 1])

    formatter = mtick.FuncFormatter(lambda x, _: f'{x:,.0f}')
    for i in range(1, len(pivot_df.columns)):
        for key, cell in table.get_celld().items():
            if key[0] != 0 and key[1] == i:  # Exclude the header row
                cell.get_text().set_text(formatter(int(float(cell.get_text().get_text()))))  # Convert to float first, then to int

    # Set cell height
    table.auto_set_column_width(col=list(range(len(pivot_df.columns))))
    table.scale(1, 1.5)  # Increase cell height (adjust as needed)

    # Set font size
    table.auto_set_font_size(False)
    table.set_fontsize(25)

    #return fig
    return pivot_df



######### Data General plot - Number of reported moderation type  per Company

def plot_automation_status_table_general2(data, harm):
    """ Sum all numbers for each automation status per company and plot the results as a table. """
    automation_status_totals_per_company = {company: {} for company in data.keys()}
    total_automation_status_totals = {}
    
    # Sum all numbers for each automation status per company
    for company, harms in data.items():
        for content_type in harms[harm].values():
            for moderation_action in content_type.values():
                for automation_status, count in moderation_action.items():
                        if automation_status not in automation_status_totals_per_company[company]:
                            automation_status_totals_per_company[company][automation_status] = 0
                        for automation_detection in count.values():
                            if pd.notna(automation_detection):  # Check if the count is not NaN
                                automation_status_totals_per_company[company][automation_status] += automation_detection
                                total_automation_status_totals[automation_status] = total_automation_status_totals.get(automation_status, 0) + automation_detection
    
    # Prepare data for DataFrame
    data_for_df = {'Company': [], 'Automation Status': [], 'N# Actions': []}
    for company, automation_statuses in automation_status_totals_per_company.items():
        for automation_status, total_actions in automation_statuses.items():
            data_for_df['Company'].append(company)
            data_for_df['Automation Status'].append(automation_status)
            data_for_df['N# Actions'].append(total_actions)
    
    df_automation_status_per_company = pd.DataFrame(data_for_df).dropna()

    # Define automated decision descriptions for mapping
    automated_decision_cleaned = {
        'AUTOMATED_DECISION_FULLY': 'Fully Automated',
        'AUTOMATED_DECISION_NOT_AUTOMATED': 'Not Automated',
        'AUTOMATED_DECISION_PARTIALLY': 'Partially Automated',
    }

    # Rename the automated decision categories in the DataFrame
    df_automation_status_per_company['Automation Status'] = df_automation_status_per_company['Automation Status'].map(automated_decision_cleaned)

    # Pivot the DataFrame
    pivot_df = df_automation_status_per_company.pivot(index='Company', columns='Automation Status', values='N# Actions').fillna(0).reset_index()

    

    # Plotting the table
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figsize as needed
    ax.axis('tight')
    ax.axis('off')

    # Create the table
    table = ax.table(cellText=pivot_df.values, colLabels=pivot_df.columns, cellLoc='center', loc='center',  bbox=[0, 0, 1, 1])


    formatter = mtick.FuncFormatter(lambda x, _: f'{x:,.0f}')
    for i in range(1, len(pivot_df.columns)):
        for key, cell in table.get_celld().items():
            if key[0] != 0 and key[1] == i:  # Exclude the header row
                cell.get_text().set_text(formatter(int(float(cell.get_text().get_text()))))  # Convert to float first, then to int

    # Set cell height
    table.auto_set_column_width(col=list(range(len(pivot_df.columns))))
    table.scale(1, 1.5)  # Increase cell height (adjust as needed)

    # Set font size
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Display the table
    plt.show()

    #return fig
    return pivot_df



def plot_normalized_automation_status2(data, harm):
    """ Plot the normalized counts of each automation status per company as a stacked bar chart. """
    automation_status_totals_per_company = {company: {} for company in data.keys()}
    
    # Sum all numbers for each automation status per company
    for company, harms in data.items():
        for content_type in harms[harm].values():
            for moderation_action in content_type.values():
                for automation_status, count in moderation_action.items():
                    if automation_status not in automation_status_totals_per_company[company]:
                        automation_status_totals_per_company[company][automation_status] = 0
                    for automation_detection in count.values():
                        if pd.notna(automation_detection):  # Check if the count is not NaN
                            automation_status_totals_per_company[company][automation_status] += automation_detection
    
    # Prepare data for DataFrame
    data_for_df = {'Company': [], 'Automation Status': [], 'N# Actions': []}
    for company, automation_statuses in automation_status_totals_per_company.items():
        for automation_status, total_actions in automation_statuses.items():
            data_for_df['Company'].append(company)
            data_for_df['Automation Status'].append(automation_status)
            data_for_df['N# Actions'].append(total_actions)
    
    df_automation_status_per_company = pd.DataFrame(data_for_df).dropna()

    # Define automated decision descriptions for mapping
    automated_decision_cleaned = {
        'AUTOMATED_DECISION_FULLY': 'Fully Automated',
        'AUTOMATED_DECISION_NOT_AUTOMATED': 'Not Automated',
        'AUTOMATED_DECISION_PARTIALLY': 'Partially Automated',
    }

    # Rename the automated decision categories in the DataFrame
    df_automation_status_per_company['Automation Status'] = df_automation_status_per_company['Automation Status'].map(automated_decision_cleaned)

    # Pivot the DataFrame
    pivot_df = df_automation_status_per_company.pivot(index='Company', columns='Automation Status', values='N# Actions').fillna(0).reset_index()

    # Normalize the values
    pivot_df.iloc[:, 1:] = pivot_df.iloc[:, 1:].div(pivot_df.iloc[:, 1:].sum(axis=1), axis=0)

    

    # Plotting the normalized data
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot_df.set_index('Company').plot(kind='bar', stacked=True, ax=ax)
    ax.set_xlabel('Company')
    ax.set_ylabel('Normalized Count')
    ax.set_title('Normalized Automation Status by Company')
    ax.legend(title='Automation Status', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.xticks(rotation=45)

    #return fig
    return pivot_df





######### Data General plot - Number of reported content type  per Harm

def plot_harm_content_type2(data, harm):
    """ Sum all numbers for each harm per content type and plot the results as both a table and a stacked bar chart. """
    harm_content_type_totals = {}
    
    # Sum all numbers for each harm per content type
    for company_data in data.values():
        for content_type, content_type_data in company_data[harm].items():
            if (harm, content_type) not in harm_content_type_totals:
                harm_content_type_totals[(harm, content_type)] = 0
            for moderation_action in content_type_data.values():
                for count in moderation_action.values():
                    for automation_detection in count.values():
                        if pd.notna(automation_detection):  # Check if the count is not NaN
                            harm_content_type_totals[(harm, content_type)] += count

    # Prepare data for DataFrame
    data_for_df = {'Harm': [], 'Content Type': [], 'N# Actions': []}
    for (harm, content_type), total_actions in harm_content_type_totals.items():
        data_for_df['Harm'].append(harm)
        data_for_df['Content Type'].append(content_type)
        data_for_df['N# Actions'].append(total_actions)

    df_harm_content_type = pd.DataFrame(data_for_df).dropna()

    # Define harm and content type descriptions for mapping
    category_descriptions = {
        'STATEMENT_CATEGORY_SCOPE_OF_PLATFORM_SERVICE': 'PLATFORM SCOPE',
        'STATEMENT_CATEGORY_DATA_PROTECTION_AND_PRIVACY_VIOLATIONS': 'GDPR VIOLATION',
        'STATEMENT_CATEGORY_PORNOGRAPHY_OR_SEXUALIZED_CONTENT': 'PORN/SEX CONTENT',
        'STATEMENT_CATEGORY_ILLEGAL_OR_HARMFUL_SPEECH': 'ILLEGAL/HARMFULL SPEECH',
        'STATEMENT_CATEGORY_VIOLENCE': 'VIOLENCE',
        'STATEMENT_CATEGORY_SCAMS_AND_FRAUD': 'SCAMS/FRAUD',
        'STATEMENT_CATEGORY_UNSAFE_AND_ILLEGAL_PRODUCTS': 'ILLEGAL PRODUCTS',
        'STATEMENT_CATEGORY_NON_CONSENSUAL_BEHAVIOUR': 'NON CONSENSUAL BEHAVIOUR',
        'STATEMENT_CATEGORY_PROTECTION_OF_MINORS': 'PROTECT MINORS',
        'STATEMENT_CATEGORY_INTELLECTUAL_PROPERTY_INFRINGEMENTS': 'COPYRIGHT',
        'STATEMENT_CATEGORY_NEGATIVE_EFFECTS_ON_CIVIC_DISCOURSE_OR_ELECTIONS': 'NEGATIVE EFFECTS ELECTIONS',
        'STATEMENT_CATEGORY_RISK_FOR_PUBLIC_SECURITY': 'RISK PUBLIC SECURITY',
        'STATEMENT_CATEGORY_ANIMAL_WELFARE': 'ANIMAL WELFARE',
        'STATEMENT_CATEGORY_SELF_HARM': 'SELF HARM'
    }

    content_type_descriptions = {
            '["CONTENT_TYPE_OTHER"]': 'OTHER',
            '["CONTENT_TYPE_SYNTHETIC_MEDIA"]': 'SYNTHETIC MEDIA',
            '["CONTENT_TYPE_IMAGE"]': 'IMAGE',
            '["CONTENT_TYPE_TEXT"]': 'TEXT',
            '["CONTENT_TYPE_VIDEO"]': 'VIDEO',
            '["CONTENT_TYPE_PRODUCT"]': 'PRODUCT',
            '["CONTENT_TYPE_APP"]': 'APP',
            '["CONTENT_TYPE_AUDIO"]': 'AUDIO',
            '["CONTENT_TYPE_IMAGE","CONTENT_TYPE_TEXT"]': 'IMAGE/TEXT',
            '["CONTENT_TYPE_IMAGE","CONTENT_TYPE_TEXT","CONTENT_TYPE_VIDEO"]': 'IMAGE/TEXT/VIDEO',
            '["CONTENT_TYPE_AUDIO","CONTENT_TYPE_TEXT"],"CONTENT_TYPE_VIDEO"]': 'AUDIO/TEXT/VIDEO', 
            '["CONTENT_TYPE_AUDIO","CONTENT_TYPE_IMAGE","CONTENT_TYPE_TEXT","CONTENT_TYPE_VIDEO"]': 'AUDIO/TEXT/VIDEO/IMAGE',
            '["CONTENT_TYPE_AUDIO","CONTENT_TYPE_IMAGE","CONTENT_TYPE_VIDEO"]': 'AUDIO/IMAGE/VIDEO'
        }

    # Rename the categories in the DataFrame
    df_harm_content_type['Harm'] = df_harm_content_type['Harm'].map(category_descriptions)
    df_harm_content_type['Content Type'] = df_harm_content_type['Content Type'].map(content_type_descriptions)

    # Pivot the DataFrame for the table
    pivot_df_table = df_harm_content_type.pivot(index='Harm', columns='Content Type', values='N# Actions').fillna(0).reset_index()

    wrapped_columns = [textwrap.fill(col, width=10) for col in pivot_df_table.columns]

    # Plotting the table
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.axis('tight')
    ax.axis('off')
    ax.set_position([0, 0, 3, 3])

    table = ax.table(cellText=pivot_df_table.values, colLabels=wrapped_columns, cellLoc='center', loc='center',  bbox=[0, 0, 1, 1])
    table.auto_set_column_width(col=list(range(len(pivot_df_table.columns))))

    formatter = mtick.FuncFormatter(lambda x, _: f'{x:,.0f}')
    for i in range(1, len(pivot_df_table.columns)):
        for key, cell in table.get_celld().items():
            if key[0] != 0 and key[1] == i:  # Exclude the header row
                cell.get_text().set_text(formatter(int(float(cell.get_text().get_text()))))  # Convert to float first, then to int


    table.scale(1, 1.5)
    table.auto_set_font_size(False)
    table.set_fontsize(25)

    plt.tight_layout()
    
    #return fig
    return pivot_df_table


def plot_harm_content_type_normalized2(data, harm):
    """ Sum all numbers for each harm per content type and plot the results as both a table and a stacked bar chart. """
    harm_content_type_totals = {}
    
    # Sum all numbers for each harm per content type
    for company_data in data.values():
        for content_type, content_type_data in company_data[harm].items():
            if (harm, content_type) not in harm_content_type_totals:
                harm_content_type_totals[(harm, content_type)] = 0
            for moderation_action in content_type_data.values():
                for count in moderation_action.values():
                    for automation_detection in count.values():
                          if pd.notna(automation_detection):  # Check if the count is not NaN
                               harm_content_type_totals[(harm, content_type)] += automation_detection

    # Prepare data for DataFrame
    data_for_df = {'Harm': [], 'Content Type': [], 'N# Actions': []}
    for (harm, content_type), total_actions in harm_content_type_totals.items():
        data_for_df['Harm'].append(harm)
        data_for_df['Content Type'].append(content_type)
        data_for_df['N# Actions'].append(total_actions)

    df_harm_content_type = pd.DataFrame(data_for_df).dropna()

    # Define harm and content type descriptions for mapping
    category_descriptions = {
        'STATEMENT_CATEGORY_SCOPE_OF_PLATFORM_SERVICE': 'PLATFORM SCOPE',
        'STATEMENT_CATEGORY_DATA_PROTECTION_AND_PRIVACY_VIOLATIONS': 'GDPR VIOLATION',
        'STATEMENT_CATEGORY_PORNOGRAPHY_OR_SEXUALIZED_CONTENT': 'PORN/SEX CONTENT',
        'STATEMENT_CATEGORY_ILLEGAL_OR_HARMFUL_SPEECH': 'ILLEGAL/HARMFULL SPEECH',
        'STATEMENT_CATEGORY_VIOLENCE': 'VIOLENCE',
        'STATEMENT_CATEGORY_SCAMS_AND_FRAUD': 'SCAMS/FRAUD',
        'STATEMENT_CATEGORY_UNSAFE_AND_ILLEGAL_PRODUCTS': 'ILLEGAL PRODUCTS',
        'STATEMENT_CATEGORY_NON_CONSENSUAL_BEHAVIOUR': 'NON CONSENSUAL BEHAVIOUR',
        'STATEMENT_CATEGORY_PROTECTION_OF_MINORS': 'PROTECT MINORS',
        'STATEMENT_CATEGORY_INTELLECTUAL_PROPERTY_INFRINGEMENTS': 'COPYRIGHT',
        'STATEMENT_CATEGORY_NEGATIVE_EFFECTS_ON_CIVIC_DISCOURSE_OR_ELECTIONS': 'NEGATIVE EFFECTS ELECTIONS',
        'STATEMENT_CATEGORY_RISK_FOR_PUBLIC_SECURITY': 'RISK PUBLIC SECURITY',
        'STATEMENT_CATEGORY_ANIMAL_WELFARE': 'ANIMAL WELFARE',
        'STATEMENT_CATEGORY_SELF_HARM': 'SELF HARM'
    }

    content_type_descriptions = {
            '["CONTENT_TYPE_OTHER"]': 'OTHER',
            '["CONTENT_TYPE_SYNTHETIC_MEDIA"]': 'SYNTHETIC MEDIA',
            '["CONTENT_TYPE_IMAGE"]': 'IMAGE',
            '["CONTENT_TYPE_TEXT"]': 'TEXT',
            '["CONTENT_TYPE_VIDEO"]': 'VIDEO',
            '["CONTENT_TYPE_PRODUCT"]': 'PRODUCT',
            '["CONTENT_TYPE_APP"]': 'APP',
            '["CONTENT_TYPE_AUDIO"]': 'AUDIO',
            '["CONTENT_TYPE_IMAGE","CONTENT_TYPE_TEXT"]': 'IMAGE/TEXT',
            '["CONTENT_TYPE_IMAGE","CONTENT_TYPE_TEXT","CONTENT_TYPE_VIDEO"]': 'IMAGE/TEXT/VIDEO',
            '["CONTENT_TYPE_AUDIO","CONTENT_TYPE_TEXT"],"CONTENT_TYPE_VIDEO"]': 'AUDIO/TEXT/VIDEO', 
            '["CONTENT_TYPE_AUDIO","CONTENT_TYPE_IMAGE","CONTENT_TYPE_TEXT","CONTENT_TYPE_VIDEO"]': 'AUDIO/TEXT/VIDEO/IMAGE',
            '["CONTENT_TYPE_AUDIO","CONTENT_TYPE_IMAGE","CONTENT_TYPE_VIDEO"]': 'AUDIO/IMAGE/VIDEO'
        }

    # Rename the categories in the DataFrame
    df_harm_content_type['Harm'] = df_harm_content_type['Harm'].map(category_descriptions).fillna('Harm')
    df_harm_content_type['Content Type'] = df_harm_content_type['Content Type'].map(content_type_descriptions).fillna('Content Type')
    
    df_harm_content_type = df_harm_content_type.groupby(['Harm', 'Content Type'], as_index=False).sum()


    # Pivot the DataFrame for the chart
    pivot_df_chart = df_harm_content_type.pivot(index='Harm', columns='Content Type', values='N# Actions').fillna(0).reset_index()
    
    # Normalize the values
    pivot_df_chart.iloc[:, 1:] = pivot_df_chart.iloc[:, 1:].div(pivot_df_chart.iloc[:, 1:].sum(axis=1), axis=0)

    # Plotting the normalized data
    fig, ax = plt.subplots(figsize=(10, 8))
    pivot_df_chart.set_index('Harm').plot(kind='bar', stacked=True, ax=ax)
    ax.set_xlabel('Harm')
    ax.set_ylabel('Normalized Count')
    ax.set_title('Normalized Content Type by Harm')
    ax.legend(title='Content Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.xticks(rotation=45)
    
    #return fig
    return pivot_df_chart


######## Data General plot - Number of reported Moderation Action  per Harm
def sum_reports_per_harm_per_moderation_action2(data, harm):
    """ Sum all numbers for each harm per moderation action and plot a table """
    # Sum all numbers for each harm per moderation action
    harm_moderation_action_totals = {}

    for company_data in data.values():
        for content_type, content_type_data in company_data[harm].items():
            for moderation_action, moderation_action_data in content_type_data.items():
                if (harm, moderation_action) not in harm_moderation_action_totals:
                    harm_moderation_action_totals[(harm, moderation_action)] = 0
                for count in moderation_action_data.values():
                    if pd.notna(count):  # Check if the count is not NaN
                        harm_moderation_action_totals[(harm, moderation_action)] += count

    # Prepare data for DataFrame
    data_for_df = {'Harm': [], 'Moderation Action': [], 'N# Actions': []}
    for (harm, moderation_action), total_actions in harm_moderation_action_totals.items():
        data_for_df['Harm'].append(harm)
        data_for_df['Moderation Action'].append(moderation_action)
        data_for_df['N# Actions'].append(total_actions)

    df_harm_moderation_action = pd.DataFrame(data_for_df).dropna()

    # Maps for renaming categories
    visibility_descriptions = {
        '["DECISION_VISIBILITY_CONTENT_REMOVED"]': 'REMOVED',
        '["DECISION_VISIBILITY_CONTENT_INTERACTION_RESTRICTED","DECISION_VISIBILITY_CONTENT_REMOVED"]': 'RESTRICTED/REMOVED',
        '["DECISION_VISIBILITY_CONTENT_REMOVED","DECISION_VISIBILITY_CONTENT_AGE_RESTRICTED"]': 'REMOVED/AGE RESTRICTED',
        '["DECISION_VISIBILITY_OTHER","DECISION_VISIBILITY_CONTENT_AGE_RESTRICTED"]': 'OTHER/AGE RESTRICTED',
        '["DECISION_VISIBILITY_CONTENT_LABELLED"]': 'LABELLED',
        '["DECISION_VISIBILITY_OTHER"]': 'OTHER',
        '["DECISION_VISIBILITY_CONTENT_AGE_RESTRICTED"]': 'AGE RESTRICTED',
        '["DECISION_VISIBILITY_CONTENT_DISABLED"]': 'DISABLED',
        '["DECISION_VISIBILITY_CONTENT_INTERACTION_RESTRICTED"]': 'INTERACTION RESTRICTED',
        '[]': 'NOT DEFINED',
        '["DECISION_VISIBILITY_CONTENT_DEMOTED"]': 'DEMOTED'
    }

    category_descriptions = {
        'STATEMENT_CATEGORY_SCOPE_OF_PLATFORM_SERVICE': 'PLATFORM SCOPE',
        'STATEMENT_CATEGORY_DATA_PROTECTION_AND_PRIVACY_VIOLATIONS': 'GDPR VIOLATION',
        'STATEMENT_CATEGORY_PORNOGRAPHY_OR_SEXUALIZED_CONTENT': 'PORN/SEX CONTENT',
        'STATEMENT_CATEGORY_ILLEGAL_OR_HARMFUL_SPEECH': 'ILLEGAL/HARMFULL SPEECH',
        'STATEMENT_CATEGORY_VIOLENCE': 'VIOLENCE',
        'STATEMENT_CATEGORY_SCAMS_AND_FRAUD': 'SCAMS/FRAUD',
        'STATEMENT_CATEGORY_UNSAFE_AND_ILLEGAL_PRODUCTS': 'ILLEGAL PRODUCTS',
        'STATEMENT_CATEGORY_NON_CONSENSUAL_BEHAVIOUR': 'NON CONSENSUAL BEHAVIOUR',
        'STATEMENT_CATEGORY_PROTECTION_OF_MINORS': 'PROTECT MINORS',
        'STATEMENT_CATEGORY_INTELLECTUAL_PROPERTY_INFRINGEMENTS': 'COPYRIGHT',
        'STATEMENT_CATEGORY_NEGATIVE_EFFECTS_ON_CIVIC_DISCOURSE_OR_ELECTIONS': 'NEGATIVE EFFECTS ELECTIONS',
        'STATEMENT_CATEGORY_RISK_FOR_PUBLIC_SECURITY': 'RISK PUBLIC SECURITY',
        'STATEMENT_CATEGORY_ANIMAL_WELFARE': 'ANIMAL WELFARE',
        'STATEMENT_CATEGORY_SELF_HARM': 'SELF HARM'
    }

    # Renaming the harm categories in the DataFrame
    df_harm_moderation_action['Harm'] = df_harm_moderation_action['Harm'].map(category_descriptions)
    df_harm_moderation_action['Moderation Action'] = df_harm_moderation_action['Moderation Action'].map(visibility_descriptions)

    # Pivot the DataFrame
    pivot_df = df_harm_moderation_action.pivot(index='Harm', columns='Moderation Action', values='N# Actions').fillna(0).reset_index()

    wrapped_columns = [textwrap.fill(col, width=10) for col in pivot_df.columns]
    

    # Plotting the table graph
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figsize as needed
    ax.axis('tight')
    ax.axis('off')
    ax.set_position([0, 0, 3, 3])
    
    # Create the table
    table = ax.table(cellText=pivot_df.values, colLabels=wrapped_columns, cellLoc='center', loc='center',  bbox=[0, 0, 1, 1])

    formatter = mtick.FuncFormatter(lambda x, _: f'{x:,.0f}')
    for i in range(1, len(pivot_df.columns)):
        for key, cell in table.get_celld().items():
            if key[0] != 0 and key[1] == i:  # Exclude the header row
                cell.get_text().set_text(formatter(int(float(cell.get_text().get_text()))))  # Convert to float first, then to int

    # Set cell height
    table.auto_set_column_width(col=list(range(len(pivot_df.columns))))
    table.scale(1, 1.5)  # Increase cell height (adjust as needed)

    # Set font size
    table.auto_set_font_size(False)
    table.set_fontsize(25)

    # Display the figure
    plt.show()

    #return fig
    return pivot_df


######### Data General plot - Number of reported Moderation type  per Harm

def plot_harm_automation_status2(data, harm):
    """ Sum all numbers for each harm per automation status and plot the results as both a table and a stacked bar chart. """
    harm_automation_status_totals = {}
    
    # Sum all numbers for each harm per automation status
    for company_data in data.values():
        for content_type, content_type_data in company_data[harm].items():
            for moderation_action, moderation_action_data in content_type_data.items():
                for automation_status, count in moderation_action_data.items():
                    if (harm, automation_status) not in harm_automation_status_totals:
                        harm_automation_status_totals[(harm, automation_status)] = 0
                    for automation_detection in count.values():
                        if pd.notna(automation_detection):  # Check if the count is not NaN                    
                            harm_automation_status_totals[(harm, automation_status)] += automation_detection  # Check if the count is not NaN
                            harm_automation_status_totals[(harm, automation_status)] += automation_detection

    # Prepare data for DataFrame
    data_for_df = {'Harm': [], 'Automation Status': [], 'N# Actions': []}
    for (harm, automation_status), total_actions in harm_automation_status_totals.items():
        data_for_df['Harm'].append(harm)
        data_for_df['Automation Status'].append(automation_status)
        data_for_df['N# Actions'].append(total_actions)

    df_harm_automation_status = pd.DataFrame(data_for_df).dropna()

    # Define automated decision and category descriptions for mapping
    automated_decision_cleaned = {
            'AUTOMATED_DECISION_FULLY': 'Fully Automated',
            'AUTOMATED_DECISION_NOT_AUTOMATED': 'Not Automated',
            'AUTOMATED_DECISION_PARTIALLY': 'Partially automated',
    }

    category_descriptions = {
        'STATEMENT_CATEGORY_SCOPE_OF_PLATFORM_SERVICE': 'PLATFORM SCOPE',
        'STATEMENT_CATEGORY_DATA_PROTECTION_AND_PRIVACY_VIOLATIONS': 'GDPR VIOLATION',
        'STATEMENT_CATEGORY_PORNOGRAPHY_OR_SEXUALIZED_CONTENT': 'PORN/SEX CONTENT',
        'STATEMENT_CATEGORY_ILLEGAL_OR_HARMFUL_SPEECH': 'ILLEGAL/HARMFULL SPEECH',
        'STATEMENT_CATEGORY_VIOLENCE': 'VIOLENCE',
        'STATEMENT_CATEGORY_SCAMS_AND_FRAUD': 'SCAMS/FRAUD',
        'STATEMENT_CATEGORY_UNSAFE_AND_ILLEGAL_PRODUCTS': 'ILLEGAL PRODUCTS',
        'STATEMENT_CATEGORY_NON_CONSENSUAL_BEHAVIOUR': 'NON CONSENSUAL BEHAVIOUR',
        'STATEMENT_CATEGORY_PROTECTION_OF_MINORS': 'PROTECT MINORS',
        'STATEMENT_CATEGORY_INTELLECTUAL_PROPERTY_INFRINGEMENTS': 'COPYRIGHT',
        'STATEMENT_CATEGORY_NEGATIVE_EFFECTS_ON_CIVIC_DISCOURSE_OR_ELECTIONS': 'NEGATIVE EFFECTS ELECTIONS',
        'STATEMENT_CATEGORY_RISK_FOR_PUBLIC_SECURITY': 'RISK PUBLIC SECURITY',
        'STATEMENT_CATEGORY_ANIMAL_WELFARE': 'ANIMAL WELFARE',
        'STATEMENT_CATEGORY_SELF_HARM': 'SELF HARM'
    }

    # Rename the categories in the DataFrame
    df_harm_automation_status['Harm'] = df_harm_automation_status['Harm'].map(category_descriptions)
    df_harm_automation_status['Automation Status'] = df_harm_automation_status['Automation Status'].map(automated_decision_cleaned)

    # Pivot the DataFrame for the table
    pivot_df_table = df_harm_automation_status.pivot(index='Harm', columns='Automation Status', values='N# Actions').fillna(0).reset_index()

    # Plotting the table
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=pivot_df_table.values, colLabels=pivot_df_table.columns, cellLoc='center', loc='center',  bbox=[0, 0, 1, 1])
    table.auto_set_column_width(col=list(range(len(pivot_df_table.columns))))

    formatter = mtick.FuncFormatter(lambda x, _: f'{x:,.0f}')
    for i in range(1, len(pivot_df_table.columns)):
        for key, cell in table.get_celld().items():
            if key[0] != 0 and key[1] == i:  # Exclude the header row
                cell.get_text().set_text(formatter(int(float(cell.get_text().get_text()))))  # Convert to float first, then to int

    table.scale(1, 1.5)
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    #return fig
    return pivot_df_table


def plot_harm_automation_status2_normalized(data, harm):
    """ Sum all numbers for each harm per automation status and plot the normalized results as a stacked bar chart. """
    harm_automation_status_totals = {}
    
    # Sum all numbers for each harm per automation status
    for company_data in data.values():
        for content_type, content_type_data in company_data[harm].items():
            for moderation_action, moderation_action_data in content_type_data.items():
                for automation_status, count in moderation_action_data.items():
                    if (harm, automation_status) not in harm_automation_status_totals:
                        harm_automation_status_totals[(harm, automation_status)] = 0
                    for automation_detection in count.values():                      
                        if pd.notna(automation_detection):  # Check if the count is not NaN
                            harm_automation_status_totals[(harm, automation_status)] += automation_detection

    # Prepare data for DataFrame
    data_for_df = {'Harm': [], 'Automation Status': [], 'N# Actions': []}
    for (harm, automation_status), total_actions in harm_automation_status_totals.items():
        data_for_df['Harm'].append(harm)
        data_for_df['Automation Status'].append(automation_status)
        data_for_df['N# Actions'].append(total_actions)

    df_harm_automation_status = pd.DataFrame(data_for_df).dropna()

    # Define automated decision and category descriptions for mapping
    automated_decision_cleaned = {
            'AUTOMATED_DECISION_FULLY': 'Fully Automated',
            'AUTOMATED_DECISION_NOT_AUTOMATED': 'Not Automated',
            'AUTOMATED_DECISION_PARTIALLY': 'Partially automated',
    }

    category_descriptions = {
        'STATEMENT_CATEGORY_SCOPE_OF_PLATFORM_SERVICE': 'PLATFORM SCOPE',
        'STATEMENT_CATEGORY_DATA_PROTECTION_AND_PRIVACY_VIOLATIONS': 'GDPR VIOLATION',
        'STATEMENT_CATEGORY_PORNOGRAPHY_OR_SEXUALIZED_CONTENT': 'PORN/SEX CONTENT',
        'STATEMENT_CATEGORY_ILLEGAL_OR_HARMFUL_SPEECH': 'ILLEGAL/HARMFULL SPEECH',
        'STATEMENT_CATEGORY_VIOLENCE': 'VIOLENCE',
        'STATEMENT_CATEGORY_SCAMS_AND_FRAUD': 'SCAMS/FRAUD',
        'STATEMENT_CATEGORY_UNSAFE_AND_ILLEGAL_PRODUCTS': 'ILLEGAL PRODUCTS',
        'STATEMENT_CATEGORY_NON_CONSENSUAL_BEHAVIOUR': 'NON CONSENSUAL BEHAVIOUR',
        'STATEMENT_CATEGORY_PROTECTION_OF_MINORS': 'PROTECT MINORS',
        'STATEMENT_CATEGORY_INTELLECTUAL_PROPERTY_INFRINGEMENTS': 'COPYRIGHT',
        'STATEMENT_CATEGORY_NEGATIVE_EFFECTS_ON_CIVIC_DISCOURSE_OR_ELECTIONS': 'NEGATIVE EFFECTS ELECTIONS',
        'STATEMENT_CATEGORY_RISK_FOR_PUBLIC_SECURITY': 'RISK PUBLIC SECURITY',
        'STATEMENT_CATEGORY_ANIMAL_WELFARE': 'ANIMAL WELFARE',
        'STATEMENT_CATEGORY_SELF_HARM': 'SELF HARM'
    }

    # Rename the categories in the DataFrame
    df_harm_automation_status['Harm'] = df_harm_automation_status['Harm'].map(category_descriptions)
    df_harm_automation_status['Automation Status'] = df_harm_automation_status['Automation Status'].map(automated_decision_cleaned)

    # Pivot the DataFrame for the chart
    pivot_df_chart = df_harm_automation_status.pivot(index='Harm', columns='Automation Status', values='N# Actions').fillna(0).reset_index()

    # Normalize the values
    pivot_df_chart.iloc[:, 1:] = pivot_df_chart.iloc[:, 1:].div(pivot_df_chart.iloc[:, 1:].sum(axis=1), axis=0)

    # Plotting the normalized data
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pivot_df_chart.set_index('Harm').plot(kind='bar', stacked=True, ax=ax)
    ax.set_xlabel('Harm')
    ax.set_ylabel('Normalized Count')
    ax.set_title('Normalized Automation Status by Harm')
    ax.legend(title='Automation Status', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.xticks(rotation=45)
    
    #return fig
    return pivot_df_chart



######### Data General plot - Number of reported Moderation type  per Content Type

def plot_content_type_automation_status2(data, harm):
    """ Sum all numbers for each content type per automation status and plot the results as both a table and a stacked bar chart. """
    content_type_automation_status_totals = {}
    
    # Sum all numbers for each content type per automation status
    for company_data in data.values():
        for content_type, content_type_data in company_data[harm].items():
            for moderation_action_data in content_type_data.values():
                for automation_status, count in moderation_action_data.items():
                    if (content_type, automation_status) not in content_type_automation_status_totals:
                        content_type_automation_status_totals[(content_type, automation_status)] = 0
                    for automation_detection in count.values():        
                        if pd.notna(automation_detection):  # Check if the count is not NaN
                            content_type_automation_status_totals[(content_type, automation_status)] += automation_detection

    # Prepare data for DataFrame
    data_for_df = {'Content Type': [], 'Automation Status': [], 'N# Actions': []}
    for (content_type, automation_status), total_actions in content_type_automation_status_totals.items():
        data_for_df['Content Type'].append(content_type)
        data_for_df['Automation Status'].append(automation_status)
        data_for_df['N# Actions'].append(total_actions)

    df_content_type_automation_status = pd.DataFrame(data_for_df).dropna()

    # Define automated decision and content type descriptions for mapping
    automated_decision_cleaned = {
        'AUTOMATED_DECISION_FULLY': 'Fully Automated',
        'AUTOMATED_DECISION_NOT_AUTOMATED': 'Not Automated',
        'AUTOMATED_DECISION_PARTIALLY': 'Partially automated',
    }

    content_type_descriptions = {
        '["CONTENT_TYPE_OTHER"]': 'OTHER',
        '["CONTENT_TYPE_SYNTHETIC_MEDIA"]': 'SYNTHETIC MEDIA',
        '["CONTENT_TYPE_IMAGE"]': 'IMAGE',
        '["CONTENT_TYPE_TEXT"]': 'TEXT',
        '["CONTENT_TYPE_VIDEO"]': 'VIDEO',
        '["CONTENT_TYPE_PRODUCT"]': 'PRODUCT',
        '["CONTENT_TYPE_APP"]': 'APP',
        '["CONTENT_TYPE_AUDIO"]': 'AUDIO',
        '["CONTENT_TYPE_IMAGE","CONTENT_TYPE_TEXT"]': 'IMAGE/TEXT',
        '["CONTENT_TYPE_IMAGE","CONTENT_TYPE_TEXT","CONTENT_TYPE_VIDEO"]': 'IMAGE/TEXT/VIDEO',
        '["CONTENT_TYPE_AUDIO","CONTENT_TYPE_TEXT"],"CONTENT_TYPE_VIDEO"]': 'AUDIO/TEXT/VIDEO',
        '["CONTENT_TYPE_AUDIO","CONTENT_TYPE_IMAGE","CONTENT_TYPE_TEXT","CONTENT_TYPE_VIDEO"]': 'AUDIO/TEXT/VIDEO/IMAGE',
        '["CONTENT_TYPE_AUDIO","CONTENT_TYPE_IMAGE","CONTENT_TYPE_VIDEO"]': 'AUDIO/IMAGE/VIDEO'
    }

    # Rename the content types and automation statuses in the DataFrame
    df_content_type_automation_status['Content Type'] = df_content_type_automation_status['Content Type'].map(content_type_descriptions).fillna('Content Type')
    df_content_type_automation_status['Automation Status'] = df_content_type_automation_status['Automation Status'].map(automated_decision_cleaned).fillna('Automation Status')
    
    df_content_type_automation_status = df_content_type_automation_status.groupby(['Content Type', 'Automation Status'], as_index=False).sum()
    # Pivot the DataFrame for the table
    pivot_df_table = df_content_type_automation_status.pivot(index='Content Type', columns='Automation Status', values='N# Actions').fillna(0).reset_index()

    # Plotting the table
    fig, ax = plt.subplots(1, figsize=(10, 8))
    
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=pivot_df_table.values, colLabels=pivot_df_table.columns, cellLoc='center', loc='center',  bbox=[0, 0, 1, 1])
    table.auto_set_column_width(col=list(range(len(pivot_df_table.columns))))
    table.scale(1, 1.5)
    table.auto_set_font_size(False)
    table.set_fontsize(10)

   
    #return fig
    return pivot_df_table


def plot_content_type_automation_status2_normalized(data, harm):
    """ Sum all numbers for each content type per automation status and plot the results as a normalized stacked bar chart. """
    content_type_automation_status_totals = {}
    
    # Sum all numbers for each content type per automation status
    for company_data in data.values():
        for content_type, content_type_data in company_data[harm].items():
            for moderation_action_data in content_type_data.values():
                for automation_status, count in moderation_action_data.items():
                    if (content_type, automation_status) not in content_type_automation_status_totals:
                        content_type_automation_status_totals[(content_type, automation_status)] = 0
                    for automation_detection in count.values():
                        if pd.notna(automation_detection):  # Check if the count is not NaN       
                            content_type_automation_status_totals[(content_type, automation_status)] += automation_detection

    # Prepare data for DataFrame
    data_for_df = {'Content Type': [], 'Automation Status': [], 'N# Actions': []}
    for (content_type, automation_status), total_actions in content_type_automation_status_totals.items():
        data_for_df['Content Type'].append(content_type)
        data_for_df['Automation Status'].append(automation_status)
        data_for_df['N# Actions'].append(total_actions)

    df_content_type_automation_status = pd.DataFrame(data_for_df).dropna()

    # Define automated decision and content type descriptions for mapping
    automated_decision_cleaned = {
        'AUTOMATED_DECISION_FULLY': 'Fully Automated',
        'AUTOMATED_DECISION_NOT_AUTOMATED': 'Not Automated',
        'AUTOMATED_DECISION_PARTIALLY': 'Partially automated',
    }

    content_type_descriptions = {
        '["CONTENT_TYPE_OTHER"]': 'OTHER',
        '["CONTENT_TYPE_SYNTHETIC_MEDIA"]': 'SYNTHETIC MEDIA',
        '["CONTENT_TYPE_IMAGE"]': 'IMAGE',
        '["CONTENT_TYPE_TEXT"]': 'TEXT',
        '["CONTENT_TYPE_VIDEO"]': 'VIDEO',
        '["CONTENT_TYPE_PRODUCT"]': 'PRODUCT',
        '["CONTENT_TYPE_APP"]': 'APP',
        '["CONTENT_TYPE_AUDIO"]': 'AUDIO',
        '["CONTENT_TYPE_IMAGE","CONTENT_TYPE_TEXT"]': 'IMAGE/TEXT',
        '["CONTENT_TYPE_IMAGE","CONTENT_TYPE_TEXT","CONTENT_TYPE_VIDEO"]': 'IMAGE/TEXT/VIDEO',
        '["CONTENT_TYPE_AUDIO","CONTENT_TYPE_TEXT"],"CONTENT_TYPE_VIDEO"]': 'AUDIO/TEXT/VIDEO',
        '["CONTENT_TYPE_AUDIO","CONTENT_TYPE_IMAGE","CONTENT_TYPE_TEXT","CONTENT_TYPE_VIDEO"]': 'AUDIO/TEXT/VIDEO/IMAGE',
        '["CONTENT_TYPE_AUDIO","CONTENT_TYPE_IMAGE","CONTENT_TYPE_VIDEO"]': 'AUDIO/IMAGE/VIDEO'
    }

    # Rename the content types and automation statuses in the DataFrame

    df_content_type_automation_status['Content Type'] = df_content_type_automation_status['Content Type'].map(content_type_descriptions).fillna('Content Type')
    df_content_type_automation_status['Automation Status'] = df_content_type_automation_status['Automation Status'].map(automated_decision_cleaned).fillna('Automation Status')


    df_content_type_automation_status = df_content_type_automation_status.groupby(['Content Type', 'Automation Status'], as_index=False).sum()

    # Pivot the DataFrame for the chart
    pivot_df_chart = df_content_type_automation_status.pivot(index='Content Type', columns='Automation Status', values='N# Actions').fillna(0).reset_index()

    # Normalize the values
    pivot_df_chart.iloc[:, 1:] = pivot_df_chart.iloc[:, 1:].div(pivot_df_chart.iloc[:, 1:].sum(axis=1), axis=0)

    # Plotting the normalized stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 8))
    pivot_df_chart.set_index('Content Type').plot(kind='bar', stacked=True, ax=ax)
    ax.set_xlabel('Content Type')
    ax.set_ylabel('Normalized Count')
    ax.set_title('Normalized Automation Status by Content Type')
    ax.legend(title='Automation Status', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.xticks(rotation=45)
    #return fig
    return pivot_df_chart

######### Data General plot - Number of reported Moderation action  per Content Type


def generate_content_type_moderation_action_figure2(data, harm):
    """ Sum all numbers for each content type per moderation action and plot a table """
    # Sum all numbers for each content type per moderation action
    content_type_moderation_action_totals = {}

    for company_data in data.values():
        for content_type, content_type_data in company_data[harm].items():
            for moderation_action, moderation_action_data in content_type_data.items():
                if (content_type, moderation_action) not in content_type_moderation_action_totals:
                    content_type_moderation_action_totals[(content_type, moderation_action)] = 0
                for automation_status in moderation_action_data.values():
                        for count in automation_status.values():
                            if pd.notna(count):  # Check if the count is not NaN
                                content_type_moderation_action_totals[(content_type, moderation_action)] += count

    # Prepare data for DataFrame
    data_for_df = {'Content Type': [], 'Moderation Action': [], 'N# Actions': []}
    for (content_type, moderation_action), total_actions in content_type_moderation_action_totals.items():
        data_for_df['Content Type'].append(content_type)
        data_for_df['Moderation Action'].append(moderation_action)
        data_for_df['N# Actions'].append(int(float(total_actions)))

    df_content_type_moderation_action = pd.DataFrame(data_for_df).dropna()

    # Maps for renaming categories
    content_type_descriptions = {
        '["CONTENT_TYPE_OTHER"]': 'OTHER',
        '["CONTENT_TYPE_SYNTHETIC_MEDIA"]': 'SYNTHETIC MEDIA',
        '["CONTENT_TYPE_IMAGE"]': 'IMAGE',
        '["CONTENT_TYPE_TEXT"]': 'TEXT',
        '["CONTENT_TYPE_VIDEO"]': 'VIDEO',
        '["CONTENT_TYPE_PRODUCT"]': 'PRODUCT',
        '["CONTENT_TYPE_APP"]': 'APP',
        '["CONTENT_TYPE_AUDIO"]': 'AUDIO',
        '["CONTENT_TYPE_IMAGE","CONTENT_TYPE_TEXT"]': 'IMAGE/TEXT',
        '["CONTENT_TYPE_IMAGE","CONTENT_TYPE_TEXT","CONTENT_TYPE_VIDEO"]': 'IMAGE/TEXT/VIDEO',
        '["CONTENT_TYPE_AUDIO","CONTENT_TYPE_TEXT"],"CONTENT_TYPE_VIDEO"]': 'AUDIO/TEXT/VIDEO',
        '["CONTENT_TYPE_AUDIO","CONTENT_TYPE_IMAGE","CONTENT_TYPE_TEXT","CONTENT_TYPE_VIDEO"]': 'AUDIO/TEXT/VIDEO/IMAGE',
        '["CONTENT_TYPE_AUDIO","CONTENT_TYPE_IMAGE","CONTENT_TYPE_VIDEO"]': 'AUDIO/IMAGE/VIDEO'
    }


    # Define visibility descriptions for mapping
    visibility_descriptions = {
    '["DECISION_VISIBILITY_CONTENT_REMOVED"]': 'REMOVED',
    '["DECISION_VISIBILITY_CONTENT_INTERACTION_RESTRICTED","DECISION_VISIBILITY_CONTENT_REMOVED"]': 'RESTRICTED/REMOVED',
    '["DECISION_VISIBILITY_CONTENT_REMOVED","DECISION_VISIBILITY_CONTENT_AGE_RESTRICTED"]': 'REMOVED/AGE RESTRICTED',
    '["DECISION_VISIBILITY_OTHER","DECISION_VISIBILITY_CONTENT_AGE_RESTRICTED"]': 'OTHER/AGE RESTRICTED',
    '["DECISION_VISIBILITY_CONTENT_LABELLED"]': 'LABELLED',
    '["DECISION_VISIBILITY_OTHER"]': 'OTHER',
    '["DECISION_VISIBILITY_CONTENT_AGE_RESTRICTED"]': 'AGE RESTRICTED',
    '["DECISION_VISIBILITY_CONTENT_DISABLED"]': 'DISABLED',
    '["DECISION_VISIBILITY_CONTENT_INTERACTION_RESTRICTED"]': 'INTERACTION RESTRICTED',
    '[]': 'NOT DEFINED',
    '["DECISION_VISIBILITY_CONTENT_DEMOTED"]': 'DEMOTED'
}
    
    # Renaming the content type categories in the DataFrame
    df_content_type_moderation_action['Content Type'] = df_content_type_moderation_action['Content Type'].map(content_type_descriptions)

    # Renaming the moderation action categories in the DataFrame
    df_content_type_moderation_action['Moderation Action'] = df_content_type_moderation_action['Moderation Action'].map(visibility_descriptions)

    # Pivot the DataFrame
    pivot_df = df_content_type_moderation_action.pivot(index='Content Type', columns='Moderation Action', values='N# Actions').fillna(0).reset_index()

    wrapped_columns = [textwrap.fill(col, width=20) for col in pivot_df.columns]

    # Plotting the table graph
    fig, ax = plt.subplots(figsize=(10, 4))  # Adjust the figsize as needed
    ax.axis('tight')
    ax.axis('off')
    ax.set_position([0, 0, 3, 3])

    # Create the table
    table = ax.table(cellText=pivot_df.values, colLabels=wrapped_columns, cellLoc='center', loc='center',  bbox=[0, 0, 1, 1])

    # Set cell height
    table.auto_set_column_width(col=list(range(len(pivot_df.columns))))
    table.scale(1, 1)  # Increase cell height (adjust as needed)

    formatter = mtick.FuncFormatter(lambda x, _: f'{x:,.0f}')
    for i in range(1, len(pivot_df.columns)):
        for key, cell in table.get_celld().items():
            if key[0] != 0 and key[1] == i:  # Exclude the header row
                cell.get_text().set_text(formatter(int(float(cell.get_text().get_text()))))  # Convert to float first, then to int

    # Set font size
    table.auto_set_font_size(False)
    table.set_fontsize(20)

    # Display the figure
    plt.show()

    #return fig
    return pivot_df

# Example usage:
# fig = generate_content_type_moderation_action_figure(data_ACC)







######### Data General plot - Number of reported Moderation type  per Moderation Action

def generate_moderation_action_automation_status_figure2(data, harm):
    """ Sum all numbers for each moderation action per automation status and plot a table """
    # Sum all numbers for each moderation action per automation status
    moderation_action_automation_status_totals = {}

    for company_data in data.values():
        for content_type_data in company_data[harm].values():
            for moderation_action, moderation_action_data in content_type_data.items():
                for automation_status, count in moderation_action_data.items():
                    if (moderation_action, automation_status) not in moderation_action_automation_status_totals:
                        moderation_action_automation_status_totals[(moderation_action, automation_status)] = 0
                    for automation_detection in count.values():         
                        if pd.notna(automation_detection):  # Check if the count is not NaN
                            moderation_action_automation_status_totals[(moderation_action, automation_status)] += automation_detection

    # Prepare data for DataFrame
    data_for_df = {'Moderation Action': [], 'Automation Status': [], 'N# Actions': []}
    for (moderation_action, automation_status), total_actions in moderation_action_automation_status_totals.items():
        data_for_df['Moderation Action'].append(moderation_action)
        data_for_df['Automation Status'].append(automation_status)
        data_for_df['N# Actions'].append(total_actions)

    df_moderation_action_automation_status = pd.DataFrame(data_for_df).dropna()

    # Maps for renaming categories
    visibility_descriptions = {
    '["DECISION_VISIBILITY_CONTENT_REMOVED"]': 'REMOVED',
    '["DECISION_VISIBILITY_CONTENT_INTERACTION_RESTRICTED","DECISION_VISIBILITY_CONTENT_REMOVED"]': 'RESTRICTED/REMOVED',
    '["DECISION_VISIBILITY_CONTENT_REMOVED","DECISION_VISIBILITY_CONTENT_AGE_RESTRICTED"]': 'REMOVED/AGE RESTRICTED',
    '["DECISION_VISIBILITY_OTHER","DECISION_VISIBILITY_CONTENT_AGE_RESTRICTED"]': 'OTHER/AGE RESTRICTED',
    '["DECISION_VISIBILITY_CONTENT_LABELLED"]': 'LABELLED',
    '["DECISION_VISIBILITY_OTHER"]': 'OTHER',
    '["DECISION_VISIBILITY_CONTENT_AGE_RESTRICTED"]': 'AGE RESTRICTED',
    '["DECISION_VISIBILITY_CONTENT_DISABLED"]': 'DISABLED',
    '["DECISION_VISIBILITY_CONTENT_INTERACTION_RESTRICTED"]': 'INTERACTION RESTRICTED',
    '[]': 'NOT DEFINED',
    '["DECISION_VISIBILITY_CONTENT_DEMOTED"]': 'DEMOTED'

    }
   

    automated_decision_cleaned = {
        'AUTOMATED_DECISION_FULLY': 'Fully Automated',
        'AUTOMATED_DECISION_NOT_AUTOMATED': 'Not Automated',
        'AUTOMATED_DECISION_PARTIALLY': 'Partially automated',
    }

    # Renaming the categories in the DataFrame
    df_moderation_action_automation_status['Moderation Action'] = df_moderation_action_automation_status['Moderation Action'].map(visibility_descriptions)
    df_moderation_action_automation_status['Automation Status'] = df_moderation_action_automation_status['Automation Status'].map(automated_decision_cleaned)

    df_moderation_action_automation_status = df_moderation_action_automation_status.groupby(
    ['Moderation Action', 'Automation Status'], as_index=False).sum()


    # Pivot the DataFrame
    pivot_df = df_moderation_action_automation_status.pivot(index='Moderation Action', columns='Automation Status', values='N# Actions').fillna(0).reset_index()

    # Plotting the table graph
    fig, ax = plt.subplots(figsize=(7, 6))  # Adjust the figsize as needed
    ax.axis('tight')
    ax.axis('off')

    # Create the table
    table = ax.table(cellText=pivot_df.values, colLabels=pivot_df.columns, cellLoc='center', loc='center',  bbox=[0, 0, 1, 1])

    formatter = mtick.FuncFormatter(lambda x, _: f'{x:,.0f}')
    for i in range(1, len(pivot_df.columns)):
        for key, cell in table.get_celld().items():
            if key[0] != 0 and key[1] == i:  # Exclude the header row
                cell.get_text().set_text(formatter(int(float(cell.get_text().get_text()))))  # Convert to float first, then to int

    # Set cell height
    table.auto_set_column_width(col=list(range(len(pivot_df.columns))))
    table.scale(1, 1.5)  # Increase cell height (adjust as needed)

    # Set font size
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    # Display the figure
    fig.tight_layout(pad=1)
    
   # return fig
    return pivot_df

# Example usage:
# fig = generate_moderation_action_automation_status_figure(data_ACC)
