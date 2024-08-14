import pickle
from azure.storage.blob import BlobServiceClient, ContentSettings
import io

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
def load_data_from_dataset(selected_index):
    blob_name = datasets[selected_index]
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

# Display available datasets and select one
print("Available datasets:")
for i, dataset in enumerate(datasets):
    print(f"{i + 1}: {dataset}")

selected_index = int(input("Select a dataset by number: ")) - 1

# Load data and extract lists from selected dataset
if 0 <= selected_index < len(datasets):
    data, List_of_companies, List_of_harms, List_of_content_type, List_of_moderation_action, List_of_automation_status = load_data_from_dataset(selected_index)
    print(f"Data loaded from {datasets[selected_index]}")
else:
    print("Invalid selection. Please try again.")
