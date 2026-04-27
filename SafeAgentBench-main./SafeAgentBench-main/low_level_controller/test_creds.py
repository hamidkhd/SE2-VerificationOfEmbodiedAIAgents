import vertexai
from vertexai.generative_models import GenerativeModel
from google.oauth2 import service_account

# 1. Path to your downloaded JSON key
KEY_PATH = "cred.json"

# 2. Manually load the credentials object
creds = service_account.Credentials.from_service_account_file(KEY_PATH)

# 3. Initialize with the OBJECT, not the string path
vertexai.init(
    project="robust-slice-493713-e1", 
    location="us-central1",
    credentials=creds
)

# 4. Test the model
model = GenerativeModel("gemini-2.5-pro")

try:
    response = model.generate_content("What is the best animal in the world. just give 5 top of them. no other explanation!")
    print(f"Success! Response: {response.text}")
    print(creds)
except Exception as e:
    print(f"An error occurred: {e}")
