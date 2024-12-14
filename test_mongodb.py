import os
from dagshub import dagshub

# Use the token from the environment variable
dagshub_token = os.getenv('DAGSHUB_TOKEN')
if dagshub_token:
    dagshub.login(token=dagshub_token)
else:
    print("DAGSHUB_TOKEN not found.")
