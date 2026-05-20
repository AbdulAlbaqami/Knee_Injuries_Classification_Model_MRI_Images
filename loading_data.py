import redivis

# 1. Connect to the dataset on Redivis
user = redivis.user("aimi")
dataset = user.dataset("mrnet_knee_mri_s:4a2c:v1_0")
table = dataset.table("train:5wjf")

# 2. Load the label/index spreadsheet as a Pandas DataFrame
df = table.to_pandas_dataframe(max_results=100)
print(df.head())

# 3. Download ALL 3,390 MRI files directly to your machine/server
# (Uncomment line 12 from your screenshot and specify a target folder)
table.download_files("./mrnet_images")