
# Multi-Brand Luxury Skincare Dashboard

This repository contains a Streamlit app that supports multiple sheets/brands inside the provided Excel workbook.
Files included:
- `ibr final responses for dashboard 2.xlsx` (your original Excel with multiple sheets)
- `streamlit_app_multi_brand.py` (main Streamlit app)
- `requirements.txt` (dependencies)

How to run locally:
1. Install requirements: `pip install -r requirements.txt`
2. Run: `streamlit run streamlit_app_multi_brand.py`
3. The app will load all sheets in the Excel workbook and let you explore / compare brands.

Notes:
- If you deploy to Streamlit Cloud, make sure to upload the Excel file into the repo or change the code to load CSVs instead.
