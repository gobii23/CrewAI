import pdfplumber
import pandas as pd

pdf_path = r"C:\Users\admin\Downloads\Top 1000 Teams.pdf"

#Extract text tables from PDF
all_rows = []
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        # Extract table if available
        tables = page.extract_tables()
        for table in tables:
            for row in table:
                all_rows.append(row)

#Converting into DataFrame
df_preview = pd.DataFrame(all_rows)

#Reset column headers to numeric indices
# df_preview.columns = range(df_preview.shape[1])

#Remove unwanted rows
df_cleaned = df_preview[~df_preview[0].astype(str).str.contains("School Innovation Marathon", na=False)]
df_cleaned = df_cleaned[~df_cleaned[1].astype(str).str.contains("SIM ID", na=False)]

#Reset index
df_cleaned = df_cleaned.reset_index(drop=True)

#Save to Excel
output_path = r"C:\Users\admin\Downloads\Top_1000_Teams_Cleaned(1).xlsx"
df_cleaned.to_excel(output_path, index=False, header=False)

print(f"Cleaned Excel saved at: {output_path}")
