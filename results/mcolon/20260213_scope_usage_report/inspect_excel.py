import pandas as pd
df = pd.read_excel('/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260213_scope_usage_report/imaging_facility_yx1_recharge_2025.xlsx')
print('Shape:', df.shape)
print('Cols:', df.columns.tolist())
print(df.head(10).to_string())
print()
print('Unique advisors:', df['Advisor'].unique() if 'Advisor' in df.columns else 'N/A')
