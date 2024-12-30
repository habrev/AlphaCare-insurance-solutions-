import pandas as pd

def calculate_kpis(data, segmentation_feature):
    
    # Handle missing values in the dataset
    data['TotalClaims'] = data['TotalClaims'].fillna(0)
    data['TotalPremium'] = data['TotalPremium'].fillna(1e-10)  # Avoid division by 0

    # Replace 0 in TotalPremium to prevent division by zero
    data['TotalPremium'] = data['TotalPremium'].replace(0, 1e-10)

    # Add calculated columns for ClaimFrequency and LossRatio
    data['ClaimFrequency'] = data['TotalClaims'] / data['TotalPremium']
    data['LossRatio'] = data['TotalClaims'] / data['TotalPremium']

    # Clean and map segmentation feature
    valid_genders = ['Male', 'Female']  # Define valid gender values
    data = data[data[segmentation_feature].isin(valid_genders)]  # Filter valid genders
    mapping = {'Female': 'A', 'Male': 'B'}  # Map Female -> A, Male -> B
    data['Group'] = data[segmentation_feature].map(mapping)

    # 1. Risk Differences Across Provinces
    province_kpis = data.groupby(['Group', 'Province']).agg(
        avg_claim_frequency=('ClaimFrequency', 'mean'),
        avg_loss_ratio=('LossRatio', 'mean'),
        total_claims=('TotalClaims', 'sum'),
        total_premium=('TotalPremium', 'sum')
    ).reset_index()

    # 2. Risk Differences Between Zip Codes
    zip_code_kpis = data.groupby(['Group', 'PostalCode']).agg(
        avg_claim_frequency=('ClaimFrequency', 'mean'),
        avg_loss_ratio=('LossRatio', 'mean'),
        total_claims=('TotalClaims', 'sum'),
        total_premium=('TotalPremium', 'sum')
    ).reset_index()

    # 3. Margin Differences Between Zip Codes
    zip_code_margin_kpis = data.groupby(['Group', 'PostalCode']).agg(
        total_premium=('TotalPremium', 'sum'),
        total_claims=('TotalClaims', 'sum')
    ).reset_index()
    zip_code_margin_kpis['gross_margin_percentage'] = (
        (zip_code_margin_kpis['total_premium'] - zip_code_margin_kpis['total_claims'])
        / zip_code_margin_kpis['total_premium'] * 100
    )

    # 4. Risk Differences Between Women and Men
    gender_kpis = data.groupby(['Group', 'Gender']).agg(
        avg_claim_frequency=('ClaimFrequency', 'mean'),
        avg_loss_ratio=('LossRatio', 'mean'),
        total_claims=('TotalClaims', 'sum'),
        total_premium=('TotalPremium', 'sum')
    ).reset_index()

    # Returning all KPIs in a dictionary for easy access
    return {
        'province_kpis': province_kpis,
        'zip_code_kpis': zip_code_kpis,
        'zip_code_margin_kpis': zip_code_margin_kpis,
        'gender_kpis': gender_kpis
    }

# Sample usage
if __name__ == '__main__':


    # Insert the feature for segmentation manually in the notebook
    # Replace 'FeatureX' with the name of the feature you want to use for segmentation
    segmentation_feature = 'FeatureX'  # Insert your specific feature here

    # Call the function to calculate KPIs
    kpi_results = calculate_kpis(data, segmentation_feature)

    # Print the results
    print("\nRisk KPIs by Province:")
    print(kpi_results['province_kpis'])

    print("\nRisk KPIs by Zip Code:")
    print(kpi_results['zip_code_kpis'])

    print("\nMargin KPIs by Zip Code:")
    print(kpi_results['zip_code_margin_kpis'])

    print("\nRisk KPIs by Gender:")
    print(kpi_results['gender_kpis'])

    # Optionally, print the first few rows to see the segmented data
    print("\nSegmented Data (first few rows):")
    print(data[['Group', segmentation_feature]].head())

    # Check the distribution of groups
    print("\nGroup Distribution:")
    print(data['Group'].value_counts())
