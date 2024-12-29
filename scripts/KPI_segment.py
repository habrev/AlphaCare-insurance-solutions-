import pandas as pd

def calculate_kpis(data, segmentation_feature):
    """
    This function calculates various KPIs for Risk and Margin differences
    based on data segmentation into Group A (Control) and Group B (Test).

    Parameters:
    - data: The input DataFrame containing the data.
    - segmentation_feature: The feature used for segmenting the data into Group A and Group B.
    
    Returns:
    - A dictionary containing the KPIs for Risk and Margin across Provinces, Zip Codes, and Gender.
    """
    
    # Segment data into Group A (Control) and Group B (Test) based on the feature
    data['Group'] = data[segmentation_feature].apply(lambda x: 'B' if x else 'A')

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
