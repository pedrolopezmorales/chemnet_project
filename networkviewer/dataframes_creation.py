import re

import pandas as pd
from django.conf import settings
import os
from .network_functions import (
        match_items_against_master,
        match_items_against_master_aff,
        split_researchers,
        normalize_name,
        create_researcher_affiliation_pairs_components,
        normalize_comma_name,
        extract_uni_affil,
        extract_country_list,
        no_dup_comp,
        new_no_dup_aff,
        university_keys,
        main,
        classify_companies_series
    )
#creating the main dataframe
def create_main_dataframe():
    data_years = range(2015, 2025)
    dataframes_list = []

    for year in data_years:
        filename = f'esandt_papers_{year}_with_inchikeys.csv'
        file_path = os.path.join(settings.BASE_DIR, 'data', filename)
        df =pd.read_csv(file_path)
        dataframes_list.append(df)

    combined_df = pd.concat(dataframes_list, ignore_index=True)

    combined_path = os.path.join(settings.BASE_DIR, 'data', "esandt_papers_all.csv")
    combined_df.to_csv(combined_path, index=False)
    print(f"Combined dataset created! Total rows: {len(combined_df)}")
#cleansing main dataframe

PLACEHOLDER_CHEMICALS = {
"graphical abstract",
"no chemicals found",
}

def clean_chemicals_cell(value):
    if pd.isna(value) or value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""

    cleaned = []
    for token in text.split(";"):
        token_clean = token.strip()
        if not token_clean:
            continue

    # Remove parenthetical metadata only for placeholder matching
        token_for_match = re.sub(r"\([^)]*\)", "", token_clean).strip().lower()
        if token_for_match in PLACEHOLDER_CHEMICALS:
            continue

        cleaned.append(token_clean)

    return "; ".join(cleaned)

def create_filtered_main_dataframe():
    all_studies_csv_url = "https://ucsf.box.com/shared/static/5igzqwyaiztqhnuj744pi4almbxns9au" #change to whatever all_studies link is 
    all_studies = pd.read_csv(all_studies_csv_url)

    if "DOI" not in all_studies.columns:
        raise KeyError("Input dataframe does not contain a 'DOI' column")

    all_studies = all_studies[
        all_studies["DOI"].astype(str).str.contains("acs.est.", case=False, na=False, regex=False)
    ].reset_index(drop=True)

    all_studies["Chemicals with InChIKey"] = all_studies["Chemicals with InChIKey"].apply(clean_chemicals_cell)
    all_studies = all_studies[
        all_studies["Chemicals with InChIKey"].str.strip() != ""
    ].reset_index(drop=True)

    main_path = os.path.join(settings.BASE_DIR, "data", "esandt_papers_filtered_main.csv")
    all_studies.to_csv(main_path, index=False)
    print(f"Filtered main dataframe saved to {main_path}")
    print(f"Total rows: {len(all_studies)}")

    return all_studies

def create_company_assoc_dataframe():
    
    comparing_companies = main.drop(['DOI', 'URL', 'Year', 'Title', 'Chemicals Mentioned', 'Abstract'], axis=1)

    comparing_companies['Matched Companies'] = match_items_against_master(comparing_companies,'Funding Sources', no_dup_comp)
    comparing_companies['Matched Chemicals'] = comparing_companies['Chemicals with InChIKey'].str.split(';').apply(lambda lst: [x.strip() for x in lst])
    comparing_companies['Matched Affiliations'] = match_items_against_master_aff(comparing_companies,'Affiliations', new_no_dup_aff)
    comparing_companies['Researchers'] = comparing_companies['Authors'].apply(split_researchers)
    comparing_companies['Aff'] = comparing_companies['Affiliations'].apply(
        lambda x: [item.strip() for item in x.split('|')] if isinstance(x, str) and x.strip() != '' else []
    )
    comparing_companies = comparing_companies.drop(['Affiliations','Funding Sources','Chemicals with InChIKey','Authors'],axis=1)

    # having one company per row, with a list of affiliations and chemicals associated alongside them

    match_chem = []
    for idx, (companies, chemicals) in comparing_companies[['Matched Companies', 'Matched Chemicals']].iterrows():
        for company in companies:
            for chemical in chemicals:
                match_chem.append({'Company': company, 'Chemical': chemical})

    match_chem_df = pd.DataFrame(match_chem)
    chemicals_per_company = (
        match_chem_df
        .groupby('Company')['Chemical']
        .agg(lambda x: list(dict.fromkeys(x)))
        .reset_index()
        .rename(columns={'Chemical': 'Chemicals'})
    )
    matched_aff = []
    for idx, (companies, affiliations) in comparing_companies[['Matched Companies', 'Matched Affiliations']].iterrows():
        for company in companies:
            for affiliation in affiliations:
                matched_aff.append({'Company': company, 'Affiliations': affiliation})

    matched_aff_df = pd.DataFrame(matched_aff)

    aff_per_company = (
        matched_aff_df
        .groupby('Company')['Affiliations']
        .agg(lambda x: list(dict.fromkeys(x)))
        .reset_index()
    )

    comparing_companies['Names'] = comparing_companies['Researchers'].apply(
        lambda name_list: [normalize_name(name) for name in name_list]
    )

    comparing_companies['ResearcherAffPairs'] = comparing_companies.apply(create_researcher_affiliation_pairs_components, axis=1)

    re_comp = comparing_companies.explode('ResearcherAffPairs')

    re_comp[['Researcher', 'Aff']] = pd.DataFrame(
        re_comp['ResearcherAffPairs'].tolist(),
        index=re_comp.index
    )

    re_comp = re_comp.explode('Matched Companies')
    re_comp = re_comp.rename(columns={'Matched Companies': 'Company'})

    final_recomp = re_comp[['Company','Researcher', 'Aff']].reset_index(drop=True)

    res_per_comp = (
        final_recomp.groupby('Company')
          .agg({
              'Researcher': list,
              'Aff': list
          })
          .reset_index()
          .rename(columns={
              'Researcher': 'Researchers',
              'Aff': 'Affs'
          })
    )

    company_assoc = pd.merge(aff_per_company, chemicals_per_company, on='Company')
    company_assoc = pd.merge(company_assoc, res_per_comp, on='Company')

    final_recomp['Researcher'] = final_recomp['Researcher'].apply(normalize_comma_name)

    res_per_comp = (
        final_recomp.groupby('Company')
          .agg({
              'Researcher': list,
              'Aff': list
          })
          .reset_index()
          .rename(columns={
              'Researcher': 'Researchers',
              'Aff': 'Affs'
          })
    )

    company_assoc = pd.merge(aff_per_company, chemicals_per_company, on='Company')
    company_assoc = pd.merge(company_assoc, res_per_comp, on='Company')

    company_assoc['Universities'] = company_assoc['Affiliations'].apply(lambda x: extract_uni_affil(x, university_keys))
    company_assoc['Countries'] = company_assoc['Affiliations'].apply(extract_country_list)

    csv_path = os.path.join(settings.BASE_DIR, 'data', 'comparing_fundingsources.csv')
    company_assoc.to_csv(csv_path, index=False)
    print(f"Saved comparing_fundingsources.csv ({len(company_assoc)} rows)")
# Having the affiliations per row

cut_down = main.drop(['DOI', 'URL','Year','Title','Chemicals Mentioned','Abstract','Authors'], axis = 1)

cut_down['Affiliation'] = match_items_against_master_aff(cut_down, 'Affiliations', new_no_dup_aff)
cut_down.drop('Affiliations',axis=1)
cut_down['Chemicals'] = cut_down['Chemicals with InChIKey'].str.split(';').apply(lambda lst: [x.strip() for x in lst])
cut_down['Companies'] = cut_down['Funding Sources'].str.split(';').apply(lambda lst: [x.strip() for x in lst])
cut_down_exploded = cut_down.explode('Affiliation').reset_index(drop=True)
comparing_affiliations = cut_down_exploded.drop(['Affiliations','Funding Sources','Chemicals with InChIKey'], axis =1)

def extract_university(affil, university_keys):
    if pd.isna(affil) or affil is None:
        return None
    affil = str(affil)
    if ',' in affil:
        found = False
        attributes = [a.strip() for a in affil.split(',')]
        for attr in attributes:
            for key in university_keys:
                if (key.lower() in attr.lower()) and (not any(char.isdigit() for char in attr)):
                    if ';' in attr:
                        att = [a.strip() for a in affil.split(';')]
                        for at in att:
                            if key.lower() in at.lower():
                                found = True
                                uni = at.strip()
                    else:
                        found = True 
                        uni = attr.strip()
        if (found == False):
                    return ', '.join(attributes)  # fallback
        else:
            return uni
    return affil
comparing_affiliations['University'] = comparing_affiliations['Affiliation'].apply(lambda x: extract_university(x, university_keys))
comparing_unis = comparing_affiliations.groupby('University').agg({
    'Chemicals': lambda x: sum(x, []),   # Flattens with duplicates
    'Companies': lambda x: sum(x, [])
})
comparing_unis.reset_index(inplace = True)

comparing_unis['Companies'] = comparing_unis['Companies'].apply(classify_companies_series)

comparing_unis.to_csv(os.path.join(settings.BASE_DIR, 'data', 'comparing_unis.csv'), index=False)


# showing researchers and their company funding

reduced = main.drop(['DOI', 'URL','Year','Title','Chemicals Mentioned','Abstract','Chemicals with InChIKey'], axis = 1)

reduced['Researchers'] = reduced['Authors'].apply(split_researchers)
reduced['Aff'] = reduced['Affiliations'].apply(
    lambda x: [item.strip() for item in x.split('|')] if isinstance(x, str) and x.strip() != '' else []
)
reduced['Companies'] = reduced['Funding Sources'].str.split(';').apply(lambda lst: [x.strip() for x in lst])
reduced = reduced.drop(['Authors','Affiliations','Funding Sources'],axis=1)

def create_researcher_affiliation_paris(row):
    researchers = row['Researchers']
    affiliations = row['Aff']
    if len(researchers) > len(affiliations):
        affiliations.extend([''] * (len(researchers) - len(affiliations)))
    elif len(researchers) < len(affiliations):
        affiliations = affiliations[:len(researchers)]
    return list(zip(researchers, affiliations))

reduced['ResearcherAffPairs'] = reduced.apply(create_researcher_affiliation_paris, axis=1)

reduced_expanded = reduced.explode('ResearcherAffPairs')


reduced_expanded[['Researcher', 'Affiliation']] = pd.DataFrame(
    reduced_expanded['ResearcherAffPairs'].tolist(),
    index=reduced_expanded.index
)

final_reduced = reduced_expanded[['Researcher', 'Affiliation', 'Companies']].reset_index(drop=True)

final_reduced['NormalizedName'] = final_reduced['Researcher'].apply(normalize_name)


final_reduced['GroupKey'] = final_reduced['NormalizedName'] + '|' + final_reduced['Affiliation'].str[:20]

comparing_researchers = final_reduced.groupby('GroupKey').agg({
    'Researcher': 'first',
    'Affiliation': lambda affs: max(affs, key=len),  # longest affiliation
    'Companies': lambda lists: sum(lists, [])        # flatten company lists
}).reset_index(drop=True)

comparing_researchers['Companies'] = comparing_researchers['Companies'].apply(classify_companies_series)

comparing_researchers.to_csv(os.path.join(settings.BASE_DIR, 'data', 'comparing_researchers.csv'), index=False)


# Creating a dataframe that has chemicals per row

red_chem = main.drop(['DOI', 'URL','Year','Title','Chemicals Mentioned','Abstract','Authors','Affiliations'], axis = 1)
def parse_chemicals(chem_string):
    chemicals = []
    for entry in chem_string.split(';'):
        entry = entry.strip()
        parts = re.findall(r'\(([^()]*)\)', entry)
        if parts:
            inchikey = parts[-1]  # Last set of parentheses is likely InChIKey
            name = entry[:entry.rfind('(')].strip()
            chemicals.append((name, inchikey))
        else:
            chemicals.append((entry, None))  # No InChIKey found
    return chemicals
def parse_companies(company_string):
    return [c.strip() for c in company_string.split(';') if c.strip()]

records = []
for idx, row in red_chem.iterrows():
    chemicals = parse_chemicals(row['Chemicals with InChIKey'])
    companies = parse_companies(row['Funding Sources'])
    
    for chem_name, inchikey in chemicals:
        for company in companies:
            records.append({
                'chemical': chem_name,
                'inchikey': inchikey,
                'company': company
            })
flat_red_chem = pd.DataFrame(records)
flat_red_chem['group_key'] = flat_red_chem.apply(
    lambda row: row['inchikey'] if row['inchikey'] != 'Error' else row['chemical'],
    axis=1
)

chem_per_row = (
    flat_red_chem
    .groupby(['group_key'])  # Smart key: inchikey or chemical name
    .agg({
        'inchikey': 'first',  # Retain original InChIKey (or Error)
        'chemical': lambda names: sorted(set(names)),  # All name variants
        'company': list  # All associated companies
    })
    .reset_index(drop=True)
)
chem_per_row['company'] = chem_per_row['company'].apply(classify_companies_series)
chem_per_row.to_csv(os.path.join(settings.BASE_DIR, 'data', 'comparing_chemicals.csv'), index=False)
