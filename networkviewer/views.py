from django.shortcuts import render
import random
import difflib 
from .network_functions import (
    show_chemical_network,
    show_company_network_pyvis,
    show_uni_network_pyvis,
    show_researcher_network_pyvis_from_row,
    comparing_researchers,
    show_chem_connections,
    show_company_connections,
    show_uni_connections,
    show_res_connections,
    chem_per_row,
    no_dup_comp,
    comparing_unis
)



def get_close_matches_custom(query, valid_names, n = 3, cutoff=0.6):
    return difflib.get_close_matches(query, valid_names, n=n, cutoff=cutoff)

def home_view(request):
    return render(request, 'networkviewer/home.html', {'show_main_nav': False})

def chemical_view(request):
    chemical = None
    inchikey = None
    iframe = None
    message = None
    connections = None 

    all_chemical_names = sorted((name for names in chem_per_row['chemical'] for name in names))
    example_chemicals = [
        ("Aspirin", "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"),
        ("Caffeine", "RYYVLZVUVIJVGH-UHFFFAOYSA-N"),
        ("Glucose", "WQZGKKKJIJFFOK-GASJEMHNSA-N"),
        ("Iron", ""),
        ("Goethite", ""),
        ("PAHs", ""),
        ("Au",""),
        ("Copper","")
    ]

    random_examples = random.sample(example_chemicals, 3)

    if request.method == 'POST':
        chemical = request.POST.get('chemical', '').strip()
        inchikey = request.POST.get('inchikey', '').strip()

        if inchikey:  # If InChIKey is provided, use the new function
            found = show_chemical_network(chemical, inch=inchikey)
            connections = show_chem_connections(inchikey=inchikey)
        elif chemical:  # If only chemical name is provided, use the old function
            found = show_chemical_network(chemical, inch='Error')
            connections = show_chem_connections(chemical)
        else:
            found = False

        if found:
            if chemical and inchikey and inchikey != 'Error':
                safe_chemical = chemical.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')
                safe_inch = inchikey.replace('/', '_').replace('\\', '_').replace('-', '_')
                iframe = f"/static/network_{safe_chemical}_{safe_inch}.html"
            else:
                safe_chemical = chemical.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')
                iframe = f"/static/network_{safe_chemical}_no_inchikey.html"
        else:
            if chemical:
                suggestions = get_close_matches_custom(chemical, all_chemical_names)
                if suggestions:
                    message = "Did you mean: " + ", ".join([f"<span style='color:red'>{s}</span>" for s in suggestions])
                else:
                    message =  F"Chemical '{chemical}' not found"
            else:
                message = f"Chemical '{chemical}' or InChIKey '{inchikey}' not found"

    context = {'chemical': chemical, 
               'inchikey': inchikey, 
               'iframe': iframe, 
               'message': message,
               'connections': connections,
               'show_main_nav': True,
               'example_chemicals': random_examples,
               'all_chemical_names': all_chemical_names
            }
    return render(request, 'networkviewer/chemical_view.html', context)

def company_view(request):
    company = None
    iframe = None
    message = None
    connections = None

    category_options = ['Affiliations', 'Chemicals', 'Researchers', 'Universities']
    chemical_group_options = ['All', 'Organic']
    sep_country_options = [False, True]

    sep_country = False
    category = 'Affiliations'
    chemical_group = 'All'

    all_company_names= sorted(set(no_dup_comp))
    example_companies = [
        "Dow Chemical Company",
        "U.S. Department of Energy",
        "U.S. Department of Agriculture",
        "BASF Corporation",
        "Agilent Foundation",
        "Natural Science Foundation of China",
        "U.S. Department of Defense"
    ]

    random_examples = random.sample(example_companies, 3)

    if request.method == 'POST':
        company = request.POST.get('company')
        category = request.POST.get('category', 'Affiliations')
        chemical_group = request.POST.get('chemical_group', 'All')
        sep_country = request.POST.get('sep_country', 'False')
        sep_country = True if sep_country == 'True' or sep_country is True else False  # <-- fix here
        found = show_company_network_pyvis(company, category=category, chemical_group=chemical_group, sep_country = sep_country)
        if found:
            safe_company = company.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')
            safe_category = category.replace(' ', '_')
            
            if category == 'Chemicals':
                if chemical_group == 'All':
                    iframe = f"/static/network_{safe_company}_{safe_category}_all.html"
                elif chemical_group == 'Organic':
                    iframe = f"/static/network_{safe_company}_{safe_category}_organic.html"
            elif category == 'Affiliations':
                if sep_country:
                    iframe = f"/static/network_{safe_company}_{safe_category}_by_country.html"
                else:
                    iframe = f"/static/network_{safe_company}_{safe_category}_combined.html"
            else:
                iframe = f"/static/network_{safe_company}_{safe_category}.html"
            connections = show_company_connections(company)
        else:
            suggestions = get_close_matches_custom(company, all_company_names)
            if suggestions:
                message = "Did you mean: " + ", ".join([f"<span style='color:red'>{s}</span>" for s in suggestions])
            else:
                message = f"Company '{company}' not found"
    context = {
        'company': company,
        'iframe': iframe, 
        'message': message, 
        'category': category, 
        'sep_country': sep_country,
        'chemical_group': chemical_group,
        'category_options': category_options,
        'chemical_group_options': chemical_group_options,
        'sep_country_options': sep_country_options,
        'connections': connections,
        'show_main_nav': True,
        'example_companies': random_examples,
        'all_company_names': all_company_names
    }
    return render(request, 'networkviewer/company_view.html', context)
def university_view(request):
    university = None
    iframe = None
    message = None
    connections = None 

    category_options = ['Chemicals','Funding Sources']
    chemical_group_options = ['All', 'Organic']
    all_university_names = sorted(comparing_unis['University'].dropna().unique())
    category = 'Funding Sources'
    chemical_group = 'All'

    example_universities = [
        "Harvard University",
        "Stanford University",
        "Massachusetts Institute of Technology",
        "University of Cambridge",
        "University of Oxford",
        "California Institute of Technology",
        "Princeton University",
        "Yale University",
        "University of Chicago",
        "Columbia University",
        "New York University"
    ]
    # Pick 3 random examples
    random_examples = random.sample(example_universities, 3)

    if request.method == 'POST':
        university = request.POST.get('university')
        category = request.POST.get('category', 'Funding Sources')
        chemical_group = request.POST.get('chemical_group', 'All')
        found = show_uni_network_pyvis(university, category=category, chemical_group=chemical_group)
        if found:
            safe_uni = university.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')
            safe_category = category.replace(' ', '_')
            
            if category == 'Chemicals':
                if chemical_group == 'All':
                    iframe = f"/static/network_{safe_uni}_{safe_category}_all.html"
                elif chemical_group == 'Organic':
                    iframe = f"/static/network_{safe_uni}_{safe_category}_organic.html"
            else:
                iframe = f"/static/network_{safe_uni}_{safe_category}.html"
            connections = show_uni_connections(university)
        else:
            suggestions = get_close_matches_custom(university, all_university_names)
            if suggestions:
                message = "Did you mean: " + ", ".join([f"<span style='color:red'>{s}</span>" for s in suggestions])
            else:
                message = f"University '{university}' not found"
    context = {'university': university,
               'iframe': iframe,
                'message': message, 
                'category': category, 
                'chemical_group': chemical_group,
                'category_options': category_options,
                'chemical_group_options': chemical_group_options,
                'connections': connections,
                'show_main_nav': True,
                'example_universities': random_examples,
                'all_university_names': all_university_names
                }
    return render(request, 'networkviewer/university_view.html', context)

def researcher_view(request):
    researcher = None
    iframe = None
    message = None
    matches = []
    selected_index = None
    combine = False
    connections = None 

    all_researcher_names = sorted(comparing_researchers['Researcher'].dropna().unique())
    example_researchers = [
        'Abrahamsson, Dimitri',
        'Jiang, Guibin',
        'Yang, Xin',
        'Xie, Hongyu',
        'Nikiforov, Vladimir A.',
        'Lynch, Iseult',
        'Pan, Wenxiao',
        'Restituito, Sophie',
        'Kyrtopoulos, Soterios A.',
        'Wei, Jing'

    ]
    random_examples = random.sample(example_researchers, 3)
    
    if request.method == 'POST':
        researcher = request.POST.get('researcher')
        selected_index = request.POST.get('selected_index')
        combine = request.POST.get('combine', '') == 'on'
        # Find all matches
        all_matches = comparing_researchers[comparing_researchers['Researcher'].str.lower() == researcher.lower()]
        matches = all_matches.to_dict('records')

        if not matches:
            suggestions = get_close_matches_custom(researcher, all_researcher_names)
            if suggestions:
                message = "Did you mean: " + ", ".join([f"<span style='color:red'>{s}</span>" for s in suggestions])
            else:
                message = f"Researcher '{researcher}' not found"
        elif len(matches) == 1:
            # Only one match, generate graph immediately
            row = matches[0]
            found = show_researcher_network_pyvis_from_row(row)
            if found:
                safe_researcher = researcher.replace(' ', '_').replace(',', '').replace('/', '_').replace('\\', '_').replace('.', '_')
                safe_aff = str(row['Affiliation'])[:20].replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')
                iframe = f"/static/network_{safe_researcher}_{safe_aff}.html"
            connections = show_res_connections(researcher=researcher)
        elif selected_index is not None or combine:
            if combine:
                # Combine all companies and affiliations
                all_companies = sum(all_matches['Companies'], [])
                unique_affiliations = all_matches['Affiliation'].dropna().unique()
                combined_aff = '; '.join(unique_affiliations)
                row = {
                    'Researcher': researcher,
                    'Affiliation': combined_aff,
                    'Companies': all_companies
                }
            else:
                row = matches[int(selected_index)]
            found = show_researcher_network_pyvis_from_row(row)
            if found:
                safe_researcher = researcher.replace(' ', '_').replace(',', '').replace('/', '_').replace('\\', '_').replace('.', '_')
                safe_aff = str(row['Affiliation'])[:20].replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')
                iframe = f"/static/network_{safe_researcher}_{safe_aff}.html"
        # If multiple matches and no selection yet, just show the options
            connections = show_res_connections(researcher)
    context = {
        'researcher': researcher,
        'iframe': iframe,
        'message': message,
        'matches': matches,
        'selected_index': selected_index,
        'combine': combine,
        'connections': connections,
        'show_main_nav': True,
        'example_researchers': random_examples,
        'all_researcher_names': all_researcher_names
    }
    return render(request, 'networkviewer/researcher_view.html', context)

def about_view(request):
    return render(request, 'networkviewer/about.html')
def data_view(request):
    return render(request, 'networkviewer/data.html')
def contact_view(request):
    return render(request, 'networkviewer/contact.html')