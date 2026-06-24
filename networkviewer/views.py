from django.shortcuts import render
import random
import difflib
import os
from django.conf import settings
from django.http import JsonResponse, FileResponse, Http404
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
    comparing_unis,
    get_pubchem_image_url,
    get_top_chemicals_for_company,
    get_pubchem_description,
    get_wikipedia_description_fundingsource,
    funding_source_table_df,
    get_funding_source_row,
    parse_chemicals_list,
    obtain_inchikey_from_pubchem,
    get_company_funding_rows,
    get_university_rows,
    get_researcher_matches,
    get_researcher_rows,
    get_chemical_row
)

# Directory where network_functions writes the generated pyvis graphs.
_GRAPH_DIR = os.path.join(settings.BASE_DIR, 'staticfiles')


def serve_network_graph(request, filename):
    """Serve a generated network graph HTML file.

    The graphs are written at runtime into the staticfiles directory. Django's
    development server serves /static/ via finders (source dirs only), so those
    runtime files are not reachable that way. Serving them through this view
    works identically in development and production.
    """
    # Only allow plain generated graph filenames; reject any path traversal.
    if (not filename.endswith('.html')
            or '/' in filename or '\\' in filename or '..' in filename):
        raise Http404('Not found')
    filepath = os.path.join(_GRAPH_DIR, filename)
    if not os.path.isfile(filepath):
        raise Http404('Graph not found')
    return FileResponse(open(filepath, 'rb'), content_type='text/html')


def resolve_case_insensitive_name(query, valid_names):
    if query is None:
        return query
    query_str = str(query).strip()
    if not query_str:
        return query_str

    lookup = {}
    for name in valid_names:
        name_str = str(name).strip()
        if name_str and name_str.lower() not in lookup:
            lookup[name_str.lower()] = name_str

    return lookup.get(query_str.lower(), query_str)


def get_close_matches_custom(query, valid_names, n=3, cutoff=0.6):
    if query is None:
        return []
    query_str = str(query).strip()
    if not query_str:
        return []

    normalized_map = {}
    for name in valid_names:
        name_str = str(name).strip()
        if name_str and name_str.lower() not in normalized_map:
            normalized_map[name_str.lower()] = name_str

    matched_keys = difflib.get_close_matches(
        query_str.lower(),
        list(normalized_map.keys()),
        n=n,
        cutoff=cutoff,
    )
    return [normalized_map[key] for key in matched_keys]

def home_view(request):
    return render(request, 'networkviewer/home.html', {'show_main_nav': False})

def chemical_view(request):
    chemical = None
    inchikey = None
    iframe = None
    message = None
    connections = None 
    image_url = None
    description = None

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
        inchikey = inchikey.upper()

        if chemical:
            chemical = resolve_case_insensitive_name(chemical, all_chemical_names)


        if inchikey:  # If InChIKey is provided, use the new function
            row = get_chemical_row(chemical=chemical, inchikey=inchikey)
            image_url = get_pubchem_image_url(chemical, inchikey)
            description = get_pubchem_description(chemical, inchikey)
            found = show_chemical_network(chemical, inch=inchikey, row=row)
            connections = show_chem_connections(inchikey=inchikey, row=row)
        elif chemical:  # If only chemical name is provided, use the old function
            row = get_chemical_row(chemical=chemical)
            image_url = get_pubchem_image_url(chemical, inchikey)
            description = get_pubchem_description(chemical)
            found = show_chemical_network(chemical, inch='Error', row=row)
            connections = show_chem_connections(chemical, row=row)
        else:
            found = False

        if found:
            if chemical and inchikey and inchikey != 'Error':
                safe_chemical = chemical.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')
                safe_inch = inchikey.replace('/', '_').replace('\\', '_').replace('-', '_')
                iframe = f"/networks/network_{safe_chemical}_{safe_inch}.html"
            else:
                safe_chemical = chemical.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')
                iframe = f"/networks/network_{safe_chemical}_no_inchikey.html"
        else:
            if chemical:
                inchikey = obtain_inchikey_from_pubchem(chemical)
                if inchikey:
                    row = get_chemical_row(chemical=chemical, inchikey=inchikey)
                    image_url = get_pubchem_image_url(chemical, inchikey)
                    description = get_pubchem_description(chemical, inchikey)
                    found = show_chemical_network(chemical, inch=inchikey, row=row)
                    connections = show_chem_connections(inchikey=inchikey, row=row)
                if found:
                    if chemical and inchikey and inchikey != 'Error':
                        safe_chemical = chemical.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')
                        safe_inch = inchikey.replace('/', '_').replace('\\', '_').replace('-', '_')
                        iframe = f"/networks/network_{safe_chemical}_{safe_inch}.html"
                else:
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
               'all_chemical_names': all_chemical_names,
               'image_url': image_url,
               'description': description
            }
    return render(request, 'networkviewer/chemical_view.html', context)

def company_view(request):
    company = None
    iframe = None
    message = None
    connections = None
    description = None


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

    if not company and request.method == 'GET':
        company = request.GET.get('company_name', '')

    if request.method == 'POST':
        company = request.POST.get('company', '').strip()
        company = resolve_case_insensitive_name(company, all_company_names)
        category = request.POST.get('category', 'Affiliations')
        chemical_group = request.POST.get('chemical_group', 'All')
        sep_country = request.POST.get('sep_country', 'False')
        sep_country = True if sep_country == 'True' or sep_country is True else False  # <-- fix here
        company_funding_rows = get_company_funding_rows(company)
        found = show_company_network_pyvis(company, category=category, chemical_group=chemical_group, sep_country = sep_country, company_funding_rows=company_funding_rows)
        if found:
            safe_company = company.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')
            safe_category = category.replace(' ', '_')
            
            if category == 'Chemicals':
                if chemical_group == 'All':
                    iframe = f"/networks/network_{safe_company}_{safe_category}_all.html"
                elif chemical_group == 'Organic':
                    iframe = f"/networks/network_{safe_company}_{safe_category}_organic.html"
            elif category == 'Affiliations':
                if sep_country:
                    iframe = f"/networks/network_{safe_company}_{safe_category}_by_country.html"
                else:
                    iframe = f"/networks/network_{safe_company}_{safe_category}_combined.html"
            else:
                iframe = f"/networks/network_{safe_company}_{safe_category}.html"
            connections = show_company_connections(company, company_funding_rows=company_funding_rows)
        else:
            suggestions = get_close_matches_custom(company, all_company_names)
            if suggestions:
                message = "Did you mean: " + ", ".join([f"<span style='color:red'>{s}</span>" for s in suggestions])
            else:
                message = f"Company '{company}' not found"
    if iframe and company and company.strip():
        description = get_wikipedia_description_fundingsource(company.strip())
    context = {
        'company': company,
        'iframe': iframe, 
        'message': message, 
        'category': category, 
        'description' : description,
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
        university = request.POST.get('university', '').strip()
        university = resolve_case_insensitive_name(university, all_university_names)
        category = request.POST.get('category', 'Funding Sources')
        chemical_group = request.POST.get('chemical_group', 'All')
        uni_rows = get_university_rows(university)
        found = show_uni_network_pyvis(university, category=category, chemical_group=chemical_group, uni_rows=uni_rows)
        if found:
            safe_uni = university.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')
            safe_category = category.replace(' ', '_')
            
            if category == 'Chemicals':
                if chemical_group == 'All':
                    iframe = f"/networks/network_{safe_uni}_{safe_category}_all.html"
                elif chemical_group == 'Organic':
                    iframe = f"/networks/network_{safe_uni}_{safe_category}_organic.html"
            else:
                iframe = f"/networks/network_{safe_uni}_{safe_category}.html"
            connections = show_uni_connections(university, uni_rows=uni_rows)
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
        researcher = request.POST.get('researcher', '').strip()
        selected_index = request.POST.get('selected_index')
        combine = request.POST.get('combine', '') == 'on'
        # Find all matches
        all_matches = get_researcher_matches(researcher)
        matches = all_matches.to_dict('records')
        researcher_rows = get_researcher_rows(researcher)

        if not matches:
            suggestions = get_close_matches_custom(researcher, all_researcher_names)
            if suggestions:
                message = "Did you mean: " + ", ".join([f"<span style='color:red'>{s}</span>" for s in suggestions])
            else:
                message = f"Researcher '{researcher}' not found"
        else:
            researcher = all_matches.iloc[0]['Researcher']

        if matches and len(matches) == 1:
            # Only one match, generate graph immediately
            row = matches[0]
            found = show_researcher_network_pyvis_from_row(row, researcher_rows=researcher_rows)
            if found:
                safe_researcher = researcher.replace(' ', '_').replace(',', '').replace('/', '_').replace('\\', '_').replace('.', '_')
                safe_aff = str(row['Affiliation'])[:20].replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')
                iframe = f"/networks/network_{safe_researcher}_{safe_aff}.html"
            connections = show_res_connections(researcher=researcher, matches=all_matches, researcher_rows=researcher_rows)
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
            found = show_researcher_network_pyvis_from_row(row, researcher_rows=researcher_rows)
            if found:
                safe_researcher = researcher.replace(' ', '_').replace(',', '').replace('/', '_').replace('\\', '_').replace('.', '_')
                safe_aff = str(row['Affiliation'])[:20].replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')
                iframe = f"/networks/network_{safe_researcher}_{safe_aff}.html"
        # If multiple matches and no selection yet, just show the options
            connections = show_res_connections(researcher, matches=all_matches, researcher_rows=researcher_rows)
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

def funding_table_view(request):
    periodic_data = []
    for _, row in funding_source_table_df.iterrows():
        count_value = row.get('study_count', row.get('count', 0))
        periodic_data.append({
            'company': row.get('company', ''),
            'count': int(count_value) if count_value is not None else 0,
            'classification': row.get('classification', 'Unknown')
        })

    context = {
        'periodic_data': periodic_data,
        'show_main_nav': True
    }
    return render(request, 'networkviewer/funding_table.html', context)

def get_company_details(request):
    """AJAX endpoint to get detailed company information for modal"""
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    company_name = request.GET.get('company_name', '').strip()
    if not company_name:
        return JsonResponse({'error': 'Company name required'}, status=400)

    try:
        row_data = get_funding_source_row(company_name)
        if row_data:
            # Parse chemicals with counts from semicolon-separated string
            top_chemicals = parse_chemicals_list(row_data.get('top_chemicals', ''))[:5]

            description_text = row_data.get('description')
            description = {
                'title': row_data.get('company', company_name),
                'description': description_text or '',
                'url': row_data.get('wiki_url') or '',
                'thumbnail': None,
            } if description_text else None

            return JsonResponse({
                'success': True,
                'top_chemicals': top_chemicals,
                'description': description,
            })

        top_chemicals = get_top_chemicals_for_company(company_name, limit=5)
        description = get_wikipedia_description_fundingsource(company_name)

        return JsonResponse({
            'success': True,
            'top_chemicals': top_chemicals,
            'description': description,
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)