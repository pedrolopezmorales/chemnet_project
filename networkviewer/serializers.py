from rest_framework import serializers

class ChemicalSearchSerializer(serializers.Serializer):
    chemical = serializers.CharField(max_length=200, required=False, allow_blank=True)
    inchikey = serializers.CharField(max_length=200, required=False, allow_blank=True)

class CompanySearchSerializer(serializers.Serializer):
    company = serializers.CharField(max_length=200)
    category = serializers.ChoiceField(
        choices=['Affiliations', 'Chemicals', 'Researchers', 'Universities'],
        default='Affiliations'
    )
    chemical_group = serializers.ChoiceField(
        choices=['All', 'Organic'],
        default='All'
    )
    sep_country = serializers.BooleanField(default=False)

class UniversitySearchSerializer(serializers.Serializer):
    university = serializers.CharField(max_length=200)
    category = serializers.ChoiceField(
        choices=['Chemicals', 'Funding Sources'],
        default='Funding Sources'
    )
    chemical_group = serializers.ChoiceField(
        choices=['All', 'Organic'],
        default='All'
    )

class ResearcherSearchSerializer(serializers.Serializer):
    researcher = serializers.CharField(max_length=200)
    selected_index = serializers.IntegerField(required=False, allow_null=True)
    combine = serializers.BooleanField(default=False)