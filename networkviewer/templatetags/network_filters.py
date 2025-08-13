from django import template
import re

register = template.Library()

@register.filter
def colorize_categories(text):
    """Colorize funding source categories in text"""
    color_map = {
        'Government': '#FF6B6B',
        'University': '#96CEB4', 
        'Foundation': '#4ECDC4',
        'Company': '#FFEAA7',
        'Unknown': '#DDD6FE'
    }
    
    for category, color in color_map.items():
        pattern = f'\\[{category}\\]'
        replacement = f'<span style="color: {color}; font-weight: bold;">[{category}]</span>'
        text = re.sub(pattern, replacement, text)
    
    return text