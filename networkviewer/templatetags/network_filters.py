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

    display_map = {
        'Unknown': 'Not Recognized',
    }
    
    for category, color in color_map.items():
        pattern = f'\\[{category}\\]'
        display_label = display_map.get(category, category)
        replacement = f'<span style="color: {color}; font-weight: bold;">[{display_label}]</span>'
        text = re.sub(pattern, replacement, text)
    
    return text