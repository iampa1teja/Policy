# __init__.py 
"""RPD Policy Package"""

try: 
    import lerobot 
except ImportError:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package"
    )

from .configuration_RPDPolicy import RPDPolicyConfig 
