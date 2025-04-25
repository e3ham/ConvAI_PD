
import sys
import re

# Read the ecapa.py file
with open("ecapa.py", "r") as f:
    content = f.read()

# Add import for our custom module
if "import custom_features" not in content:
    import_line = "import custom_features"
    import_pos = content.find("import speechbrain as sb")
    if import_pos >= 0:
        content = content[:import_pos] + import_line + "\n" + content[import_pos:]

# Write the modified content back
with open("ecapa.py", "w") as f:
    f.write(content)
