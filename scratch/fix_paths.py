import re

with open('cut_analysis.py', 'r') as f:
    content = f.read()

# Change sys.path injection since it's now in root
content = content.replace("sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))", "sys.path.insert(0, str(Path(__file__).parent.resolve()))")

# Change DEFAULT_DATA_DIRS
new_dirs = 'DEFAULT_DATA_DIRS = ["Tdis_pfaff5"] # Add your default target folders here'
old_dirs_regex = r'DEFAULT_DATA_DIRS = \[\n.*?\] # Add your default target folders here'
content = re.sub(old_dirs_regex, new_dirs, content, flags=re.DOTALL)

with open('cut_analysis.py', 'w') as f:
    f.write(content)
