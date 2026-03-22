import re

with open('index.html', 'r', encoding='utf-8') as f:
    content = f.read()

with open('_embedded_data.txt', 'r', encoding='utf-8') as f:
    new_data = f.read().strip()

# Find start and end by brace counting
start = content.find('const allData = {')
if start < 0:
    match = None
else:
    i = content.index('{', start)
    depth = 1
    while i < len(content) and depth > 0:
        i += 1
        c = content[i]
        if c == '{': depth += 1
        elif c == '}': depth -= 1
    end = i + 1  # position after }
    if content[end:end+1] == ';':
        end += 1
    match = True
if match:
    new = 'const allData = ' + new_data + ';'
    content = content[:start] + new + content[end:]
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Replaced allData')
else:
    print('Pattern not found')
