import subprocess

res = subprocess.run(['modal', 'app', 'logs', 'graphrag-inference'], capture_output=True, text=True, encoding='utf-8', errors='ignore')
with open('logs.txt', 'w', encoding='utf-8') as f:
    f.write(res.stdout[-8000:])
    f.write(res.stderr[-8000:])
