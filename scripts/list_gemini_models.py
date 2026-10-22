import os, requests, sys
k = os.environ.get("GEMINI_API_KEY")
if not k:
    sys.exit("GEMINI_API_KEY not set")
d = requests.get("https://generativelanguage.googleapis.com/v1beta/models?key="+k, timeout=30).json()
for m in d.get("models", []):
    if "generateContent" in m.get("supportedGenerationMethods", []):
        print(m["name"], "|", m.get("supportedGenerationMethods"))
