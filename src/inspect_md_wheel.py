"""Inspect the downloaded megadetector wheel (not installed): metadata, dependencies, model URLs."""
import glob
import re
import zipfile

whl = glob.glob("/tmp/mdpkg/*.whl")[0]
print("wheel:", whl.split("/")[-1])
z = zipfile.ZipFile(whl)
meta = next(n for n in z.namelist() if n.endswith("METADATA"))
for line in z.read(meta).decode().splitlines():
    if line.startswith(("Name", "Version", "License", "Requires-Dist", "Home-page", "Project-URL")):
        print(" ", line)
urls = set()
for n in z.namelist():
    if n.endswith(".py"):
        urls |= set(re.findall(r"https://[^'\"\s]+\.pt", z.read(n).decode("utf8", "ignore")))
print("model URLs found in package source:")
for u in sorted(urls):
    print(" ", u)
