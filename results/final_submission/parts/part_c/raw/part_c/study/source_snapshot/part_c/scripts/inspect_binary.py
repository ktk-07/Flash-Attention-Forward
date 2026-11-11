"""Save static compiler resources and per-function HMMA instruction evidence."""
import argparse
import csv
import re
import subprocess
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("binary",type=Path)
p.add_argument("output",type=Path)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
commands=[]
resource_command=["cuobjdump","--dump-resource-usage",str(a.binary)]
commands.append(resource_command)
text=subprocess.check_output(resource_command,text=True)
(a.output/"compiler_resources.txt").write_text(text)
records=[]
for symbol,values in re.findall(r" Function (\S+):\n([^\n]+)",text):
    values=dict(re.findall(r"(REG|STACK|SHARED|LOCAL):(\d+)",values))
    records.append({"symbol":symbol,**values})
with (a.output/"compiler_resources.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["symbol","REG","STACK","SHARED","LOCAL"]);w.writeheader();w.writerows(records)
sass_command=["cuobjdump","--dump-sass",str(a.binary)]
commands.append(sass_command)
process=subprocess.Popen(sass_command,stdout=subprocess.PIPE,text=True)
counts={};current=None
for line in process.stdout:
    match=re.search(r"Function : (\S+)",line)
    if match:current=match.group(1);counts[current]=0
    if current and re.search(r"\bHMMA\.",line):counts[current]+=1
if process.wait()!=0:raise SystemExit("SASS inspection failed")
with (a.output/"tensor_core_instructions.csv").open("w",newline="") as f:
    w=csv.writer(f);w.writerow(["symbol","static_hmma_instruction_count"]);w.writerows(counts.items())
(a.output/"inspection_commands.txt").write_text("\n".join(" ".join(c) for c in commands)+"\n")
print(f"Saved resources for {len(records)} functions; HMMA present in {sum(n>0 for n in counts.values())} functions")
