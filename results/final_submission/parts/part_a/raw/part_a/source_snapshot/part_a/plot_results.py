import argparse,shutil,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import results_schema as schema
from plot_combined import generate

p=argparse.ArgumentParser(description="Regenerate Part A report figures")
p.add_argument('results',type=Path,nargs='?',default=Path(__file__).resolve().parent/'results')
a=p.parse_args()
with tempfile.TemporaryDirectory() as folder:
    out=Path(folder)
    for name in schema.HEADERS:shutil.copy2(a.results/f'{name}.csv',out/f'{name}.csv')
    generate(out)
    figures=a.results/'figures';figures.mkdir(exist_ok=True)
    for path in figures.iterdir():
        if path.suffix in ['.png','.pdf','.svg']:path.unlink()
    for path in (out/'plots/part_a').iterdir():
        if path.suffix in ['.png','.pdf']:shutil.copy2(path,figures/path.name)
print('Saved Part A report figures')
