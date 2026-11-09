"""Read all raw Nsight metrics, retaining unavailable values and exact native CSVs."""
import csv,math
from pathlib import Path
def captures(path):
    with Path(path).open(newline='') as stream:rows=list(csv.reader(stream))
    if len(rows)<3 or rows[0][0]!='ID':raise ValueError('Unexpected Nsight raw CSV')
    names,units=rows[:2]
    return [(dict(zip(names,row)),dict(zip(names,units))) for row in rows[2:] if row and row[0]!='ID']
def metric_rows(context,data,units,repeat,source,stage):
    metadata={'ID','Process ID','Process Name','Host Name','Kernel Name','Context','Stream','Block Size','Grid Size','Device','CC'}
    for name,raw in data.items():
        if name in metadata:continue
        try:available=math.isfinite(float(raw.replace(',','')))
        except ValueError:available=False
        yield {**context,'kernel':data['Kernel Name'],'metric':name,'unit':units.get(name,''),'value':raw.replace(',','') if available else '', 'stage':stage,'profile_repeat':str(repeat),'availability':'available' if available else 'unavailable','reason':'' if available else 'Nsight value unavailable; original retained in raw CSV','source_report':source}

# Nsight counter aliases
ALIASES={
    'dram__bytes_read.sum':('dram__bytes_read.sum','dram__bytes_op_read.sum'),
    'dram__bytes_write.sum':('dram__bytes_write.sum','dram__bytes_op_write.sum'),
    'dram__throughput.avg.pct_of_peak_sustained_elapsed':('dram__throughput.avg.pct_of_peak_sustained_elapsed','gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed'),
}
def metric_aliases(name):return ALIASES.get(name,(name,))
