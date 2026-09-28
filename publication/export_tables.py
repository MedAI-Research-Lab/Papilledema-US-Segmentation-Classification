"""Reproduce exact final table records as CSV and Markdown, without Word."""
from pathlib import Path
import argparse,csv,hashlib,json,re,zipfile
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent

def local_record(edition,path):
    routing=json.loads((HERE/f'tables/local_table_{edition}.json').read_text('utf-8'))
    if hashlib.sha256(path.read_bytes()).hexdigest()!=routing['source_sha256']:
        raise ValueError('Local supplementary document does not match this edition source checksum.')
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    text=lambda el:''.join(n.text or '' for n in el.findall('.//w:t',ns))
    with zipfile.ZipFile(path) as z:
        children=list(ET.fromstring(z.read('word/document.xml')).find('w:body',ns))
    records=[]
    for i,el in enumerate(children):
        if el.tag!='{'+ns['w']+'}tbl':continue
        caption=text(children[i-1]);number=re.search(r'Table (S?\d+)\.',caption)
        if number and number[1]==routing['number']:
            rows=[[text(c) for c in row.findall('w:tc',ns)] for row in el.findall('w:tr',ns)]
            note=text(children[i+1]) if i+1<len(children) and text(children[i+1]).startswith('Note.') else ''
            records.append({'section':'supplement','number':number[1],'caption':caption,'rows':rows,'note':note})
    assert len(records)==1
    return records[0]

def export(edition,out,supplement_docx=None):
    records=json.loads((HERE/f'tables/edition_{edition}.json').read_text('utf-8'))
    assert len(records)==30
    if supplement_docx:
        destination=out.resolve();repository=HERE.parent.resolve()
        if destination==repository or repository in destination.parents:
            raise ValueError('Tables extracted from a local clinical document must be written outside this repository.')
        records.append(local_record(edition,supplement_docx))
        records.sort(key=lambda r:(r['section']!='main',int(r['number'].lstrip('S'))))
    out.mkdir(parents=True,exist_ok=True)
    document=[]
    for record in records:
        name=record['section']+'_table_'+record['number']
        with (out/f'{name}.csv').open('w',newline='',encoding='utf-8') as f:
            csv.writer(f).writerows(record['rows'])
        with (out/f'{name}.csv').open(newline='',encoding='utf-8') as f:
            assert list(csv.reader(f))==record['rows']
        escape=lambda s:s.replace('|','\\|').replace('\n','<br>')
        rows=record['rows'];md=['## '+record['caption'],'','| '+' | '.join(map(escape,rows[0]))+' |','| '+' | '.join(['---']*len(rows[0]))+' |']
        md.extend('| '+' | '.join(map(escape,row))+' |' for row in rows[1:])
        if record['note']:md.extend(['',record['note']])
        value='\n'.join(md)+'\n';(out/f'{name}.md').write_text(value,'utf-8');document.append(value)
    (out/f'Tables_{edition}.md').write_text('\n'.join(document),'utf-8')
    return len(records)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--edition',choices=[3000,5000],type=int,default=3000);p.add_argument('--out','--output',dest='out',type=Path,required=True)
    p.add_argument('--supplement-docx',type=Path,help='Authorized local source document for the clinical-provenance table; requires an output outside the repository.')
    a=p.parse_args()
    print(json.dumps({'edition':a.edition,'tables':export(a.edition,a.out,a.supplement_docx),'format':'exact text; Markdown/CSV layout, not the original Word typography'}))

if __name__=='__main__':main()
