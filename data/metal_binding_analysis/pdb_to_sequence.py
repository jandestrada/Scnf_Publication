import pandas as pd 
import sys



def drop_consecutive_duplicates(values):
    output = []
    prev_value = None
    for val in values:
         if prev_value == None:
              output.append(val)
              prev_value = val
              continue
         if prev_value == val:
              continue
         else:
              output.append(val)
              prev_value = val
    return output






file_path = sys.argv[1]

pdb_file = open(file_path, 'r').read().splitlines()
raw_sequence = []
for line in pdb_file:
     if len(line.split()) > 1:
          raw_sequence.append(line.split()[3])
     else:
          break

seq_series = pd.Series(raw_sequence)

# idea stolen from https://stackoverflow.com/questions/19463985/pandas-drop-consecutive-duplicates
seq_series = seq_series.loc[seq_series.shift() != seq_series]
#seq_series = drop_consecutive_duplicates(seq_series)

# print(seq_series)
# print(len(seq_series))

translator = {
     'ALA':'A',
     'ARG':'R',
     'ASN':'N',
     'ASP':'D',
     'CYS':'C',
     'GLU':'E',
     'GLN':'Q',
     'GLY':'G',
     'HIS':'H',
     'ILE':'I',
     'LEU':'L',
     'LYS':'K',
     'MET':"M",
     'PHE':'F',
     'PRO':'P',
     'SER':'S',
     'THR':'T',
     'TRP':'W',
     'TYR':'Y',
     'VAL':'V'
}

output_seq = [translator[aa] for _, aa in seq_series.items()]

print("".join(output_seq))

	
