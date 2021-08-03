import argparse
import pandas as pd
import swifter
import os
from zoonosis_helper_functions import FASTASeq, getSequenceFeatures

os.environ["MODIN_ENGINE"] = "ray"

# csvFile = 'Viral+attachment+to+host+cell+[KW-1161].tab.gz' ############################################## remember to change file name
# df = pd.read_csv(csvFile)  # usecols=['Entry', 'Taxonomic lineage IDs', 'Protein names'])


# fastaFileName = 'uniprot-keyword Virus+entry+into+host+cell+[KW-1160] +fragment no.fasta' ############### remember to change file name

def ReadFasta(fastaFileName):
    with open(fastaFileName, 'r') as f:
        file = f.read()

    file = file.split('>')
    file = list(filter(None, file))

    fasta_sequence_dictionary = {}
    for seq in file:
        seq = seq.split('\n')
        entry = seq[0].split('|')[1]
        seq = '\n'.join(seq[1:])
        # Protein name
        seqObj = FASTASeq(entry, seq)
        fasta_sequence_dictionary[entry] = seqObj

    seq_items = fasta_sequence_dictionary.items() # creates list of key, value tuples
    fasta_dictionary = sorted(seq_items) # sorts tuples by key
    return fasta_dictionary # returns tuple


def groupProteins(csvFile, fasta_dictionary: dict):
    
    df = pd.read_table(csvFile)
    
    df.sort_values(by='Entry', inplace=True)

    objList = [obj for _, obj in fasta_dictionary]
    # for entry, obj in fasta_dictionary:
    #     objList.append(obj)

    df['Sequence'] = objList

    df['Sequence'] = (df.swifter
                        # .progress_bar()
                        .apply(
                            lambda x: getSequenceFeatures(
                            x['Sequence'],
                            x['Entry'],
                            x['Protein name']),
                            # x['Species name']), 
                            axis=1))
    df['Protein name'] = df['Protein name'].str.replace('/', '-')
    df['Protein name'] = df['Protein name'].str.strip()
    df['Protein name'] = df['Protein name'].str.replace(' ', '_')

    df = df.groupby('Protein name', as_index=False).agg(list)
    return df

def groupSave(df):
    for _, row in df.iterrows():
        with open(
            f"{row['Protein name']}.fasta", 'a+') as fObj:

            for seq in row['Sequence']:
                fObj.write(f'{seq.getFASTA()}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        usage="it's usage tip.",
        description='Group protein FASTA sequences by name for downstream MSA alignment')

    parser.add_argument(
        '-f', '--fasta',
        required=True,
        help='Protein FASTA sequence file')

    parser.add_argument(
        '-t', '--tab',
        required=True,
        help='The Species and Hosts tab file with protein names obtained from python_name script')
    
    args = parser.parse_args()
    
    fasta_dictionary = ReadFasta(args.fasta)
    df = groupProteins(args.tab, fasta_dictionary)
    groupSave(df)