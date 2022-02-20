from ete3 import NCBITaxa
# from io import StringIO
from Bio import SeqIO #, Phylo
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
ncbi = NCBITaxa()

class FASTASeq:
    
    """
    Creates cutomised FASTA sequence object with features specified in the init function.
    """
    
    def __init__(self, entry, sequence, protein):
        """
        entry           : Protein sequence entry identifier
        sequence        : Protein sequence
        protein_name    : Name of the protein
        organism        : Scientific name (i.e. genus species) of organism
        status          : Whether the entry has been reviewed or not ()
        """
        self.entry = entry
        self.seq = sequence
        self.protein_name = protein
        self.organism = None
        self.status = None


    def getFASTA(self):
        """
        class method:
            formats the sequence output format
        """

        if self.status != None:
            fasta = f">{self.status}|{self.entry}|{self.protein_name} OS={self.organism}\n{self.seq}"
        else:
            fasta = f">{self.protein_name} OS={self.organism}\n{self.seq}"
        return fasta


# Apply while reading sequence
def getSequenceFeatures(seqObj, entry, **kwargs):
    """
    params:
        seqObj      : Sequence object instantiated with FASTASeq()
        entry       : Protein sequence entry identifier
        
        **kwargs    : either of the following keyword arguments ->
            protein     : Name of the protein
            organism    : Scientific name (i.e. genus species) of organism
            status      : Whether the entry has been reviewed or not ()
    returns:
        Sequence object with features
    """

    if seqObj.entry == entry:
        if seqObj.protein_name == None:
            seqObj.protein_name = kwargs.get('protein', None)
        else:
            pass
        if seqObj.organism == None:
            seqObj.organism = kwargs.get('organism', None)
        else:
            pass
        if seqObj.status == None:
            seqObj.status = kwargs.get('status', None)
        else:
            pass
    else:
        pass
    return seqObj

def read_fasta(fastaFileName: str):
    """
    params:
        Fasta file
    returns:
        list of (entry, FASTASeq object)
    """
    with open(fastaFileName, 'r') as f:
        file = f.read()
        
    # >sp|O11457|VGP_EBOG4 Envelope glycoprotein OS=Zaire ebolavirus (strain Gabon-94) OX=128947 GN=GP PE=1 SV=1
    
    file = file.split('>') # Divide to individual sequences
    file = list(filter(None, file)) # Remove empty spaces
    
    def split_sequences(seq):
        seq = seq.split('\n')
        entry = seq[0].split('|')[1]
        entryname_protein_name = seq[0].split('|')[2].split('OS=')[0]
        protein_name_lst = entryname_protein_name.split(' ')[1:]
        protein_name = ' '.join(protein_name_lst).strip()
        seq = '\n'.join(seq[1:])
        seqObj = FASTASeq(entry, seq, protein_name)
        
        return (entry, seqObj)
    
    fasta_sequence_pair = list(map(split_sequences, file))
    
    fasta_sequence_pair = sorted(fasta_sequence_pair) # sorts tuples by key
    return fasta_sequence_pair # returns tuple of (entry, FASTASeq object)


# Uses ete3 taxonomy to obtain the rank of an organism from its taxonomic identifier
def getRankName(idf: int, rank: str):
    """
    params:
        idf     : Taxonomic identifier
        rank    : Query taxonomic rank
    returns:
        organism rank  name
    
    example:
    species_name = getRankName(9606, 'species')
    print(species_name)
    >>> Homo Sapiens
    """
    dict = {} # Empty dictionary for later use to reorganise the one returned by ete3
    try:
        for key, value in ncbi.get_rank(ncbi.get_lineage(idf)).items():
            dict[value] = key # Dictionary reorganisation
        if f'{rank}' in dict.keys():
            name_dict = ncbi.get_taxid_translator([dict[f'{rank}']])
            return str(name_dict[dict[f'{rank}']]) # Output rank
        else:
            name_dict = ncbi.get_taxid_translator([idf])
            return str(name_dict[idf]) # Output rank
    except Exception as e:
        print(e)
        return None # Output if there is no match in the ete3 database

# Function to get rank taxonomic identifier from given identifier
def getRankID(idf: int, rank: str):
    """
    params:
        idf     : Taxonomic identifier
        rank    : Query taxonomic rank
    returns:
        organism rank identifier
    
    example:
    species_idf = getRankName(9606, 'species')
    print(species_idf)
    >>> 9606
    """
    dictn = {}
    
    try:
        for key, value in ncbi.get_rank(ncbi.get_lineage(idf)).items():
            dictn[value] = key
        if f'{rank}' in dictn.keys():
            return dictn[f'{rank}']
        else:
            return idf
    except Exception as e:
        print(e)
        return None

# Function to get taxonomic identifier from organism name
def getIDfromName(org_name: str):
    """
    params:
        org_name : Organism name

    returns:
        organism species taxonomic identifier
    """
    try:
        name_dict = ncbi.get_name_translator([f'{org_name}'])
        org_id = name_dict[f'{org_name}'][0]
        rank_dict = ncbi.get_rank([org_id])
        if rank_dict[org_id] != 'species':
            try:
                org_species = getRankName(org_id, 'species')
                name_dict = ncbi.get_name_translator([f'{org_name}'])
                org_id = name_dict[f'{org_name}'][0]
                return org_id
            except Exception as e:
                print("can't get species from higher rank")
                print(e)
        else:
            return org_id
    except Exception as e:
        print(e)

def UpdateMain(data1, data2):
    """
    Updates main dataframe by appending with notnull values from the second dataframe
    params:
        data1 : main dataframe
        data2 : second dataframe
    returns:
        a tuple with updated dataframe 1 and updated dataframe 2 (keeping only null)
    """
    data1 = data1.append(data2[~data2['Virus hosts'].isnull()])
    data2 = data2[data2['Virus hosts'].isnull()]
    return (data1, data2)

def UpdateHosts(data1, data2, data1_v_taxid, data2_v_taxid):   # updates the viruses hosts
    """
    params
        data1           : main dataframe
        data2           : second dataframe
        data1_v_taxid   : main dataframe virus taxonomic identifier column name
        data2_v_taxid   : second dataframe virus taxonomic identifier column name
    returns:
        main dataframe with updated virus host names
    """
    data1 = data1.merge(
        data2,
        left_on=data1_v_taxid,
        right_on=data2_v_taxid,
        how='left')
        
    data1 = (data1 
                .drop(['Virus hosts', data2_v_taxid], axis=1)
                .rename({'Host name':'Virus hosts'}, axis=1))
    return data1

def AggregateHosts(data, species_id, host_name):
    """
    params:
        data        : dataframe
        species_id  : column name of species id (used for grouping)
        host_name   : column name of host names
    returns:
        dataframe
    """

    df = data[[species_id, host_name]]
    df = df.groupby(species_id)[host_name].apply(list)
    df = df.reset_index(name=host_name)
    df[host_name] = df[host_name].apply(makeSet)
    df[host_name] = df[host_name].apply(makeList)
    df[host_name] = df[host_name].apply('; '.join)
    return df

# def getScientificName(org_name):
#     sci_name = NCBI.search(org_name)
#     try:
#         sci_name = sci_name[f'{org_name}'][0]['ScientificName']
#     except IndexError:
#         sci_name = org_name
#     finally:
#         print("|", end='')
#     return sci_name

def nameMerger(x: str, y):
    """
    params:
        x : species name
        y : species taxonomic id
    """
    new_name = f'{x} [TaxID: {y}]'
    return new_name

def valueChanger(x, dictn={}):
    """
    change x to a new value if found in the dictionary
    params:
        x : x as the original value and a dictionary with old values as keys and new values as values
    
    returns:
        returns new value if x is found in the dictionary and returns x if not found in the dictionary
    """
    if x in dictn.keys():
        value = dictn[x]
    else:
        return x
    return value

def makeList(x):
    if type(x) is not type(float()):
        return list(x)
    else:
        return x
    
def makeSet(x):
    if type(x) is not type(float()):
        return set(x)
    else:
        return x

def rep(word, times, toList=True, sep=' '):
    ret = word+sep
    ret = ret * times
    ret = ret.strip()
    if toList:
        ret = ret.split(sep)
    return ret

def mergeRows(df, id_col, mergecolumn):
    """
    params:
        df          : dataframe
        id_col      : column name to groupby
        mergecolumn : column to merge rows separated by a ;
    Returns:
        dataframe with host names separaed by a ;
    """
    cols = df.columns.tolist()
    cols.remove(id_col)
    cols.remove(mergecolumn)
    app_map = {mergecolumn:'; '.join}
    dictn = dict(zip(cols, rep('first', len(cols))))
    app_map.update(dictn)
    df = df.groupby(id_col, as_index=False).agg(app_map).copy()
    return df

def get_species_name(node_name_string):
    """
    params:
        node
    returns:
        node species name
    """
    try:
        node_name = int(node_name_string)
        return getRankName(node_name, 'species')
    except Exception as e:
        print(e)
        return node_name_string


def createHostNewickTree(hostList):
    """
    params:
        hostList : host taxonomic identifiers as comma separated integers
    returns:
        Newick phyogenetic tree
    """
    hostList = hostList.split(', ')
    hostList = [int(x) for x in hostList]
    tree = ncbi.get_topology(hostList)
    tree.set_species_naming_function(get_species_name)
    for n in tree.get_leaves():
        n.name = n.species
    tree = tree.write(features=["sci_name"])
#     tree = Phylo.read(StringIO(tree), "newick")
    return tree

def save_sequences(df, fileName):
    """
    params:
        df       : dataframe
        fileName : name of file for saving
    return:
        fasta file
    """
    with open(f'{fileName}.fasta', 'a+') as fObj:
        for row in df.to_dict('records'):
            fObj.write(f"{row['Sequence'].getFASTA()}")

#     df.to_csv(f'{fileName}.csv.gz', index=False, compression='gzip')