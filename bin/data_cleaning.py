#!/usr/local/bin/python

import pandas as pd
import swifter
import numpy as np
import re
import os
import argparse
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from zoonosis_helper_functions import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="%s is for zoonosis data cleaning", usage="")

    parser.add_argument('--uniprot', required=True, help='')
    # parser.add_argument('--uniprotedited', required=True, help='')
    parser.add_argument('--ncbivirusdb', required=True, help='')
    parser.add_argument('--liverpooluni', required=True, help='')
    parser.add_argument('--virusdb', required=True, help='')
    parser.add_argument('--fasta', required=True, help='')

    args = parser.parse_args()

    os.environ["MODIN_ENGINE"] = "ray"

    # Load dataset downloaded from Uniprot
    df = pd.read_table(args.uniprot) #('../data/uniprot-keyword Virus+entry+into+host+cell+[KW-1160] +fragment no.tab.gz')

    df['Virus hosts'] = np.where(df['Virus hosts'].isnull(), '',df['Virus hosts'])

    df['Virus hosts'] = df['Virus hosts'].str.split('; ').apply(set).apply('; '.join)
    # df['Virus hosts'] = (df['Virus hosts'].swifter.progress_bar(enable=True, desc='Removing duplicate host names'))
    # df['Virus hosts'] = (df['Virus hosts'].swifter.progress_bar(enable=True, desc='Joining host names list'))

    df['Protein names'] = df['Protein names'].str.split('; ').apply(set).apply('; '.join)
    # df['Protein names'] = (df['Protein names'].swifter.progress_bar(enable=True, desc='Removing duplicate protein names').apply(set))
    # df['Protein names'] = (df['Protein names'].swifter.progress_bar(enable=True, desc='Joining protein names list').apply('; '.join))

    df['Organism'] = df['Organism'].str.split('; ').apply(set).apply('; '.join)
    # df['Organism'] = (df['Organism'].swifter.progress_bar(enable=True, desc='Removing duplicate organism names').apply(set))
    # df['Organism'] = (df['Organism'].swifter.progress_bar(enable=True, desc='Joining organism names list').apply('; '.join))

    # Apply function to get species ID from organism ID
    df['Species taxonomic ID'] = (df['Taxonomic lineage IDs']
                                .swifter #.progress_bar(enable=True,
                                #   desc='Getting Viruses taxonomic IDs')
                                .apply(getRankID, rank='species'))

    dff = df[['Entry', 'Species taxonomic ID']].copy()

    # Get the species name of the earlier unidentified taxonomic IDs
    idx_species_name = df.columns.get_loc('Taxonomic lineage (SPECIES)')
    idx_organism_id = df.columns.get_loc('Species taxonomic ID')

    for row in range(len(df)):
        if np.isnan(df.iat[row, idx_organism_id]):
            df.iat[row, idx_organism_id] = getIDfromName(df.iat[row, idx_species_name])

    df['Species taxonomic ID'] = df['Species taxonomic ID'].apply(int)

    df = (df.drop(['Status','Taxonomic lineage IDs'], axis=1)
        .groupby('Species taxonomic ID', as_index=False)
        .agg({'Virus hosts':set, 'Organism':set,
                'Protein names':set, 'Taxonomic lineage (SPECIES)':'first'}))

    df['Virus hosts'] = df['Virus hosts'].str.join('; ')
    df['Organism'] = df['Organism'].str.join('; ')
    df['Protein names'] = df['Protein names'].str.join('; ')

    df['Species name'] = (df.drop('Taxonomic lineage (SPECIES)', axis=1)
                        .swifter#.progress_bar(enable=True, desc='Getting Species name')
                        .apply(lambda x: getRankName(x['Species taxonomic ID'],
                                                    rank='species'), axis=1))

    df['Species superkingdom'] = (df['Species taxonomic ID']
                                .apply(getRankName, rank='superkingdom'))

    df['Species family'] = (df['Species taxonomic ID']
                            .apply(getRankName, rank='family'))

    df = df[df['Species superkingdom'] == 'Viruses']

    df.drop(['Taxonomic lineage (SPECIES)'], axis=1, inplace=True)

    df['Virus hosts'] = (np.where(df['Virus hosts']=='',
                        np.nan,
                        df['Virus hosts']))

    df.drop('Organism', axis=1, inplace=True)

    # List of viruses which do not have assigned hosts in the data
    noHostViruses = df[df['Virus hosts'].isnull()]['Species name'].unique().tolist()

    # Create independent dataframe of viruses with no assigned host and simltaneously identify the same viruses from the data
    # whcih already have assigned hosts and assign host names based on those.
    df_na_hosts = (df[(~df['Virus hosts'].isnull()) &\
        (df['Species name']
        .isin(noHostViruses))][['Species name', 'Virus hosts']])

    df_na_hosts = (df_na_hosts
                .groupby('Species name')['Virus hosts']
                .apply(list))

    df_na_hosts = (df_na_hosts
                .reset_index(name='Viral hosts nw'))

    # # Previous code reurns a list for multiple host so this code melts the lists into regular string entries
    df_na_hosts['Viral hosts nw'] = (df_na_hosts['Viral hosts nw']
                                    #  .swifter.progress_bar(desc='Joining host names list', enable=True)
                                    .apply('; '.join))

    # # updates the viruses hosts
    df_naa = (df[df['Virus hosts'].isnull()]
            .merge(df_na_hosts, on='Species name', how='left')
            .drop('Virus hosts', axis=1)
            .rename({'Viral hosts nw':'Virus hosts'}, axis=1))

    # Creates independant dataset with viruses which have hosts
    df_notna = df[~df['Virus hosts'].isnull()]

    # merges the updated virus hosts dataset with the dataset with viruses which have hosts
    df = df_naa.append(df_notna)

    # merges the updated virus hosts dataset with the dataset with viruses which have hosts
    df = df_naa.append(df_notna)

    df['Virus hosts'] = np.where(df['Virus hosts'].isnull(), '',df['Virus hosts'])

    df = mergeRows(df, 'Species taxonomic ID','Virus hosts')

    dfna = df[df['Virus hosts'] == '']
    df = df[~(df['Virus hosts'] == '')]

    # Updating host names from external sources
    df2 = pd.read_csv(args.ncbivirusdb) #('../data/sequences.csv')
    df2.drop_duplicates(inplace=True)

    df2['Species ID'], df2['Host ID'] = df2['Species'].apply(getIDfromName), df2['Host'].apply(getIDfromName)

    df2.dropna(inplace=True)
    df2['Species ID'], df2['Host ID'] = df2['Species ID'].astype(int), df2['Host ID'].astype(int)

    df2['Host name'] = df2.apply(lambda x: nameMerger(
        x['Host'], x['Host ID']),
        axis=1)
    # Remove Host and Host ID columns as they have been merged and are no longer needed
    df2.drop(['Host', 'Host ID'],
        axis=1, inplace=True)

    df2['Species ID'] = df2['Species ID'].apply(getRankID, rank='species')

    dfff = df2.copy()

    df_na_hosts = AggregateHosts(df2,'Species ID', 'Host name')
    dfna = dfna.merge(df_na_hosts, left_on='Species taxonomic ID', right_on='Species ID', how='left')
    dfna = dfna.drop(['Virus hosts', 'Species ID'], axis=1).rename({'Host name':'Virus hosts'}, axis=1)
    dfna = UpdateHosts(dfna, df_na_hosts, 'Species taxonomic ID', 'Species ID')
    df, dfna = UpdateMain(df, dfna)
    df = mergeRows(df, 'Species taxonomic ID', 'Virus hosts')

    df2 = pd.read_table(args.virusdb)#('../data/virushostdb.tsv')

    df2 = df2[['virus tax id', 'virus name', 'host tax id', 'host name']].copy()
    df2.drop_duplicates(inplace=True)

    df2.dropna(inplace=True)

    df2['host tax id'] = df2['host tax id'].astype(int)

    df2['Species ID'] = df2['virus tax id'].apply(getRankID, rank='species')

    df2['Host name'] = df2.apply(lambda x: nameMerger(x['host name'], x['host tax id']), axis=1)
    # Remove Host and Host ID columns as they have been merged and are no longer needed
    df2.drop(['host name', 'host tax id'], axis=1, inplace=True)

    df_na_hosts = AggregateHosts(df2,'Species ID', 'Host name')
    dfna = dfna.merge(df_na_hosts, left_on='Species taxonomic ID', right_on='Species ID', how='left')
    dfna = dfna.drop(['Virus hosts', 'Species ID'], axis=1).rename({'Host name':'Virus hosts'}, axis=1)
    dfna = UpdateHosts(dfna, df_na_hosts, 'Species taxonomic ID', 'Species ID')
    df, dfna = UpdateMain(df, dfna)
    df = mergeRows(df, 'Species taxonomic ID', 'Virus hosts')

    df2 = pd.read_csv(args.liverpooluni) #('../data/virus_host_4rm_untitled.csv')

    df2 = df2[['Host_name', 'Host_TaxId', 'Virus_name', 'Virus_TaxId']].copy()
    df2['Species ID'] = df2['Virus_TaxId'].apply(getRankID, rank='species')
    df2['Host name'] = df2.apply(lambda x: nameMerger(x['Host_name'], x['Host_TaxId']), axis=1)
    df2.drop(['Host_name', 'Host_TaxId'], axis=1, inplace=True)
    df2.dropna(inplace=True)

    df_na_hosts = AggregateHosts(df2,'Species ID', 'Host name')
    dfna = dfna.merge(df_na_hosts, left_on='Species taxonomic ID', right_on='Species ID', how='left')
    dfna = dfna.drop(['Virus hosts', 'Species ID'], axis=1).rename({'Host name':'Virus hosts'}, axis=1)
    dfna = UpdateHosts(dfna, df_na_hosts, 'Species taxonomic ID', 'Species ID')
    df, dfna = UpdateMain(df, dfna)
    df = mergeRows(df, 'Species taxonomic ID', 'Virus hosts')

    df['Infects human'] = np.where(df['Virus hosts'].str.contains(r'960[56]'), 'human-true','human-false')

    df['Virus hosts'] = df['Virus hosts'].str.split('; ')
    df['Virus hosts'] = df.apply(lambda x: list(filter(None, x['Virus hosts'])), axis=1)
    df['Virus hosts'] = df['Virus hosts'].apply('; '.join)

    df = (df.set_index(df.columns.drop('Virus hosts',1).tolist())['Virus hosts'].str.split(';', expand=True)
            .stack()
            .reset_index()
            .rename(columns={0:'Virus hosts'})
            .loc[:, df.columns]
            ).copy()

    df['Virus hosts ID'] = None
    idx_organism = df.columns.get_loc('Virus hosts')
    idx_host_id = df.columns.get_loc('Virus hosts ID')

    pattern = r'(\d+)\]'
    for row in range(len(df)):
        host_id = re.search(pattern, df.iat[row, idx_organism]).group()
        df.iat[row, idx_host_id] = host_id

    df['Virus hosts ID'] = df['Virus hosts ID'].str.strip('\]')

    df['Virus hosts ID'] = df['Virus hosts ID'].apply(int)

    df['Virus hosts ID'] = df['Virus hosts ID'].apply(getRankID, rank='species')
    df['Virus host name'] = df['Virus hosts ID'].apply(getRankName, rank='species')
    df['Host superkingdom'] = df['Virus hosts ID'].apply(getRankName, rank='superkingdom')
    df['Host kingdom'] = df['Virus hosts ID'].apply(getRankName, rank='kingdom')

    df['Virus hosts ID'] = df['Virus hosts ID'].apply(int)

    df['Virus hosts'] = (df.drop('Virus hosts', axis=1)
                        .apply(lambda x: nameMerger(
                            x['Virus host name'], x['Virus hosts ID']),
                            axis=1))

    df['Virus hosts'] = (df.drop('Virus hosts', axis=1)
                        .apply(lambda x: nameMerger(
                            x['Virus host name'], x['Virus hosts ID']),
                            axis=1))

    df = (df.set_index(df.columns.drop('Protein names',1).tolist())['Protein names'].str.split(';', expand=True)
            .stack()
            .reset_index()
            .rename(columns={0:'Protein names'})
            .loc[:, df.columns]
            ).copy()

    # Restructuring the data

    fastaFileName = args.fasta #'../data/uniprot-keyword Virus+entry+into+host+cell+[KW-1160] +fragment no.fasta'
    entry_seq = read_fasta(fastaFileName)

    dff.sort_values(by='Entry', inplace=True)

    objList = []
    for entry, obj in entry_seq:
        objList.append(obj)

    dff['Sequence'] = objList

    df.drop(['Virus host name', 'Protein names', 'Species superkingdom'], axis=1, inplace=True)
    df = df.merge(dff, on='Species taxonomic ID', how='left')

    # free memory
    del dff, df2

    df.drop_duplicates(inplace=True)

    df['Virus hosts ID'] = df['Virus hosts ID'].apply(str)

    df = (df.groupby('Entry', as_index=False)
        .agg({'Virus hosts':set, #'Protein':'first',
                'Infects human':'first', 'Species name':'first',
                'Host superkingdom':set,
                'Host kingdom':set,
                'Virus hosts ID':set,
                'Species family':'first',
                'Species taxonomic ID':'first',
                'Sequence': 'first'}))

    df['Virus hosts'] = (df['Virus hosts']
                        #  .swifter.progress_bar(enable=True,
                        #                        desc='Joining host names list')
                        .apply('; '.join))
    df['Virus hosts ID'] = (df['Virus hosts ID']
                            # .swifter.progress_bar(enable=True,
                            #                       desc='Joining host IDs')
                            .apply('; '.join))
    df['Host kingdom'] = (df['Host kingdom']
                        #   .swifter.progress_bar(enable=True,
                        #                         desc='Joining host kingdom names')
                        .apply('; '.join))
    df['Host superkingdom'] = (df['Host superkingdom']
                            #    .swifter.progress_bar(enable=True,
                            #                          desc='Joining host superkingdom names')
                            .apply('; '.join))
    df['Sequence'] = df.apply(lambda x: getSequenceFeatures(
        seqObj=x['Sequence'], entry=x['Entry'],
        organism=x['Species name'], status=x['Infects human']), axis=1)

    df['Protein'] = df['Sequence'].apply(lambda x: x.protein_name)
    # Append sequences to dataframe
    # df2 = pd.read_csv('../data/sequences.csv')
    dfff.rename({
        'Species ID': 'Species taxonomic ID',
        'Molecule_type': 'Molecule type'},
        axis=1,
        inplace=True)

    df['Species taxonomic ID'] = df['Species taxonomic ID'].apply(int)

    df = df.merge(
        dfff[['Species taxonomic ID', 'Molecule type']],
        how='left',
        on='Species taxonomic ID')

    # free memory
    del dfff

    df.drop_duplicates(inplace=True)

    df = df[['Entry', 'Protein', 'Species name',
            'Species taxonomic ID', 'Species family', 'Virus hosts',
            'Virus hosts ID', 'Host kingdom',
            'Host superkingdom', 'Molecule type', 'Infects human', 'Sequence']]


    # Split Dataframe to multiple datasets for testing

    df['Molecule type'] = np.where(df['Molecule type'].isna(), '', df['Molecule type'])

    unfiltered = df

    metazoa = df[df['Host kingdom'].str.contains('Metazoa')].copy()

    plant_human = (df[(df['Host kingdom']
                    .str.contains('Viridiplantae')) |\
                        df['Virus hosts']
                        .str.contains('[Hh]omo [Ss]apiens')]).copy()

    NonEukaryote_Human = (df[(df['Host superkingdom']
                        .isin(['Bacteria', 'Viruses', 'Archaea'])) |\
                            (df['Virus hosts']
                            .str.contains('[Hh]omo [Ss]apiens'))]).copy()

    DNA_MetazoaZoonosis = metazoa[metazoa['Molecule type'].str.contains('DNA')].copy()

    RNA_MetazoaZoonosis = metazoa[metazoa['Molecule type'].str.contains('RNA')].copy()

    metazoaFile = 'MetazoaZoonosis'
    plant_humanFile = 'Plant-HumanZoonosis'
    unfilteredFile = 'Zoonosis'
    NonEukaryote_HumanFile = 'NonEukaryote-Human'
    DNA_metazoaFile = 'DNA-MetazoaZoonosis'
    RNA_metazoaFile = 'RNA-MetazoaZoonosis'

    # Write file sequences to fasta for feature extraction

    dirs = [
        'MetazoaZoonosisData','ZoonosisData',
        'Plant-HumanZoonosisData', 'NonEukaryote-HumanData',
        'DNA-MetazoaZoonosisData', 'RNA-MetazoaZoonosisData']

    for dir in dirs:
        if not os.path.isdir(dir):
            os.makedirs(dir)

    files = [
        metazoaFile, unfilteredFile,
        plant_humanFile, NonEukaryote_HumanFile,
        DNA_metazoaFile, RNA_metazoaFile]

    dataframes = [
        metazoa, unfiltered,
        plant_human, NonEukaryote_Human,
        DNA_MetazoaZoonosis, RNA_MetazoaZoonosis]

    # Resample data

    # Undersample majority class such that minority class (human-false) is 60% of the majority class (human-true317316)
    seed = 960505
    rus = RandomUnderSampler(sampling_strategy=1., random_state=seed)
    sampled_dataframes = []
    for dt in dataframes:
        clas = dt['Infects human']
        dt, _ = rus.fit_resample(dt, clas)
        sampled_dataframes.append(dt)

    # Save data and sequences
    toSave = list(zip(sampled_dataframes, files, dirs))

    for dff, file, folder in toSave:
       # save dataframes as csv
        dff.drop('Sequence', axis=1).to_csv(f'{folder}/{file}Data.csv.gz', index=False, compression='gzip')

       # Create subdirectories
        os.makedirs(os.path.join(folder, 'train'), exist_ok=True)
        os.makedirs(os.path.join(folder, 'test'), exist_ok=True)
        # os.makedirs(os.path.join(folder, 'train/human-false'), exist_ok=True)
        # os.makedirs(os.path.join(folder, 'test/human-false'), exist_ok=True)
       # Split data to train and test data
        train, test = train_test_split(dff, test_size=0.2, random_state=seed) # Will further split 15% of train as validation during training
       # Save test and train sequences
        save_sequences(train, f'{folder}/train/Sequences') # Will move to subdirectories after feature extraction
        save_sequences(test, f'{folder}/test/Sequences')

        print('Done with', file)
