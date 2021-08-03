#!/usr/bin/env nextflow
nextflow.enable.dsl=2

process GetAAComposition {
    input:
    path fasta
    output:
    path "*.csv"
    script:
    """
    ilearnprotein --file $fasta --out $out --method AAC --format csv --userDefinedOrder alphabetically
    """
}


process TabFormatAAComposition {
    input:
    path datafile
    path AACfile
    output:
    path "*.csv"
    script:
    """
    aminofmt --csv $datafile --file $AACfile
    """
}

process GroupProteinsByName {
    input:
    path tabfile
    path fastafile
    output:
    path "*"
    script:
    """
    source activate tensorEnv
    python ${baseDir}/groupProteins.py --tab $tabfile --fasta $fastafile
    """
}

process FASTAGetProteinName {
    
    // label "with_cpus"
    
    input:
    path datafile
    path fastafile
    output:
    path "*.tab.gz"
    script:
    """
    source activate tensorEnv
    python ${baseDir}/protein_name.py --file $datafile --fasta $fastafile
    """
}

process ProteinAlignmentMuscle {
    if (params.save) {
        publishDir "${params.baseDir}/data/Protein-Alignments"
    }
    
    label "with_cpus"
    
    input:
        each fasta
    
    output:
        path "*.fasta"
    
    script:
        """
        muscle -in $fasta -out ${fasta} -maxiters 2 -diags1 -sv -distance1 kbit20_3
        """
}