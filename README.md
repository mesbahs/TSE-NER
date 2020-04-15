# TSE-NER

We contribute TSE-NER, an iterative low-cost approach for training NER/NET classifiers for long-tail entity types by exploiting
Term and Sentence Expansion. This approach relies on minimal human input  a seed set of instances of the targeted entity type.
We introduce different strategies for training data extraction, semantic expansion, and result entity filtering.

#Folders
-The "crf_trained_files" contains the trained models
-The "data" folder contains the manually annotated test sets.
-The "evaluation_files" folder contains the files containing the seed terms for the 'dataset' entity type
-The "evaluation_filesMet" folder contains the files containing the seed terms for the 'method' entity type
-The "models" folder contains the word2vec and doc2vec models trained on the corpus
-The "postprocessing" folder contains the scripts for training data generation and filtering
-The "preprocessing" folder contains the scripts for term and sentence expansion strategies and the NER training
-The "prop_files" folder contains the property files used for NER training
-The "stanford_files" folder contains the stanford-ner.jar


#Training a new model
-Change the ROOTPATH in the default_config.py.
-Put your seed terms in the "/data/dataset-names-train.txt"
-Index all the sentences of the publication's text in the corpus using the elasticsearch to extract sentences quickly
-run main.py

#Example
For training an NER model for dataset entity type we need to generate the training data as follows:

we	O
performed	O
a	O
systematic	O
set	O
of	O
experiments	O
using	O
the	O
LETOR	DATA
benchmark	O
collections	O
OHSUMED	DATA
,	O
TD2004	DATA
,	O
and	O
TD2003	DATA

Assume we had a seed term "Letor". Using that term, the "training_data_extraction.extract" function extract all
the sentences that contains that word and that are not in the testing set. However as seen in the above example
also OHSUMED, TD2004 and TD2003 are names of the dataset entity type, but they are not in our seed terms.
In this context, in order to label the other entities also corectly and avoid  the false negatives
we perform the Term Expansion  "expansion.term_expansion_dataset" where we extract all the entities from the text
and we cluster all terms extracted from the sentences with respect to their embedding vectors using K-means, Silhouette
analysis is used to find the optimal number k of clusters. Finally, clusters that contain at least one of the seed
terms are considered to (only) contain entities the same type (e.g Dataset). This means if OHSUMED, TD2004 and TD2003
also appear in the same cluster as Letor they will be label as DATA otherwise O. The entities of the Term expansion step
will be stored in the /evaluation_files folder. Next, we use the seeds and the expanded entities to annotate the sentences
that we extracted using the seeds in the "trainingdata_generation.generate_trainingTE" which should generate the training
data as above. Then we use the training data to traing a new NER. We first generate the property files for the Stanford
NER using the ner_training.create_austenprop. Next we use "ner_training.train" to train the new model. The trained model can
be tested on the test set using "ner_training.test".

For the next iterations, we used the trained-models in the crf_trained_files and use "extract_new_entities.ne_extraction" to
extract all the entities from the corpus. However the extracted entities might be noisy. So we use the filtering techniques to filter
out irrelevant entities using the "filtering". Next the new filtered entities can be used as new seeds for the next iteration...



For the Sentence Expansion approach, everything is the same except, for each sentence (trainingdata_generation.generate_trainingSE), we find its most similar sentence in the corpus
using doc2vec. If the new sentence contains a word of the Term Expansion we will label it accordingly, if not we use it as negative example.



## TSE-NER  cleaned source code: 
https://github.com/mvallet91/SmartPub-TSENER (check Pipeline_Preparation and Pipeline_TSENER Notebooks for a clear example)

#Seed Terms used: https://docs.google.com/spreadsheets/d/1h2PXyG9hKnMIcaorU5_YzOn8T90nP6COtSnSK3bz6rU/edit?usp=sharing



## Baselines used for evaluation:

Concept Extractor: https://github.com/cttsai/concept-extractor
Hearst Patterns: https://github.com/mmichelsonIF/hearst_patterns_python/tree/master/hearstPatterns


## Titles of papers used
https://drive.google.com/file/d/1XHa5zqwYfuZR21J2XGhmzK_uM1ZGHQa7/view?usp=sharing 

Additional entities evaluated: https://docs.google.com/spreadsheets/d/1kW9AjSYXdgCVnRSgANTAX3iqdkM3OAPjpr2oWYyAN4Y/edit?usp=sharing 

## Sources used for Seed term selection:

Datasets (1): https://github.com/caesar0301/awesome-public-datasets 
Datasets (2): https://www.analyticsvidhya.com/blog/2016/11/25-websites-to-find-datasets-for-data-science-projects/ 
Methods: https://en.wikipedia.org/w/api.php?action=query&format=json&list=categorymembers&cmnamespace=14&cmlimit=500&cmtitle=Category:Algorithms 
Proteins: http://obofoundry.org/ontology/pr.html


## Context Words used for PMI filtering

Dataset_contextwords = [dataset, corpus, collection, repository, benchmark, website] 
Method_contextwords = [method, model, algorithm, approach]
Protein_contextwords  = [protein, receptor]
Corpus Information
20,519 Scientific publications, including computer science (for dataset and method entities) and biomedical (for protein entities) domains.

## Computer Science Conferences:

The evaluation is performed in the domain of scientific publications with a focus on data science and processing. In our corpus, we have 15,994 papers from eight conference series: the International World Wide Web Conference (WWW - 2106 papers from 2001 to 2016); the International Conference on Software Engineering (ICSE - 2983 papers from 1976 to 2016); the International Conference on Very Large Databases (VLDB - 1883 papers from 1975 to 2007); the Joint conference on Digital Libraries (JCDL - 1543 papers from 2001 to 2016); the Text Retrieval Conference (TREC - 1415 papers from 1999 to 2015); the International Conference on Research and Development in Information Retrieval (SIGIR - 3561 papers from 1971 to 2016); the International Conference On Web and Social Media (ICWSM - 815 papers from 2007 to 2016); the European Conference on Research and Advanced Technology on Digital Libraries (ECDL - 797 papers from 1997 to 2010); the Extended Semantic Web Conference (ESWC – 626 papers from 2005) and the International Conference on Theory and Practice of Digital Libraries (TPDL - 276 papers, from 2011 to 2016).



## Biomedical Journals:

For publications on the biomedical domain, the journals are the same as the ones selected for the CRAFT corpus (available in the Open Access subset of PubMed Central). In this work we use 4,525 papers from 10 journals: Genome Biology and Evolution (GBE - 130 papers from 2012 to 2018); Breast Cancer Research (BCR - 416 papers from 2001 to 2018); BMC Neuroscience (BMC Neurosci - 476 papers from 2003 to 2018); Genome Biology (Genome Biol -716 papers from 2003 to 2018); Breast Cancer Research and Treatment (Breast Cancer Res. Treat - 23 papers from 2016 to 2018); BMC Evolutionary Biology (BMC Evol Biol - 469 papers from 2004 to 2018); BMC Genomics (BMC Genomics - 53 papers from 2002 to 2018); PLoS Biology (PLoS Biol - 875 papers from 2003 to 2018); BMC Biotechnology (BMC Biotechnol - 423 papers from 2002 to 2018); PLoS Genetics (PLoS Genet - 944 papers from 2005 to 2018).
