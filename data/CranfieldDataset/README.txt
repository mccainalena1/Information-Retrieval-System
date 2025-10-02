Cranfield collection.
1398 abstracts (numbered 1 through 1400).
Aerodynamics.

Smallish collection, with large number of queries (225)

This directory contains:
cran.all.Z
        Compressed version of document text. 
qrels.text.Z
        Relation giving relevance judgements.  Columns of file are
                query_id  doc_id   0    0
        to indicate doc_id is relevant to query_id.
query.text.Z
        Text of queries.  
tf_doc.Z
        Indexed documents.  Columns of file are
             doc_id  0  concept_number  tf_weight  stemmed_word
        to indicate stemmed_word occurs in doc_id tf_weight times,
        and has been assigned the designator concept_number.
tf_query.Z
        Indexed queries.   Columns of file are
             query_id  0  concept_number  tf_weight  stemmed_word
        to indicate stemmed_word occurs in query_id tf_weight times,
        and has been assigned the designator concept_number.

