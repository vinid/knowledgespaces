# Knowledge Graph Embeddings from Text

This package contains code to generate KG embeddings starting from text

## How To
The mikolov_example.ipynb contains a short tutorial on how to run 
and use the software to test and replicate results

## Features implemented

+ Text Annotation using DBpedia Spotlight
+ Word2vec gensim wrapper for entities and types
+ Word Entity linking function for analogies
+ Analogical Reasoning evaluators


## Classes Explanation

###Models

#### Term embedding
Wrapper class around gensim word2vec models. Integrate also
some metadata.

#### TeeEmbeddings
Allows to concatenate entity and type embeddings

###Bridges

####Bridge 
Represents the word-anchor that allow 
to bridge words to entities. They also provide a method
to disambiguate analogies by considering the analogical input.

####CleanBridge
Fake class that returns the input, can be used with word base embedding

###Evaluetors

####AnalogyEvaluetor
Offers the possibility to evaluate analogies by taking in input a
file that contains analogy.

####EvaluatorHandler
Just an handler class that allows to handle the evaluation
and to save results in a file.




