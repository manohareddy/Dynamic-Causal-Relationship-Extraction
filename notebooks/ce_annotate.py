import warnings
warnings.filterwarnings("ignore")
import spacy
import scispacy
import nltk
import argparse
import time
import pickle
from nltk.corpus import stopwords
from collections import OrderedDict
import re
import numpy as np
import pandas as pd
from itertools import permutations 
nlp = spacy.load('en_ner_bc5cdr_md')
print("spacy model loaded")
from transformers import pipeline
nlp_qa = pipeline('question-answering',device=0)
print("HF model loaded")
try:
  from textblob import TextBlob
  STOPWORDS=list(stopwords.words('english'))
except:
  print("nltk not completely donloaded")
  nltk.download("popular")
  nltk.download('brown')
  from textblob import TextBlob
  STOPWORDS=list(stopwords.words('english'))
print("All libraries loaded")


def clean_text(sent):
  sent = REPLACE_BY_SPACE_RE.sub(' ',sent)
  sent = BAD_SYMBOLS_RE.sub(' ',sent)
  sent = ' '.join([word for word in sent.split() if word not in STOPWORDS])
  return(sent)

def ie_process(sent):
  dep_list = nltk.pos_tag(nltk.word_tokenize(sent))
  dep_dict = OrderedDict()
  for tok in dep_list:
    if(tok[0] in list(dep_dict.keys())):
      dep_dict[tok[0]] = "|".join([dep_dict[tok[0]],tok[1]])
    else:
      dep_dict[tok[0]] = tok[1]
  return(dep_dict)

def get_dps(sent,phrase):
  dep_dict = ie_process(sent)
  return("|".join([dep_dict[word] for word in nltk.word_tokenize(str(phrase)) if word in list(dep_dict.keys())]))

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|,;]^a-zA-Z#')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

def get_nps(sent):
  sent = REPLACE_BY_SPACE_RE.sub(' ',sent)
  sent = BAD_SYMBOLS_RE.sub(' ',sent)
  nps_tb = TextBlob(sent).noun_phrases   #can be substituted with an alternative way of extracting noun phrases
  nps_spacy = [token.text for token in nlp(sent).noun_chunks]
  nps = nps_tb+nps_spacy
  nps = [clean_text(NP) for NP in nps]
  nps = [str(NP) for NP in nps if 'NN' in get_dps(sent,NP).split('|')]
  nps = list(np.unique(nps)) 
  return(nps)

#checks if given sentence contains any causal words
def get_question_template(sent,causal_words,cause_dict):
  questions = []
  for causal_word in causal_words:
    if(causal_word in sent):
      questions += cause_dict[causal_word]
  if(len(questions)>0):
    return("|".join(questions))
  else:
    return(np.nan)


#this function takes in a sentence, and returns a dictionary (if possible to extract)
ce_count = 0
sent_count = 0
t_start = 0
def get_ce_dict(x):
  sent = x.raw_sentence
  sent = REPLACE_BY_SPACE_RE.sub(' ',sent)
  sent = BAD_SYMBOLS_RE.sub(' ',sent)
  sent = ' '.join(sent.split())
  question_set = x.question_set.split('|')
  global ce_count,sent_count,t_start 
  nps = get_nps(sent)
  #print(" Raw sentence :{}".format(sent))
  #print(" Noun phrases : {}".format(nps))
  curr_score = 0.8  #can be altered
  curr_dict = np.nan
  qt_1 = [question for question in question_set if question.count('{}')==1]
  qt_2 = [question for question in question_set if question.count('{}')==2]
  if(len(nps)>1):
    perms = list(permutations(nps,2))
    questions = []
    causes = []
    effects = []
    for question in question_set:
      if(question.count('{}')==2):
        questions += [question.format(a,b) for (a,b) in perms]
        causes += [a for (a,b) in perms]
        effects += [b for (a,b) in perms]
      elif(question.count('{}')==1):
        questions += [question.format(a) for a in nps]
        causes += [None for i in range(0,len(nps))]
        effects += nps
      else:
        print(question)
    assert(len(causes)==len(effects))
    qa_input = [{'context':sent,'question':question} for question in questions]
    results = nlp_qa(qa_input)
    for i,result in enumerate(results):
      result = dict(result)
      if(result['score']>=curr_score):
        # print("Pair found for sentence : {}".format(sent))
        # print(" question:'{}',result:'{}'".format(questions[i],result))
        curr_dict  = result
        if(causes[i]!=None):
          curr_dict['cause'] = causes[i]
        else:
          curr_dict['cause'] = result['answer']
        curr_dict['effect'] = effects[i]
        curr_dict['question'] = questions[i]
        curr_dict['noun_phrases'] = nps
        curr_dict['clean_sentence'] = sent
        curr_score = result['score']
  if(curr_score>0.8):
    sent_count += 1
  ce_count+=1
  if(ce_count%50==0):
    tt = round((time.time() - t_start)/(ce_count),3)
    print('sentences read == {}, sentences annotated = {}, time per annotation = {}'.format(ce_count,sent_count,tt))
    status_dict = {"sentences_read":ce_count,"sentences_annotated":sent_count,"time_per_annotation":tt}
    pickle.dump(status_dict,open("status_dict.pkl","wb"))
  return(curr_dict)


def annotate(args):

  global t_start
  cause_df = pd.read_csv(args.causal_words_file).dropna()
  cause_dict = dict()
  for i,cause_word in enumerate(cause_df.cause_words.values):
    cause_dict[cause_word] = cause_df.questions_1.values[i].split('|')+cause_df.questions_2.values[i].split('|')
  causal_words = list(cause_dict.keys())

  
 
  cr_df = pd.read_csv(args.input_file).iloc[args.start_idx:args.end_idx,:]
  print("Starting annotation for indices range {} - {}".format(args.start_idx,args.end_idx))
  t_start = time.time()
  cr_df["HF_result"] = cr_df[["raw_sentence","question_set"]].apply((lambda x:get_ce_dict(x)),axis=1)
  # tt = round((time.time() - t_start)/(cr_df.dropna().shape[0]),3)
  # print("Time taken per annotation:{}".format(tt))

  cr_df = cr_df.dropna()
  cr_df["clean_sentence"] = cr_df["HF_result"].apply(lambda x:x["clean_sentence"])
  cr_df["cause"] = cr_df["HF_result"].apply(lambda x:x["cause"])
  cr_df["effect"] = cr_df["HF_result"].apply(lambda x:x["effect"])
  cr_df["question"] = cr_df["HF_result"].apply(lambda x:x["question"])
  cr_df["score"] = cr_df["HF_result"].apply(lambda x:round(x["score"],3))
  cr_df["noun_phrases"] = cr_df["HF_result"].apply(lambda x:x["noun_phrases"])
  cr_df["answer"] = cr_df["HF_result"].apply(lambda x:x["answer"])
  output_filename = args.output_file + "{}-{}.csv".format(args.start_idx,args.end_idx)
  print("Saving out to {}".format(output_filename))
  cr_df[["clean_sentence","cause","effect","question","answer","score","noun_phrases"]].to_csv(output_filename,index=False)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
    
  parser.add_argument("--start_idx", default=0, type=int)  # number of sentences to read
  parser.add_argument("--end_idx", default=5000, type=int)  # number of sentences to read
  parser.add_argument("--print_freq", default=50, type=int)  # how frequently to output update
  parser.add_argument("--input_file", default='annotation_input_file.csv', type=str)  # number of sentences to read
  parser.add_argument("--causal_words_file", default='cause_df.csv', type=str)  # number of sentences to read
  parser.add_argument("--output_file", default='ce_annot', type=str) #output csv file

  annotate(parser.parse_args())
    