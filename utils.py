import re

def removeNonAlphabet(word):
    return (re.sub(r'[^a-zA-Z]', '', word)).lower()

def characterVectorisation(word, maxlen):
    if len(word) < maxlen:
        word = word+' '*(maxlen-len(word))
    else:
        word = word[:maxlen]
    charList = list(word)
    return [max(0, ord(char)-96) for char in charList]
            
def preprocess(df, maxlen=15, train=True):
    
    # remove non alphabet and change to lowercase
    df['name'] = [removeNonAlphabet(name) for name in df['name']]
    
    # character vectorisation
    df['name'] = [characterVectorisation(name, maxlen) for name in df['name']]
    
    # encoding gender to numbers
    if train:
        df['gender'] = [0.0 if gender=='F' else 1.0 for gender in df['gender']]
    return df
