from TextRobustness.component.sample.dp_sample import DPSample


word = ['Influential', 'members', 'of', 'the', 'House', 'Ways', 'and',
        'Means', 'Committee', 'introduced', 'legislation', 'that',
        'would', 'restrict', 'how', 'the', 'new', 'savings-and-loan',
        'bailout', 'agency', 'can', 'raise', 'capital', ',', 'creating',
        'another', 'potential', 'obstacle', 'to', 'the', 'government',
        "'s", 'sale', 'of', 'sick', 'thrifts', '.']
postag = ['JJ', 'NNS', 'IN', 'DT', 'NNP', 'NNP', 'CC', 'NNP', 'NNP',
          'VBD', 'NN', 'WDT', 'MD', 'VB', 'WRB', 'DT', 'JJ', 'JJ', 'NN',
          'NN', 'MD', 'VB', 'NN', ',', 'VBG', 'DT', 'JJ', 'NN', 'TO',
          'DT', 'NN', 'POS', 'NN', 'IN', 'JJ', 'NNS', '.']
head = ['2', '10', '2', '6', '6', '3', '6', '9', '6', '0', '10', '14',
        '14', '11', '22', '20', '20', '20', '20', '22', '22', '14', '22',
        '14', '14', '28', '28', '25', '28', '31', '33', '31', '29', '33',
        '36', '34', '10']
deprel = ['amod', 'nsubj', 'prep', 'det', 'nn', 'pobj', 'cc', 'nn',
          'conj', 'root', 'dobj', 'nsubj', 'aux', 'rcmod', 'advmod',
          'det', 'amod', 'amod', 'nn', 'nsubj', 'aux', 'ccomp', 'dobj',
          'punct', 'xcomp', 'det', 'amod', 'dobj', 'prep', 'det', 'poss',
          'possessive', 'pobj', 'prep', 'amod', 'pobj', 'punct']

word_1 = ['Shares', 'also', 'closed', 'sharply', 'lower', 'across', 'Europe',
        ',', 'particularly', 'in', 'Frankfurt', ',', 'although', 'London',
        'and', 'a', 'few', 'other', 'markets', 'recovered', 'some', 'ground',
        'after', 'stocks', 'began', 'to', 'rebound', 'in', 'New', 'York', '.']
postag_1 = ['NNP', 'RB', 'VBD', 'RB', 'JJR', 'IN', 'NNP', ',', 'RB', 'IN',
          'NNP', ',', 'IN', 'NNP', 'CC', 'DT', 'JJ', 'JJ', 'NNS', 'VBD', 'DT',
          'NN', 'IN', 'NNS', 'VBD', 'TO', 'VB', 'IN', 'NNP', 'NNP', '.']
head_1 = ['3', '3', '0', '5', '3', '3', '6', '3', '10', '3', '10', '3', '20',
        '20', '14', '19', '19', '19', '14', '3', '22', '20', '25', '25', '20',
        '27', '25', '27', '30', '28', '3']
deprel_1 = ['nsubj', 'advmod', 'root', 'advmod', 'advmod', 'prep', 'pobj',
          'punct', 'advmod', 'prep', 'pobj', 'punct', 'mark', 'nsubj', 'cc',
          'det', 'amod', 'amod', 'conj', 'advcl', 'det', 'dobj', 'mark',
          'nsubj', 'advcl', 'aux', 'xcomp', 'prep', 'nn', 'pobj', 'punct']

data = {'word': word, 'postag': postag, 'head': head, 'deprel': deprel}
sample = DPSample(data)

data_1 = {'word': word_1, 'postag': postag_1,
          'head': head_1, 'deprel': deprel_1}
sample_1 = DPSample(data_1)
