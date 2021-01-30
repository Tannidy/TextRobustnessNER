"""
TextProcessor Class
============================================

"""
import threading
# from ltp import LTP


class CWSTextProcessor:
    """
        Text Processor class implement NER.
    """
    _instance_lock = threading.Lock()

    def __init__(self):
        self.__tokenize = None
        self.__ner = None
        self.__seg = None
        self.__ner_list = None
        self.__pos_tag = None

    # Single instance mode
    def __new__(cls, *args, **kwargs):
        if not hasattr(CWSTextProcessor, "_instance"):
            with CWSTextProcessor._instance_lock:
                if not hasattr(CWSTextProcessor, "_instance"):
                    CWSTextProcessor._instance = object.__new__(cls)
        return CWSTextProcessor._instance

    @staticmethod
    def word_tokenize(sent):
        # Turn a sentence into a single word
        assert isinstance(sent, str)

        return [word for word in sent]

    def get_ner(self, sentence):
        """
            NER function.
            Returns two forms of tags
            The first is the triple form (tags,start,end)
            The second is the list form, which marks the ner label of each word such as
            周小明去玩
            ['Nh', 'Nh', 'Nh', 'O', 'O']
        """
        if isinstance(sentence, list):
            # Turn the list into sentence
            tmp = ''
            for word in sentence:
                tmp += word
            sentence = tmp
        if self.__ner is None:
            ltp = LTP()
            seg, hidden = ltp.seg([sentence])
            seg = seg[0]
            ner = ltp.ner(hidden)
            ner = ner[0]
            self.__ner = ner
            self.__seg = seg
        ner_label = len(sentence) * ['O']

        for i in range(len(self.__ner)):
            tag, start, end = self.__ner[i]
            tmp = 0
            for j in range(start):
                tmp += len(self.__seg[j])
            start = tmp
            tmp = 0
            for j in range(end + 1):
                tmp += len(self.__seg[j])
            end = tmp
            self. __ner[i] = (tag, start, end - 1)
            for j in range(start, end):
                ner_label[j] = tag
            self.__ner_list = ner_label

        return self.__ner, ner_label

    def get_pos_tag(self, sentence):
        """
        pos tag function.
        Returns the triple form (tags,start,end)
        """
        if isinstance(sentence, list):
            # Turn the list into sentence
            tmp = ''
            for word in sentence:
                tmp += word
            sentence = tmp
        if self.__pos_tag is None:
            # get pos tag
            ltp = LTP()
            seg, hidden = ltp.seg([sentence])
            pos = ltp.pos(hidden)
            seg = seg[0]
            pos = pos[0]
            pos_tag = []
            cnt = 0
            for tag in range(len(pos)):
                pos_tag.append([pos[tag], cnt, cnt + len(seg[tag]) - 1])
                cnt += len(seg[tag])
            self.__pos_tag = pos_tag

        return self.__pos_tag


cws_text_processor = CWSTextProcessor()


if __name__ == "__main__":
    x = '小明想要去上海'
    processor = CWSTextProcessor()
    print(processor.get_ner(x))
    print(processor.get_pos_tag(x))
