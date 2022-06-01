import os

class FileManager():
    def __init__(self, path = os.path.dirname(os.path.abspath(__file__)), file_name_list=["beam", "prob"], beam_size = 5) -> None:
        self.path = path + "/"
        self.beam_size=beam_size
        self.file_name_list = file_name_list

        self.file_list = []

        self.beam_file_list = []
        self.BI_beam_file_list = []
        self.prob_file_list = []

        self.file_open()
    
    def file_open(self):
        for file_name in self.file_name_list:
            for i in range(self.beam_size):
                if file_name == "beam":
                    file_buf = open(self.path + file_name + str(i) + ".txt", 'w', encoding="utf-8-sig")
                    file_bi_buf = open(self.path + "BI_" + file_name + str(i) + ".txt", 'w', encoding="utf-8-sig")
                    self.beam_file_list.append(file_buf)
                    self.BI_beam_file_list.append(file_bi_buf)
                elif file_name == "prob" :
                    file_buf = open(self.path + file_name + str(i) + ".txt", 'w', encoding="utf-8-sig")
                    self.prob_file_list.append(file_buf)
                else :
                    pass
        for f in self.beam_file_list:
            self.file_list.append(f)
        for f in self.BI_beam_file_list:
            self.file_list.append(f)
        for f in self.prob_file_list:
            self.file_list.append(f)            

    def file_close(self):
        for file in self.file_list:
            file.close()

    def make_sentence(self, morphs, tags):
        sentence =[]
        for morph, tag in zip(morphs, tags):
            if morph != "<pad>" and tag != "<pad>":
                sentence.append(morph)
                sentence.append(tag)
            else :
                pass
        return sentence

    def sent_to_beam_write(self, sent, i):
        self.beam_file_list[i].write(sent)
        self.beam_file_list[i].write("\n")

    def beam_write(self, morphs, BIs, i):
        #tags = self.BI_to_tag(BIs)
        tags = self.force_BI_to_tag(BIs)
        sentence = self.make_sentence(morphs, tags)
        sentence = "".join(sentence)
        self.beam_file_list[i].write(sentence)
        self.beam_file_list[i].write("\n")

    def BI_write(self, morphs, BIs, i):
        sentence = self.make_sentence(morphs, BIs)
        sentence = " ".join(sentence)
        self.BI_beam_file_list[i].write(sentence)
        self.BI_beam_file_list[i].write("\n")
    
    def prob_write(self):
        pass
    def BI_to_tag(self, BI):
        bi = BI  
        result_tag = []
        bi_len = len(bi)
        bi.append("")
        for i in range(bi_len):
            if "B-" in bi[i]:
                if bi[i].replace("B-", "I-") == bi[i+1]:
                    result_tag.append("")
                    bi[i+1] = bi[i+1].replace("I-", "B-")
                else :
                    result_tag.append(bi[i].replace("B-", ""))
            elif bi[i] == "/O":
                result_tag.append("")
            elif bi[i] == "/O+":
                result_tag.append("")
            elif "I-" in bi[i]:
                result_tag.append(bi[i] + "<wrong>")
            elif bi[i]=="<pad>":
                result_tag.append("")
            else:
                result_tag.append("/" + bi[i])
        return result_tag

    def BI_to_tag_v2(self, BI:list):
        BI_len = len(BI)
        bi = BI[:]
        result_tag = []

        cur_tag = bi.pop(0)
        comp_tag = ""
        for _ in range(len(bi) ):
            comp_tag = bi.pop(0)

            if "I-" in comp_tag:
                result_tag.append("")
            elif "B-" in comp_tag:
                cur_tag = cur_tag.replace("B-", "")
                cur_tag = cur_tag.replace("I-", "")
                result_tag.append(cur_tag)
                cur_tag = comp_tag
            elif comp_tag == "/O+" or comp_tag == "/O" or comp_tag == "<pad>":
                cur_tag = cur_tag.replace("B-", "")
                cur_tag = cur_tag.replace("I-", "")
                result_tag.append(cur_tag)
                cur_tag = ""
            elif comp_tag == "<unk>":
                cur_tag = cur_tag.replace("B-", "")
                cur_tag = cur_tag.replace("I-", "")
                result_tag.append(cur_tag)
                cur_tag = "/<unk-tag>"

        
        comp_tag = comp_tag.replace("B-", "")
        comp_tag = comp_tag.replace("I-", "")
        result_tag.append(comp_tag)
        if result_tag[-1] == "":
            result_tag.pop()

        assert BI_len == len(result_tag), f"tag wrong\n{BI}\n{result_tag}\n{BI_len}   {len(result_tag)}"

        return result_tag

           

    def force_BI_to_tag(self, BI : list):
        bi = BI
        result_tag = []

        cur_tag = bi.pop(0)
        for _ in range(len(bi) - 1):
            comp_tag = bi.pop(0)
                
            if "I-" in comp_tag:
                result_tag.append("")
            else :
                cur_tag = cur_tag.replace("B-", "")
                if cur_tag == "/O+" or cur_tag =="/O":
                    result_tag.append("")
                else :
                    result_tag.append(cur_tag)
                cur_tag = comp_tag

        return result_tag
