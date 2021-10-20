import os
import random
import time


def CoNLL_f1(sent_list, pred_list, gold_list, script_path):
    fn_out = 'eval_%04d.txt' % random.randint(0, 10000)
    if os.path.isfile(fn_out):
        os.remove(fn_out)

    text_file = open(fn_out, mode='w', encoding="utf-8")
    for i, words in enumerate(sent_list):
        tags_1 = gold_list[i]
        tags_2 = pred_list[i]
        for j, word in enumerate(words):
            tag_1 = tags_1[j]
            tag_2 = tags_2[j]
            text_file.write('%s %s %s\n' % (word, tag_1, tag_2))
        text_file.write('\n')
    text_file.close()

    cmd = 'perl %s < %s' % (os.path.join('.', script_path), fn_out)
    msg = '\nStandard CoNNL perl script (author: Erik Tjong Kim Sang <erikt@uia.ua.ac.be>, version: 2004-01-26):\n'
    msg += ''.join(os.popen(cmd).readlines())
    time.sleep(1.0)
    if fn_out.startswith('eval_') and os.path.exists(fn_out):
        os.remove(fn_out)
    return float(msg.split('\n')[3].split(':')[-1].strip()) * 0.01
