#!/usr/bin/env python

# Author : TG ;; Created : July 9, 2018
import re
import argparse
import sys

# thanks to https://github.com/carpedm20/emoji/
emoji_codes = {169, 174, 8205, 8252, 8265, 8419, 8482, 8505, 8596, 8597, 8598, 8599, 8600, 8601, 8617, 8618, 8986, 8987,
               9000, 9167, 9193, 9194, 9195, 9196, 9197, 9198, 9199, 9200, 9201, 9202, 9203, 9208, 9209, 9210, 9410,
               9642, 9643, 9654, 9664, 9723, 9724, 9725, 9726, 9728, 9729, 9730, 9731, 9732, 9742, 9745, 9748, 9749,
               9752, 9757, 9760, 9762, 9763, 9766, 9770, 9774, 9775, 9784, 9785, 9786, 9792, 9794, 9800, 9801, 9802,
               9803, 9804, 9805, 9806, 9807, 9808, 9809, 9810, 9811, 9824, 9827, 9829, 9830, 9832, 9851, 9855, 9874,
               9875, 9876, 9877, 9878, 9879, 9881, 9883, 9884, 9888, 9889, 9898, 9899, 9904, 9905, 9917, 9918, 9924,
               9925, 9928, 9934, 9935, 9937, 9939, 9940, 9961, 9962, 9968, 9969, 9970, 9971, 9972, 9973, 9975, 9976,
               9977, 9978, 9981, 9986, 9989, 9992, 9993, 9994, 9995, 9996, 9997, 9999, 10002, 10004, 10006, 10013,
               10017, 10024, 10035, 10036, 10052, 10055, 10060, 10062, 10067, 10068, 10069, 10071, 10083, 10084, 10133,
               10134, 10135, 10145, 10160, 10175, 10548, 10549, 11013, 11014, 11015, 11035, 11036, 11088, 11093, 12336,
               12349, 12951, 12953, 65039, 126980, 127183, 127344, 127345, 127358, 127359, 127374, 127377, 127378,
               127379, 127380, 127381, 127382, 127383, 127384, 127385, 127386, 127462, 127463, 127464, 127465, 127466,
               127467, 127468, 127469, 127470, 127471, 127472, 127473, 127474, 127475, 127476, 127477, 127478, 127479,
               127480, 127481, 127482, 127483, 127484, 127485, 127486, 127487, 127489, 127490, 127514, 127535, 127538,
               127539, 127540, 127541, 127542, 127543, 127544, 127545, 127546, 127568, 127569, 127744, 127745, 127746,
               127747, 127748, 127749, 127750, 127751, 127752, 127753, 127754, 127755, 127756, 127757, 127758, 127759,
               127760, 127761, 127762, 127763, 127764, 127765, 127766, 127767, 127768, 127769, 127770, 127771, 127772,
               127773, 127774, 127775, 127776, 127777, 127780, 127781, 127782, 127783, 127784, 127785, 127786, 127787,
               127788, 127789, 127790, 127791, 127792, 127793, 127794, 127795, 127796, 127797, 127798, 127799, 127800,
               127801, 127802, 127803, 127804, 127805, 127806, 127807, 127808, 127809, 127810, 127811, 127812, 127813,
               127814, 127815, 127816, 127817, 127818, 127819, 127820, 127821, 127822, 127823, 127824, 127825, 127826,
               127827, 127828, 127829, 127830, 127831, 127832, 127833, 127834, 127835, 127836, 127837, 127838, 127839,
               127840, 127841, 127842, 127843, 127844, 127845, 127846, 127847, 127848, 127849, 127850, 127851, 127852,
               127853, 127854, 127855, 127856, 127857, 127858, 127859, 127860, 127861, 127862, 127863, 127864, 127865,
               127866, 127867, 127868, 127869, 127870, 127871, 127872, 127873, 127874, 127875, 127876, 127877, 127878,
               127879, 127880, 127881, 127882, 127883, 127884, 127885, 127886, 127887, 127888, 127889, 127890, 127891,
               127894, 127895, 127897, 127898, 127899, 127902, 127903, 127904, 127905, 127906, 127907, 127908, 127909,
               127910, 127911, 127912, 127913, 127914, 127915, 127916, 127917, 127918, 127919, 127920, 127921, 127922,
               127923, 127924, 127925, 127926, 127927, 127928, 127929, 127930, 127931, 127932, 127933, 127934, 127935,
               127936, 127937, 127938, 127939, 127940, 127941, 127942, 127943, 127944, 127945, 127946, 127947, 127948,
               127949, 127950, 127951, 127952, 127953, 127954, 127955, 127956, 127957, 127958, 127959, 127960, 127961,
               127962, 127963, 127964, 127965, 127966, 127967, 127968, 127969, 127970, 127971, 127972, 127973, 127974,
               127975, 127976, 127977, 127978, 127979, 127980, 127981, 127982, 127983, 127984, 127987, 127988, 127989,
               127991, 127992, 127993, 127994, 127995, 127996, 127997, 127998, 127999, 128000, 128001, 128002, 128003,
               128004, 128005, 128006, 128007, 128008, 128009, 128010, 128011, 128012, 128013, 128014, 128015, 128016,
               128017, 128018, 128019, 128020, 128021, 128022, 128023, 128024, 128025, 128026, 128027, 128028, 128029,
               128030, 128031, 128032, 128033, 128034, 128035, 128036, 128037, 128038, 128039, 128040, 128041, 128042,
               128043, 128044, 128045, 128046, 128047, 128048, 128049, 128050, 128051, 128052, 128053, 128054, 128055,
               128056, 128057, 128058, 128059, 128060, 128061, 128062, 128063, 128064, 128065, 128066, 128067, 128068,
               128069, 128070, 128071, 128072, 128073, 128074, 128075, 128076, 128077, 128078, 128079, 128080, 128081,
               128082, 128083, 128084, 128085, 128086, 128087, 128088, 128089, 128090, 128091, 128092, 128093, 128094,
               128095, 128096, 128097, 128098, 128099, 128100, 128101, 128102, 128103, 128104, 128105, 128106, 128107,
               128108, 128109, 128110, 128111, 128112, 128113, 128114, 128115, 128116, 128117, 128118, 128119, 128120,
               128121, 128122, 128123, 128124, 128125, 128126, 128127, 128128, 128129, 128130, 128131, 128132, 128133,
               128134, 128135, 128136, 128137, 128138, 128139, 128140, 128141, 128142, 128143, 128144, 128145, 128146,
               128147, 128148, 128149, 128150, 128151, 128152, 128153, 128154, 128155, 128156, 128157, 128158, 128159,
               128160, 128161, 128162, 128163, 128164, 128165, 128166, 128167, 128168, 128169, 128170, 128171, 128172,
               128173, 128174, 128175, 128176, 128177, 128178, 128179, 128180, 128181, 128182, 128183, 128184, 128185,
               128186, 128187, 128188, 128189, 128190, 128191, 128192, 128193, 128194, 128195, 128196, 128197, 128198,
               128199, 128200, 128201, 128202, 128203, 128204, 128205, 128206, 128207, 128208, 128209, 128210, 128211,
               128212, 128213, 128214, 128215, 128216, 128217, 128218, 128219, 128220, 128221, 128222, 128223, 128224,
               128225, 128226, 128227, 128228, 128229, 128230, 128231, 128232, 128233, 128234, 128235, 128236, 128237,
               128238, 128239, 128240, 128241, 128242, 128243, 128244, 128245, 128246, 128247, 128248, 128249, 128250,
               128251, 128252, 128253, 128255, 128256, 128257, 128258, 128259, 128260, 128261, 128262, 128263, 128264,
               128265, 128266, 128267, 128268, 128269, 128270, 128271, 128272, 128273, 128274, 128275, 128276, 128277,
               128278, 128279, 128280, 128281, 128282, 128283, 128284, 128285, 128286, 128287, 128288, 128289, 128290,
               128291, 128292, 128293, 128294, 128295, 128296, 128297, 128298, 128299, 128300, 128301, 128302, 128303,
               128304, 128305, 128306, 128307, 128308, 128309, 128310, 128311, 128312, 128313, 128314, 128315, 128316,
               128317, 128329, 128330, 128331, 128332, 128333, 128334, 128336, 128337, 128338, 128339, 128340, 128341,
               128342, 128343, 128344, 128345, 128346, 128347, 128348, 128349, 128350, 128351, 128352, 128353, 128354,
               128355, 128356, 128357, 128358, 128359, 128367, 128368, 128371, 128372, 128373, 128374, 128375, 128376,
               128377, 128378, 128391, 128394, 128395, 128396, 128397, 128400, 128405, 128406, 128420, 128421, 128424,
               128433, 128434, 128444, 128450, 128451, 128452, 128465, 128466, 128467, 128476, 128477, 128478, 128481,
               128483, 128488, 128495, 128499, 128506, 128507, 128508, 128509, 128510, 128511, 128512, 128513, 128514,
               128515, 128516, 128517, 128518, 128519, 128520, 128521, 128522, 128523, 128524, 128525, 128526, 128527,
               128528, 128529, 128530, 128531, 128532, 128533, 128534, 128535, 128536, 128537, 128538, 128539, 128540,
               128541, 128542, 128543, 128544, 128545, 128546, 128547, 128548, 128549, 128550, 128551, 128552, 128553,
               128554, 128555, 128556, 128557, 128558, 128559, 128560, 128561, 128562, 128563, 128564, 128565, 128566,
               128567, 128568, 128569, 128570, 128571, 128572, 128573, 128574, 128575, 128576, 128577, 128578, 128579,
               128580, 128581, 128582, 128583, 128584, 128585, 128586, 128587, 128588, 128589, 128590, 128591, 128640,
               128641, 128642, 128643, 128644, 128645, 128646, 128647, 128648, 128649, 128650, 128651, 128652, 128653,
               128654, 128655, 128656, 128657, 128658, 128659, 128660, 128661, 128662, 128663, 128664, 128665, 128666,
               128667, 128668, 128669, 128670, 128671, 128672, 128673, 128674, 128675, 128676, 128677, 128678, 128679,
               128680, 128681, 128682, 128683, 128684, 128685, 128686, 128687, 128688, 128689, 128690, 128691, 128692,
               128693, 128694, 128695, 128696, 128697, 128698, 128699, 128700, 128701, 128702, 128703, 128704, 128705,
               128706, 128707, 128708, 128709, 128715, 128716, 128717, 128718, 128719, 128720, 128721, 128722, 128736,
               128737, 128738, 128739, 128740, 128741, 128745, 128747, 128748, 128752, 128755, 128756, 128757, 128758,
               128759, 128760, 129296, 129297, 129298, 129299, 129300, 129301, 129302, 129303, 129304, 129305, 129306,
               129307, 129308, 129309, 129310, 129311, 129312, 129313, 129314, 129315, 129316, 129317, 129318, 129319,
               129320, 129321, 129322, 129323, 129324, 129325, 129326, 129327, 129328, 129329, 129330, 129331, 129332,
               129333, 129334, 129335, 129336, 129337, 129338, 129340, 129341, 129342, 129344, 129345, 129346, 129347,
               129348, 129349, 129351, 129352, 129353, 129354, 129355, 129356, 129360, 129361, 129362, 129363, 129364,
               129365, 129366, 129367, 129368, 129369, 129370, 129371, 129372, 129373, 129374, 129375, 129376, 129377,
               129378, 129379, 129380, 129381, 129382, 129383, 129384, 129385, 129386, 129387, 129408, 129409, 129410,
               129411, 129412, 129413, 129414, 129415, 129416, 129417, 129418, 129419, 129420, 129421, 129422, 129423,
               129424, 129425, 129426, 129427, 129428, 129429, 129430, 129431, 129472, 129488, 129489, 129490, 129491,
               129492, 129493, 129494, 129495, 129496, 129497, 129498, 129499, 129500, 129501, 129502, 129503, 129504,
               129505, 129506, 129507, 129508, 129509, 129510, 917602, 917603, 917605, 917607, 917612, 917614, 917619,
               917620, 917623, 917631}


def emoji_regex_str(codes=emoji_codes):
    codes = sorted(codes)
    last = codes[0]
    ranges = []

    def utf(code_pt):
        if code_pt < 2**16:
            return f'\\u{code_pt:04X}'
        else:
            return f'\\U{code_pt:08X}'

    for i in range(1, len(codes)):
        if codes[i] != codes[i-1] + 1:
            ranges.append((last, codes[i-1]))
            last = codes[i]
    ranges.append((last, codes[-1]))
    re_parts = []
    for start, end in ranges:
        if start == end:    # single char
            re_parts.append(utf(start))
        else:   # range
            re_parts.append(f'{utf(start)}-{utf(end)}')
    return '([' + ''.join(re_parts) + ']+)'


# use ur'[]' instead of r'[]' for older python
# Note that some emojis are over \uFFFF  so we need UTF-32 (\Uxxxxxxxx) instead of UTF-16 (\uxxxx)
emoji_regex = re.compile(r'([\u00A9\u00AE\u200D\u203C\u2049\u20E3\u2122\u2139\u2194-\u2199\u21A9-\u21AA\u231A-\u231B\u2328\u23CF\u23E9-\u23F3\u23F8-\u23FA\u24C2\u25AA-\u25AB\u25B6\u25C0\u25FB-\u25FE\u2600-\u2604\u260E\u2611\u2614-\u2615\u2618\u261D\u2620\u2622-\u2623\u2626\u262A\u262E-\u262F\u2638-\u263A\u2640\u2642\u2648-\u2653\u2660\u2663\u2665-\u2666\u2668\u267B\u267F\u2692-\u2697\u2699\u269B-\u269C\u26A0-\u26A1\u26AA-\u26AB\u26B0-\u26B1\u26BD-\u26BE\u26C4-\u26C5\u26C8\u26CE-\u26CF\u26D1\u26D3-\u26D4\u26E9-\u26EA\u26F0-\u26F5\u26F7-\u26FA\u26FD\u2702\u2705\u2708-\u270D\u270F\u2712\u2714\u2716\u271D\u2721\u2728\u2733-\u2734\u2744\u2747\u274C\u274E\u2753-\u2755\u2757\u2763-\u2764\u2795-\u2797\u27A1\u27B0\u27BF\u2934-\u2935\u2B05-\u2B07\u2B1B-\u2B1C\u2B50\u2B55\u3030\u303D\u3297\u3299\uFE0F\U0001F004\U0001F0CF\U0001F170-\U0001F171\U0001F17E-\U0001F17F\U0001F18E\U0001F191-\U0001F19A\U0001F1E6-\U0001F1FF\U0001F201-\U0001F202\U0001F21A\U0001F22F\U0001F232-\U0001F23A\U0001F250-\U0001F251\U0001F300-\U0001F321\U0001F324-\U0001F393\U0001F396-\U0001F397\U0001F399-\U0001F39B\U0001F39E-\U0001F3F0\U0001F3F3-\U0001F3F5\U0001F3F7-\U0001F4FD\U0001F4FF-\U0001F53D\U0001F549-\U0001F54E\U0001F550-\U0001F567\U0001F56F-\U0001F570\U0001F573-\U0001F57A\U0001F587\U0001F58A-\U0001F58D\U0001F590\U0001F595-\U0001F596\U0001F5A4-\U0001F5A5\U0001F5A8\U0001F5B1-\U0001F5B2\U0001F5BC\U0001F5C2-\U0001F5C4\U0001F5D1-\U0001F5D3\U0001F5DC-\U0001F5DE\U0001F5E1\U0001F5E3\U0001F5E8\U0001F5EF\U0001F5F3\U0001F5FA-\U0001F64F\U0001F680-\U0001F6C5\U0001F6CB-\U0001F6D2\U0001F6E0-\U0001F6E5\U0001F6E9\U0001F6EB-\U0001F6EC\U0001F6F0\U0001F6F3-\U0001F6F8\U0001F910-\U0001F93A\U0001F93C-\U0001F93E\U0001F940-\U0001F945\U0001F947-\U0001F94C\U0001F950-\U0001F96B\U0001F980-\U0001F997\U0001F9C0\U0001F9D0-\U0001F9E6\U000E0062-\U000E0063\U000E0065\U000E0067\U000E006C\U000E006E\U000E0073-\U000E0074\U000E0077\U000E007F]+)')
social_pat = re.compile(r'^(https?://[^ ]+|@[^@/:\- ]+|#[^# ]+|([;:]-?[()BDPoO83/*|\]])+)')
#                Begin with(  URL   | @handle   | #hash | emoticons+                 )


def is_copy_tok(tok):
    return social_pat.match(tok) or all(ord(code) in emoji_codes for code in tok)


def extract_copy_toks(tok):
    """
    Given a token, extract pieces of it which are to be copied.
    For example a token can be a mixture of alphanumeric and emoji, emoticon
    :param tok:
    :return:
    """

    sub_toks = []
    left = 0   # cursor
    last = 0
    while left < len(tok):
        # check if this is a sequence of emoji
        if ord(tok[left]) in emoji_codes:
            right = left + 1
            while right < len(tok) and ord(tok[right]) in emoji_codes:
                right += 1

            if left > last:  # normal boring tokens
                sub_toks.append((tok[last:left], 0))
            sub_toks.append((tok[left:right], 1))
            last = left = right
        else:
            # check if URL or hashtag starts from here
            match = social_pat.search(tok[left:])
            if match:
                if left > last:  # normal boring tokens
                    sub_toks.append((tok[last:left], 0))
                sub_toks.append((match.group(), 1))
                left += len(match.group())
                last = left
            else:
                # if none of the above them this is just a boring token
                left += 1  # advance the left cursor

    if left > last:  # normal boring tokens
        sub_toks.append((tok[last:left], 0))
    return sub_toks


def filter_copy_toks(text, tokenized=False, placeholder=None):
    translate, copy_toks = [], []
    toks = text.split()
    if tokenized: # one tok get one tag
        toks_tagged = [(tok, is_copy_tok(tok)) for tok in toks]
    else:
        # tok is sub split to look for patterns like emojis inside
        toks_tagged = [(subtok, tag) for tok in toks for subtok, tag in extract_copy_toks(tok)]

    for tok, copy_tag in toks_tagged:
        if copy_tag:
            if placeholder:
                translate.append(placeholder)
            copy_toks.append(tok)
        else:
            translate.append(tok)
    return " ".join(translate), " ".join(copy_toks)


def main(inp, out, tokenized=False, placeholder=None):
    for line in inp:
        line = line.strip()
        keep, off = filter_copy_toks(line, tokenized=tokenized, placeholder=placeholder)
        out.write(f'{keep}\t{off}\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-v', '--info', action='store_true', help='Print info such as emoji regex')
    p.add_argument('-i', '--inp', help='Input file. One sentence per line', default=sys.stdin)
    p.add_argument('-o', '--out', help='Output file. One sentence per line', default=sys.stdout)
    p.add_argument('-t', '--tokenized', action='store_true', help='Text is tokenized, dont subsplit tokens',)
    p.add_argument('-p', '--placeholder', help='Insert this token in the place of removed copy tokens', default=None)
    args = vars(p.parse_args())
    if args.pop('info'):
        print(emoji_regex_str())
    else:
        main(**args)


