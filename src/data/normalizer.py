import re
from whisper_normalizer.basic import BasicTextNormalizer


def remove_special_characters(s):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:"]'
    return re.sub(chars_to_ignore_regex, "", s).lower()


def normalize_wav2vec2(s):
    return remove_special_characters(s)


def normalize_whisper(s):
    normalizer = BasicTextNormalizer()
    return normalizer(s)

def is_valid_equation_sentence(s):
    unescaped_dollars = re.sub(r"\\\$", "", s)
    return unescaped_dollars.count("$") % 2 == 0

def create_latex_normalizer(normalizer):
    def normalize_latex(s: str) -> str:
        # if not is_valid_equation_sentence(s):
        #     raise ValueError("Unmatched '$' in input string")
        
        math_pattern = r"(\$\$.*?\$\$|(?<!\\)\$.*?(?<!\\)\$)"
        
        tokens = re.split(math_pattern, s, flags=re.DOTALL)

        res = []
        for token in tokens:
            if not token:
                continue

            # Normalize non-math text while keeping math text the same
            if re.fullmatch(math_pattern, token, flags=re.DOTALL):
                res.append(token)
            else:
                res.append(normalizer(token))

        return "".join(res).strip()

    return normalize_latex
