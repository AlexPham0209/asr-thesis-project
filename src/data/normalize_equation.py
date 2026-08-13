#!/usr/bin/env python3
import sys
import re
from pylatexenc.latexwalker import (
    LatexWalker,
    LatexCharsNode,
    LatexGroupNode,
    LatexMacroNode,
    LatexMathNode,
    LatexEnvironmentNode,
    LatexSpecialsNode,
)


class NormalizeFormula:
    def __init__(self):
        self.norm_str = []

    def build_expression(self, node_list):
        """Recursively traverses AST node list and renders normalized LaTeX."""
        for node in node_list:
            self.build_group(node)

    def build_group(self, node):
        """Dispatches nodes to their respective handler based on type."""
        if isinstance(node, LatexCharsNode):
            # Characters / Raw text / Operators
            self.norm_str.append(node.chars)

        elif isinstance(node, LatexGroupNode):
            # ordgroup equivalent: { ... }
            self.norm_str.append("{")
            self.build_expression(node.nodelist)
            self.norm_str.append("}")

        elif isinstance(node, LatexMacroNode):
            macro_name = node.macroname

            # Handle \rm and legacy font switches -> \mathrm{...}
            if macro_name in ["rm", "font"]:
                self.norm_str.append(f"\\mathrm{{")
                if node.nodeargd and node.nodeargd.argnlist:
                    for arg in node.nodeargd.argnlist:
                        if arg:
                            self.build_group(arg)
                self.norm_str.append("}")

            # Handle fractions and binomials (\genfrac equivalent)
            elif macro_name in ["frac", "binom"]:
                self.norm_str.append(f"\\{macro_name}")
                if node.nodeargd and node.nodeargd.argnlist:
                    for arg in node.nodeargd.argnlist:
                        if arg:
                            self.build_group(arg)

            # Handle square roots (\sqrt)
            elif macro_name == "sqrt":
                self.norm_str.append("\\sqrt")
                if node.nodeargd and node.nodeargd.argnlist:
                    # pylatexenc separates optional args [...] and mandatory args {...}
                    for arg in node.nodeargd.argnlist:
                        if arg:
                            self.build_group(arg)

            # Handle accents (\hat, \bar, \vec, etc.)
            elif macro_name in ["hat", "bar", "vec", "tilde", "dot", "ddot"]:
                self.norm_str.append(f"\\{macro_name}")
                if node.nodeargd and node.nodeargd.argnlist:
                    for arg in node.nodeargd.argnlist:
                        if arg:
                            self.build_group(arg)

            # Default command rendering
            else:
                self.norm_str.append(f"\\{macro_name}")
                if node.nodeargd and node.nodeargd.argnlist:
                    for arg in node.nodeargd.argnlist:
                        if arg:
                            self.build_group(arg)
                else:
                    self.norm_str.append(" ")

        elif isinstance(node, LatexEnvironmentNode):
            # Matrix / Cases / Array / Tabular environments
            env_name = node.environmentname
            self.norm_str.append(f"\\begin{{{env_name}}}")
            self.build_expression(node.nodelist)
            self.norm_str.append(f"\\end{{{env_name}}}")

        elif isinstance(node, LatexMathNode):
            # Math delimiters
            self.norm_str.append("$")
            self.build_expression(node.nodelist)
            self.norm_str.append("$")

        elif isinstance(node, LatexSpecialsNode):
            self.norm_str.append(node.specials_chars)

    def get_result(self) -> str:
        return "".join(self.norm_str)


def preprocess_line(line: str) -> str:
    """Replicates pre-AST cleaning steps from JS script."""
    line = line.strip()

    # Strip leading comment
    if line.startswith("%"):
        line = line[1:]

    # Replace \~ with space
    line = line.replace(r"\~", " ")

    # Strip \> spaces, $ signs, and \label{...} commands
    line = re.sub(r"\\>", " ", line)
    line = line.replace("$", " ")
    line = re.sub(r"\\label\{.*?\}", "", line)

    # Legacy font conversions: {\rm ...} -> \mathrm{...}
    line = re.sub(r"\{\s*\\rm", r"\\mathrm{", line)
    line = re.sub(r"\\rm\{", r"\\mathrm{", line)

    return line


def postprocess_line(norm_str: str) -> str:
    """Replicates post-AST cleaning steps from JS script."""
    norm_str = norm_str.replace("SSSSSS", "$")
    norm_str = norm_str.replace(" S S S S S S", "$")
    norm_str = norm_str.replace(r"{ \@not } =", r"\neq")

    # Final cleanup of labels and multiple spaces
    norm_str = re.sub(r"\\label\s*\{\s*.*?\s*\}", "", norm_str)
    norm_str = re.sub(r"\s+", " ", norm_str).strip()
    return norm_str


def process_line(line: str, mode: str = "normalize"):
    clean_line = preprocess_line(line)
    if not clean_line:
        return

    try:
        walker = LatexWalker(clean_line)
        nodes, _, _ = walker.get_latex_nodes()

        if mode == "tokenize":
            # Tokenize mode prints raw AST nodes representation
            print([str(n) for n in nodes])
        else:
            converter = KaTeXASTConverter()
            converter.build_expression(nodes)
            raw_result = converter.get_result()
            final_result = postprocess_line(raw_result)
            print(final_result)

    except Exception as e:
        sys.stderr.write(f"Error processing line: {clean_line}\n")
        sys.stderr.write(f"{e}\n")
        print("")
