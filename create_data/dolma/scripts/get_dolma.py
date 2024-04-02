OUT_DIRECTORY = "/data/jiasen/dolma/data"

# URLs for cc_en_head
cc_en_head_base_url = "https://huggingface.co/datasets/allenai/dolma/resolve/main/data/common-crawl/cc_en_head/cc_en_head-"
cc_en_head_url_list = [f"{cc_en_head_base_url}{str(i).zfill(4)}.json.gz\n  dir={OUT_DIRECTORY}/cc_en_head\n  out=cc_en_head-{str(i).zfill(4)}.json.gz" for i in range(612)]

# URLs for cc_en_middle
cc_en_middle_base_url = "https://huggingface.co/datasets/allenai/dolma/resolve/main/data/common-crawl/cc_en_middle/cc_en_middle-"
cc_en_middle_url_list = [f"{cc_en_middle_base_url}{str(i).zfill(4)}.json.gz\n  dir={OUT_DIRECTORY}/cc_en_middle\n  out=cc_en_middle-{str(i).zfill(4)}.json.gz" for i in range(777)]

# URLs for cc_en_tail
cc_en_tail_base_url = "https://huggingface.co/datasets/allenai/dolma/resolve/main/data/common-crawl/cc_en_tail/cc_en_tail-"
cc_en_tail_url_list = [f"{cc_en_tail_base_url}{str(i).zfill(4)}.json.gz\n  dir={OUT_DIRECTORY}/cc_en_tail\n  out=cc_en_tail-{str(i).zfill(4)}.json.gz" for i in range(1493)]

# URLs for s2_v3
s2_v3_base_url = "https://huggingface.co/datasets/allenai/dolma/resolve/main/data/peS2o/s2_v3-"
s2_v3_url_list = [f"{s2_v3_base_url}{str(i).zfill(4)}.json.gz\n  dir={OUT_DIRECTORY}/peS2o\n  out=s2_v3-{str(i).zfill(4)}.json.gz" for i in range(42)]

# URLs for The Stack
LANG_TO_FILES = {'lasso': 1, 'nsis': 1, 'literate-agda': 1, 'metal': 1, 'xojo': 1, 'max': 8, 'jupyter-notebook': 101, 'asp': 7, 'elixir': 14, 'html+erb': 19, 'julia': 22, 'dart': 63, 'ragel-in-ruby-host': 1, 'api-blueprint': 1, 'gams': 1, 'tex': 71, 'xml': 101, 'smalltalk': 17, 'cmake': 11, 'piglatin': 1, "cap'n-proto": 1, 'common-lisp': 21, 'stylus': 3, 'typescript': 101, 'jflex': 1, 'factor': 1, 'arc': 1, 'parrot-internal-representation': 1, 'aspectj': 1, 'go': 101, 'urweb': 1, 'dns-zone': 1, 'purebasic': 1, 'toml': 15, 'erlang': 11, 'hy': 1, 'component-pascal': 2, 'oz': 1, 'opa': 1, 'handlebars': 10, 'gas': 15, 'less': 17, 'gnuplot': 15, 'harbour': 1, 'vhdl': 16, 'octave': 1, 'powershell': 21, 'clips': 1, 'fish': 1, 'prolog': 1, 'sparql': 1, 'objective-j': 1, 'scaml': 1, 'twig': 20, 'gettext-catalog': 101, 'purescript': 2, 'vala': 1, 'gosu': 1, 'apacheconf': 1, 'xc': 1, 'lean': 3, 'mako': 1, 'r': 4, 'unrealscript': 1, 'solidity': 21, 'pike': 1, 'cartocss': 1, 'maple': 1, 'graphql': 3, 'unity3d-asset': 101, 'swift': 101, 'dockerfile': 13, 'digital-command-language': 1, 'scala': 83, 'sqf': 2, 'logtalk': 1, 'coq': 1, 'shellsession': 1, 'befunge': 1, 'nu': 1, 'ecere-projects': 1, 'zimpl': 1, 'shen': 1, 'golo': 1, 'web-ontology-language': 12, 'sas': 2, 'uno': 1, 'livescript': 1, 'literate-haskell': 1, 'clojure': 8, 'perl6': 1, 'zig': 3, 'liquid': 2, 'ec': 1, 'blitzbasic': 1, 'sql': 101, 'http': 2, 'xproc': 1, 'kit': 1, 'textile': 1, 'netlinx': 1, 'propeller-spin': 1, 'cython': 5, 'realbasic': 1, 'dogescript': 1, 'llvm': 9, 'pawn': 1, 'groff': 40, 'html+django': 3, 'csound': 1, 'd': 1, 'agda': 2, 'css': 101, 'yacc': 7, 'robotframework': 1, 'kotlin': 101, 'grace': 1, 'abap': 2, 'blitzmax': 1, 'webassembly': 3, 'ampl': 1, 'postscript': 16, 'nit': 1, 'gentoo-eclass': 1, 'xpages': 1, 'linker-script': 2, 'yang': 3, 'jade': 4, 'standard-ml': 6, 'javascript': 101, 'moonscript': 1, 'mtml': 1, 'saltstack': 1, 'freemarker': 5, 'ston': 1, 'html+eex': 1, 'xs': 1, 'c++': 101, 'matlab': 1, 'm4': 2, 'xbase': 1, 'perl': 37, 'emacs-lisp': 7, 'bison': 1, 'slim': 2, 'grammatical-framework': 1, 'rdoc': 1, 'nix': 10, 'clean': 1, 'module-management-system': 1, 'nimrod': 6, 'raml': 1, 'forth': 1, 'squirrel': 1, 'alloy': 1, 'opencl': 3, 'c': 101, 'sass': 4, 'eiffel': 2, 'papyrus': 1, 'html': 109, 'java': 101, 'hcl': 14, 'isabelle': 2, 'markdown': 101, 'gentoo-ebuild': 2, 'objdump': 1, 'emberscript': 1, 'text': 101, 'bro': 1, 'opal': 1, 'haskell': 35, 'mupad': 1, 'desktop': 1, 'modelica': 2, 'coldfusion-cfc': 2, 'fantom': 1, 'glsl': 10, 'ocaml': 16, 'nesc': 2, 'scheme': 7, 'crystal': 5, 'tcsh': 1, 'c2hs-haskell': 1, 'idris': 1, 'logos': 4, 'coffeescript': 13, 'g-code': 10, 'sage': 1, 'haml': 4, 'tcl': 7, 'smt': 5, 'ox': 1, 'chuck': 1, 'xquery': 1, 'batchfile': 7, 'pod': 2, 'xtend': 1, 'restructuredtext': 61, 'rmarkdown': 1, 'turtle': 33, 'jsx': 45, 'protocol-buffer': 8, "ren'py": 2, 'diff': 32, 'slash': 1, 'darcs-patch': 1, 'numpy': 1, 'augeas': 1, 'wisp': 1, 'edn': 15, 'ooc': 1, 'bitbake': 2, 'labview': 1, 'inform-7': 1, 'rust': 101, 'creole': 1, 'apl': 1, 'arduino': 11, 'openscad': 2, 'cuda': 9, 'thrift': 1, 'yaml': 101, 'fancy': 1, 'coldfusion': 1, 'python': 101, 'clarion': 1, 'glyph': 1, 'parrot': 1, 'lookml': 1, 'java-server-pages': 19, 'oxygene': 1, 'flux': 1, 'scilab': 1, 'groovy-server-pages': 2, 'rhtml': 1, 'eagle': 52, 'parrot-assembly': 1, 'igor-pro': 1, 'webidl': 1, 'bluespec': 1, 'unified-parallel-c': 1, 'smali': 38, 'haxe': 9, 'ada': 7, 'lua': 48, 'pascal': 21, 'html+php': 6, 'irc-log': 1, 'x10': 1, 'netlogo': 1, 'ioke': 1, 'dm': 1, 'self': 1, 'elm': 5, 'ats': 1, 'brainfuck': 1, 'mask': 1, 'rouge': 1, 'turing': 1, 'lex': 2, 'gap': 1, 'pogoscript': 1, 'kicad': 30, 'io': 1, 'objective-c++': 8, 'qml': 4, 'redcode': 1, 'autoit': 2, 'processing': 4, 'systemverilog': 6, 'gdscript': 5, 'f-sharp': 12, 'fortran': 23, 'monkey': 1, 'c-sharp': 101, 'xslt': 9, 'viml': 6, 'renderscript': 1, 'scss': 84, 'cucumber': 4, 'verilog': 1, 'genshi': 1, 'racket': 1, 'krl': 1, 'actionscript': 10, 'pan': 1, 'cirru': 1, 'chapel': 1, 'pure-data': 2, 'm': 1, 'applescript': 1, 'inno-setup': 1, 'volt': 1, 'myghty': 1, 'groovy': 17, 'ags-script': 1, 'mirah': 1, 'lsl': 1, 'brightscript': 1, 'python-traceback': 1, 'sourcepawn': 2, 'maxscript': 1, 'zephir': 1, 'supercollider': 1, 'mathematica': 20, 'awk': 1, 'autohotkey': 2, 'lfe': 1, 'ruby': 101, 'visual-basic': 20, 'ini': 59, 'red': 1, 'omgrofl': 1, 'idl': 1, 'rebol': 1, 'vue': 101, 'ninja': 2, 'ecl': 1, 'lolcode': 1, 'tea': 1, 'txl': 1, 'smarty': 9, 'vcl': 1, 'php': 101, 'literate-coffeescript': 1, 'click': 1, 'pony': 1, 'mediawiki': 5, 'stata': 5, 'stan': 1, 'nginx': 1, 'asciidoc': 16, 'antlr': 1, 'cobol': 1, 'org': 5, 'latte': 1, 'makefile': 32, 'ceylon': 1, 'graphviz-(dot)': 13, 'lilypond': 1, 'dylan': 1, 'qmake': 1, 'muf': 1, 'j': 1, 'pov-ray-sdl': 1, 'jasmin': 1, 'shell': 73, 'cycript': 1, 'boo': 1, 'hlsl': 2}
stack_base_url = "https://huggingface.co/datasets/allenai/dolma/resolve/main/data/stack-code/"
stack_url_list = []
for lang, num_files in sorted(LANG_TO_FILES.items()):
    for i in range(num_files):
        stack_url_list.append(f"{stack_base_url}{lang}/v3-{str(i).zfill(4)}.json.gz\n  dir={OUT_DIRECTORY}/stack-code/{lang}\n  out=v3-{str(i).zfill(4)}.json.gz")

# URLs for the c4
c4_base_url = "https://huggingface.co/datasets/allenai/dolma/resolve/main/data/c4/c4-"
c4_url_list = [f"{c4_base_url}{str(i).zfill(4)}.json.gz\n  dir={OUT_DIRECTORY}/c4\n  out=c4-{str(i).zfill(4)}.json.gz" for i in range(86)]

# URLs for the gutenberg-books
books_base_url = "https://huggingface.co/datasets/allenai/dolma/resolve/main/data/gutenberg-books/books-"
books_base_url_url_list = [f"{books_base_url}{str(i).zfill(4)}.json.gz\n  dir={OUT_DIRECTORY}/gutenberg-books\n  out=books-{str(i).zfill(4)}.json.gz" for i in range(3)]

# URLs for the wiki-en-simple
wiki_base_url = "https://huggingface.co/datasets/allenai/dolma/resolve/main/data/wiki-en-simple/en_simple_wiki-"
wiki_base_url_url_list = [f"{wiki_base_url}{str(i).zfill(4)}.json.gz\n  dir={OUT_DIRECTORY}/wiki-en-simple\n  out=en_simple_wiki-{str(i).zfill(4)}.json.gz" for i in range(2)]


# Combine all URL lists
all_url_list = cc_en_head_url_list + cc_en_middle_url_list + cc_en_tail_url_list + s2_v3_url_list + stack_url_list + c4_url_list + books_base_url_url_list + wiki_base_url_url_list

out = open("files.txt", "a")
# Print the combined list of URLs
for i, url in enumerate(all_url_list):
    out.write(url + "\n")
