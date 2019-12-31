#!/usr/bin/env bash

# pandoc --pdf-engine=xelatex --highlight-style breezedark -V geometry:"top=2cm, bottom=1.5cm, left=2cm, right=2cm" --toc -N outline.md -o outline.pdf

pandoc --pdf-engine=xelatex --highlight-style breezedark --toc -N -H head.tex outline.md -o outline.pdf
